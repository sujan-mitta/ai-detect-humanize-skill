#!/usr/bin/env python3
"""Check that a rewrite kept every fact from the original - and added none.

Usage: python verify_rewrite.py ORIGINAL REWRITE [--json OUT.json]

Works on files (txt/md/pdf/docx) or chunk files. Compares:
  must match  - numbers, percentages, money, years/dates, citations
                ("(Smith, 2020)", "[12]"), URLs, emails, quoted passages,
                code spans, headings
  check       - names and other proper nouns, acronyms (style rewrites may
                legitimately move these, so they're warnings)
  shape       - length ratio and paragraph count

Exit code 1 when a must-match item is missing or a new one was added, so it
can gate a rewrite loop: fix the chunk, run again, repeat until it passes.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extract_text import extract  # noqa: E402

MONTHS = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
SENT_START = re.compile(r"(^|[.!?:]\s+|\n\s*[-*•]?\s*|#+\s*|[\"'(]\s*)$")
STOP_CAPS = {"I", "I'm", "I've", "I'd", "I'll", "The", "A", "An", "This", "That", "These", "Those", "It", "We", "You",
             "He", "She", "They", "But", "And", "Or", "So", "If", "In", "On", "At", "For", "As", "To", "Of"}


UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve "
                                     "thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
SCALES = {"hundred": 100, "thousand": 1000, "million": 10**6, "billion": 10**9}
_NUM = "|".join(list(UNITS) + list(TENS))
NUMWORD_RE = re.compile(rf"\b(?:{_NUM}|a(?= (?:hundred|thousand|million|billion)))"
                        rf"(?:(?:[ -]|(?<=hundred) and |(?<=thousand) and )(?:{_NUM}|hundred|thousand|million|billion))*\b",
                        re.I)


def words_to_digits(text):
    """'ten minutes' -> '10 minutes', 'two million' -> '2000000', so switching between
    words and digits still matches and spelled-out numbers get checked too.
    A bare 'one' is left alone (usually a pronoun: 'no one', 'one of')."""
    def conv(m):
        words = [w for w in re.split(r"[ -]+", m.group(0).lower()) if w != "and"]
        if words == ["one"]:
            return m.group(0)
        total, cur = 0, 0
        for w in words:
            if w == "a":
                cur = 1
            elif w in UNITS:
                cur += UNITS[w]
            elif w in TENS:
                cur += TENS[w]
            elif w == "hundred":
                cur = (cur or 1) * 100
            else:
                total += (cur or 1) * SCALES[w]
                cur = 0
        return str(total + cur)
    return NUMWORD_RE.sub(conv, text)


def normalize_numbers(text):
    text = words_to_digits(text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*(million|billion|thousand)\b",
                  lambda m: str(int(float(m.group(1)) * SCALES[m.group(2).lower()])), text, flags=re.I)
    t = re.sub(r"(\d)\s*(per ?cent|percent)\b", r"\1%", text, flags=re.I)
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)  # 1,000 -> 1000
    return t


def facts(text):
    t = normalize_numbers(text)
    found = {
        "citations": re.findall(r"\([A-Z][A-Za-z'\-]+(?: et al\.?| and [A-Z][A-Za-z'\-]+| & [A-Z][A-Za-z'\-]+)?,? \d{4}[a-z]?(?:[,;] ?(?:p+\. ?)?\d+(?:[-–]\d+)?)?\)"
                                r"|\[\d+(?:[,–-]\s*\d+)*\]", t),
        "urls": re.findall(r"https?://\S+?(?=[)\]>,.;:!?]*(?:\s|$))|www\.\S+?(?=[)\]>,.;:!?]*(?:\s|$))", t),
        "emails": re.findall(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", t),
        "dates": re.findall(rf"\b{MONTHS}\.? \d{{1,2}}(?:st|nd|rd|th)?,? \d{{4}}\b|\b\d{{1,2}} {MONTHS} \d{{4}}\b|\b{MONTHS} \d{{4}}\b"
                            r"|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b", t),
        # pair quote marks in order (short quotes included) so the narration between two
        # lines of dialogue is never mistaken for a quote; then keep the substantial ones
        "quotes": [q for q in re.findall(r"\"([^\"\n]{1,400})\"|“([^”\n]{1,400})”", t)
                   for q in [q[0] or q[1]] if len(q.split()) >= 3],
        "code": re.findall(r"`[^`\n]+`", t),
        "headings": [h.strip() for h in re.findall(r"^#{1,6}\s+(.+)$", t, re.M)],
    }
    # numbers not already part of a date/citation; keep units/symbols attached
    masked = t
    for k in ("citations", "dates", "urls", "emails"):
        for item in found[k]:
            masked = masked.replace(item, " ")
    found["numbers"] = re.findall(r"(?<![\w.])[$£€₹]?(?:\d+(?:\.\d+)?|\.\d+)(?:%|\s?(?:million|billion|thousand|k|m|bn)\b)?", masked)
    # warnings: proper nouns not at sentence start, acronyms
    names = []
    for m in re.finditer(r"\b[A-Z][a-zA-Z'\-]+(?:[ \t]+(?:(?:of|de|van|von|the|and)[ \t]+)?[A-Z][a-zA-Z'\-]+)*", t):
        if SENT_START.search(t[:m.start()][-6:] if m.start() else ""):
            continue
        if m.group(0) not in STOP_CAPS:
            names.append(m.group(0))
    found["names"] = names
    found["acronyms"] = re.findall(r"\b[A-Z]{2,}[0-9]*s?\b", t)
    return found


MUST = ["numbers", "dates", "citations", "urls", "emails", "quotes", "code", "headings"]
CHECK = ["names", "acronyms"]


def compare(orig_text, new_text):
    a, b = facts(orig_text), facts(new_text)
    result = {"missing": {}, "added": {}, "warnings": {}}
    for k in MUST + CHECK:
        ca, cb = Counter(a[k]), Counter(b[k])
        missing = sorted((ca - cb).elements())
        added = sorted((cb - ca).elements())
        if k == "headings":
            # a PDF has no '#' headings but the rebuilt .docx does; compare heading text, not syntax
            flat = lambda s: re.sub(r"\W+", " ", s).strip().lower()
            missing = [h for h in missing if flat(h) not in flat(new_text)]
            added = [h for h in added if flat(h) not in flat(orig_text)]
        if k in MUST:
            if missing:
                result["missing"][k] = missing
            if added:
                result["added"][k] = added
        else:
            # warn only when a name appears nowhere in the other text (moving it to the
            # start of a sentence or out of a bold label is fine)
            gone = sorted(n for n in set(ca) if not re.search(r"\b" + re.escape(n) + r"\b", new_text))
            new = sorted(n for n in set(cb) if not re.search(r"\b" + re.escape(n) + r"\b", orig_text))
            if gone or new:
                result["warnings"][k] = {"gone": gone, "new": new}
    # the sentence each missing/added item came from, so it's easy to find
    result["context"] = {}
    for side, src in (("missing", normalize_numbers(orig_text)), ("added", normalize_numbers(new_text))):
        for vals in result[side].values():
            for v in vals:
                i = src.find(v)
                if i >= 0:
                    start = max(src.rfind(". ", 0, i) + 2, src.rfind("\n", 0, i) + 1)
                    ends = [x for x in (src.find(". ", i), src.find("\n", i)) if x >= 0]
                    result["context"][f"{side}:{v}"] = src[start:min(ends or [len(src)]) + 1].strip()[:200]
    wa, wb = len(orig_text.split()), len(new_text.split())
    pa = len([p for p in re.split(r"\n\s*\n", orig_text) if p.strip()])
    pb = len([p for p in re.split(r"\n\s*\n", new_text) if p.strip()])
    result["shape"] = {"words_before": wa, "words_after": wb, "length_ratio": round(wb / max(1, wa), 3),
                       "paragraphs_before": pa, "paragraphs_after": pb}
    result["passed"] = not result["missing"] and not result["added"]
    return result


def item_lines(r, side):
    out = []
    for k, vals in r[side].items():
        for v in vals:
            ctx = r.get("context", {}).get(f"{side}:{v}")
            out.append(f"- {k}: `{v}`" + (f'  in: "{ctx}"' if ctx else ""))
    return out


def render(r):
    s = r["shape"]
    lines = ["# Rewrite check: " + ("PASSED" if r["passed"] else "FAILED"), "",
             f"Length: {s['words_before']} -> {s['words_after']} words ({s['length_ratio']:.0%})"
             + ("  <- outside 85-115%" if not 0.85 <= s["length_ratio"] <= 1.15 else ""),
             f"Paragraphs: {s['paragraphs_before']} -> {s['paragraphs_after']}", ""]
    if r["missing"]:
        lines.append("## Missing from the rewrite (restore these)")
        lines += item_lines(r, "missing")
        lines.append("")
    if r["added"]:
        lines.append("## New in the rewrite (not in the original - remove unless intended)")
        lines += item_lines(r, "added")
        lines.append("")
    if r["warnings"]:
        lines.append("## Worth a look (names/acronyms that appeared or disappeared)")
        for k, w in r["warnings"].items():
            if w["gone"]:
                lines.append(f"- {k} gone: " + ", ".join(w["gone"][:15]))
            if w["new"]:
                lines.append(f"- {k} new: " + ", ".join(w["new"][:15]))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("original")
    ap.add_argument("rewrite")
    ap.add_argument("--json")
    args = ap.parse_args()
    r = compare(extract(args.original), extract(args.rewrite))
    if args.json:
        Path(args.json).write_text(json.dumps(r, indent=2), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(render(r))
    sys.exit(0 if r["passed"] else 1)


if __name__ == "__main__":
    main()
