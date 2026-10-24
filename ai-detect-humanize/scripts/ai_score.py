#!/usr/bin/env python3
"""Heuristic AI-likeness score for a text (0 = reads human, 100 = reads AI).

Usage:
  python ai_score.py INPUT [--report REPORT.md] [--json OUT.json] [--compare BEFORE.json]

INPUT can be .txt/.md/.pdf/.docx. The score combines stylistic signals that
AI-generated prose tends to show: uniform sentence length (low burstiness),
stock vocabulary and phrases, sentence-initial transitions, rule-of-three
lists, uniform paragraph length, few contractions and heavy em-dash use.

This is a heuristic, not proof of authorship. Formal human writing (legal,
academic methods sections) can score moderately high; casual AI text that
was prompted for a voice can score low.
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extract_text import extract  # noqa: E402

STOCK_WORDS = [
    "delve", "delves", "delving", "tapestry", "testament", "realm", "intricate",
    "intricacies", "multifaceted", "pivotal", "crucial", "robust", "seamless",
    "seamlessly", "leverage", "leveraging", "utilize", "utilizing", "utilization",
    "foster", "fostering", "underscore", "underscores", "showcase", "showcasing",
    "embark", "unlock", "unleash", "harness", "harnessing", "elevate", "empower",
    "empowering", "holistic", "synergy", "paradigm", "nuanced", "meticulous",
    "meticulously", "comprehensive", "ever-evolving", "cutting-edge",
    "game-changer", "game-changing", "vibrant", "bustling", "myriad", "plethora",
    "landscape", "navigate", "navigating", "streamline", "streamlined",
    "transformative", "invaluable", "noteworthy", "commendable", "furthermore",
    "moreover", "additionally", "consequently", "notably", "indispensable",
    "paramount", "endeavor", "endeavour", "bolster", "resonate", "resonates",
]

STOCK_PHRASES = [
    r"it is important to note", r"it'?s important to note", r"it is worth noting",
    r"it'?s worth noting", r"in today'?s (fast-paced|digital|modern|rapidly)",
    r"ever-evolving (landscape|world)", r"plays? an? (crucial|pivotal|vital|key|significant) role",
    r"a wide (range|array|variety) of", r"whether you'?re", r"when it comes to",
    r"not only\b.{1,80}\bbut also", r"serves? as a testament", r"in conclusion",
    r"in summary", r"at the end of the day", r"this highlights the importance",
    r"let'?s dive", r"dive (deep|deeper) into", r"in the realm of", r"stands? as a",
    r"a deep(er)? understanding", r"the (power|importance|significance) of",
    r"navigat\w+ the complexities", r"rich tapestry", r"it'?s not just\b",
    r"isn'?t (just|merely|only) about", r"key takeaways?", r"in essence",
    r"ultimately,", r"overall,", r"by doing so", r"in order to",
    r"(unlock|unleash) (the|its|their|your) (full )?potential",
    r"(paves?|paving) the way", r"(a|an) (unique|valuable) (opportunity|insight)",
]

TRANSITIONS = {
    "furthermore", "moreover", "additionally", "however", "consequently",
    "therefore", "thus", "hence", "notably", "importantly", "ultimately",
    "overall", "indeed", "nevertheless", "nonetheless", "similarly",
    "conversely", "subsequently", "accordingly", "firstly", "secondly",
    "lastly", "finally", "in addition", "in conclusion", "in summary",
    "as a result", "on the other hand", "for instance", "for example",
}

CONTRACTION_RE = re.compile(r"\b\w+'(s|t|re|ve|ll|d|m)\b", re.I)
TRIPLET_RE = re.compile(r"\b[\w-]+(?: [\w-]+)?, [\w-]+(?: [\w-]+)?,? and [\w-]+", re.I)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")

# name, weight, human-readable explanation
SIGNALS = [
    ("burstiness", 0.25, "Sentence length variety (uniform lengths read as AI)"),
    ("stock_language", 0.30, "AI-typical words and phrases per 1,000 words"),
    ("transitions", 0.15, "Sentences opening with Furthermore/Moreover/Additionally..."),
    ("paragraph_uniformity", 0.10, "Paragraphs all about the same length"),
    ("triplets", 0.08, "Rule-of-three lists ('X, Y, and Z')"),
    ("contractions", 0.07, "Few contractions (don't, it's...)"),
    ("em_dashes", 0.05, "Heavy em-dash use"),
]


def clamp(x):
    return max(0.0, min(1.0, x))


def prose_blocks(text):
    """Paragraphs excluding headings, tables and code."""
    blocks = []
    for block in re.split(r"\n\s*\n", text):
        b = block.strip()
        if not b or b.startswith("```") or b.startswith("|"):
            continue
        lines = [l for l in b.splitlines() if not re.match(r"^#{1,6}\s", l)]
        if lines:
            blocks.append(" ".join(l.strip() for l in lines))
    return blocks


def sentences(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\b(e\.g|i\.e|etc|vs|Dr|Mr|Mrs|Ms|Fig|al)\.", lambda m: m.group(0).replace(".", "․"), text)
    parts = re.split(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])", text)
    return [p.replace("․", ".").strip() for p in parts if len(WORD_RE.findall(p)) >= 3]


def stock_hits(text):
    low = text.lower()
    hits = []
    for w in STOCK_WORDS:
        for _ in re.finditer(r"\b" + re.escape(w) + r"\b", low):
            hits.append(w)
    for p in STOCK_PHRASES:
        for m in re.finditer(p, low):
            hits.append(m.group(0))
    return hits


def starts_with_transition(sentence):
    low = sentence.lower().lstrip("\"'( ")
    return any(low.startswith(t + ",") or low.startswith(t + " ") for t in TRANSITIONS)


def analyze(text):
    blocks = prose_blocks(text)
    prose = "\n\n".join(blocks)
    sents = [s for b in blocks for s in sentences(b)]
    words = WORD_RE.findall(prose)
    n_words = max(1, len(words))
    per_k = 1000.0 / n_words

    lengths = [len(WORD_RE.findall(s)) for s in sents]
    if len(lengths) >= 3:
        cv = statistics.pstdev(lengths) / max(1e-9, statistics.mean(lengths))
    else:
        cv = 0.55  # not enough data: neutral
    para_lengths = [len(WORD_RE.findall(b)) for b in blocks if len(WORD_RE.findall(b)) >= 15]
    if len(para_lengths) >= 4:
        pcv = statistics.pstdev(para_lengths) / max(1e-9, statistics.mean(para_lengths))
    else:
        pcv = 0.5

    hits = stock_hits(prose)
    trans = sum(starts_with_transition(s) for s in sents)
    trans_ratio = trans / max(1, len(sents))
    contractions = len(CONTRACTION_RE.findall(prose)) * per_k
    triplets = len(TRIPLET_RE.findall(prose)) * per_k
    dashes = (prose.count("—") + len(re.findall(r"\w -- \w|\w--\w", prose))) * per_k

    raw = {
        "words": len(words),
        "sentences": len(sents),
        "mean_sentence_length": round(statistics.mean(lengths), 1) if lengths else 0,
        "sentence_length_cv": round(cv, 3),
        "paragraph_length_cv": round(pcv, 3),
        "stock_hits_per_1k": round(len(hits) * per_k, 2),
        "transition_start_ratio": round(trans_ratio, 3),
        "contractions_per_1k": round(contractions, 2),
        "triplets_per_1k": round(triplets, 2),
        "em_dashes_per_1k": round(dashes, 2),
    }
    sig = {
        "burstiness": clamp((0.65 - cv) / 0.40),
        "stock_language": clamp(len(hits) * per_k / 12.0),
        "transitions": clamp((trans_ratio - 0.03) / 0.15),
        "paragraph_uniformity": clamp((0.60 - pcv) / 0.45),
        "triplets": clamp(triplets / 6.0),
        "contractions": clamp(1 - contractions / 15.0),
        "em_dashes": clamp(dashes / 8.0),
    }
    score = round(100 * sum(sig[name] * w for name, w, _ in SIGNALS))
    common = {}
    for h in hits:
        common[h] = common.get(h, 0) + 1
    return {
        "score": score,
        "verdict": verdict(score),
        "low_confidence": len(words) < 150,
        "signals": {k: round(v, 3) for k, v in sig.items()},
        "raw": raw,
        "top_stock_terms": sorted(common.items(), key=lambda kv: -kv[1])[:15],
        "_sentences": sents,
    }


def verdict(score):
    if score >= 60:
        return "likely AI"
    if score >= 40:
        return "mixed / possibly AI-assisted"
    return "likely human"


def sections(text, target=350):
    """Split by headings when present, otherwise into ~target-word groups."""
    out, title, buf, count = [], "Start", [], 0
    for block in re.split(r"\n\s*\n", text):
        b = block.strip()
        if not b:
            continue
        m = re.match(r"^#{1,6}\s+(.*)", b)
        if m and "\n" not in b:
            if buf:
                out.append((title, "\n\n".join(buf)))
            title, buf, count = m.group(1).strip()[:60], [], 0
            continue
        buf.append(b)
        count += len(b.split())
        if count >= target * 1.6:
            out.append((title, "\n\n".join(buf)))
            title, buf, count = title + " (cont.)" if "(cont.)" not in title else title, [], 0
    if buf:
        out.append((title, "\n\n".join(buf)))
    # merge tiny sections into the previous one
    merged = []
    for t, body in out:
        if merged and len(body.split()) < 60:
            merged[-1] = (merged[-1][0], merged[-1][1] + "\n\n" + body)
        else:
            merged.append((t, body))
    return merged


def sentence_flags(sents, limit=10):
    scored = []
    for s in sents:
        reasons = []
        hits = stock_hits(s)
        if hits:
            reasons.append("stock: " + ", ".join(sorted(set(hits))[:4]))
        if starts_with_transition(s):
            reasons.append("transition opener")
        if "—" in s:
            reasons.append("em dash")
        if TRIPLET_RE.search(s):
            reasons.append("rule of three")
        weight = len(hits) * 2 + starts_with_transition(s) + ("—" in s) + bool(TRIPLET_RE.search(s))
        if weight >= 2:
            scored.append((weight, s, reasons))
    scored.sort(key=lambda x: -x[0])
    return [{"sentence": s[:300], "reasons": r} for _, s, r in scored[:limit]]


def build(text):
    overall = analyze(text)
    secs = []
    for title, body in sections(text):
        a = analyze(body)
        secs.append({"title": title, "words": a["raw"]["words"], "score": a["score"], "verdict": a["verdict"]})
    overall["sections"] = secs
    overall["flagged_sentences"] = sentence_flags(overall.pop("_sentences"))
    return overall


def render(result, source, before=None):
    lines = [f"# AI-likeness report: {Path(source).name}", ""]
    if before:
        d = result["score"] - before["score"]
        lines.append(f"**Score:** {before['score']} -> **{result['score']}** ({d:+d})  |  "
                     f"{before['verdict']} -> **{result['verdict']}**")
    else:
        lines.append(f"**Score:** {result['score']}/100 - **{result['verdict']}**")
    lines.append(f"Words: {result['raw']['words']}, sentences: {result['raw']['sentences']}"
                 + ("  (short text - low confidence)" if result["low_confidence"] else ""))
    lines += ["", "_Heuristic estimate from writing style. Not proof of authorship._", "",
              "## Signals", "", "| Signal | AI-ness (0-1) | Measured |", "|---|---|---|"]
    measured = {
        "burstiness": f"sentence-length CV {result['raw']['sentence_length_cv']} (human ~0.5-0.9)",
        "stock_language": f"{result['raw']['stock_hits_per_1k']} per 1k words",
        "transitions": f"{result['raw']['transition_start_ratio']:.0%} of sentences",
        "paragraph_uniformity": f"paragraph-length CV {result['raw']['paragraph_length_cv']}",
        "triplets": f"{result['raw']['triplets_per_1k']} per 1k words",
        "contractions": f"{result['raw']['contractions_per_1k']} per 1k words",
        "em_dashes": f"{result['raw']['em_dashes_per_1k']} per 1k words",
    }
    for name, _, desc in SIGNALS:
        bv = f" (was {before['signals'][name]})" if before else ""
        lines.append(f"| {desc} | {result['signals'][name]}{bv} | {measured[name]} |")
    if result["top_stock_terms"]:
        lines += ["", "**Most frequent AI-typical terms:** "
                  + ", ".join(f"{t} ({n})" for t, n in result["top_stock_terms"])]
    if len(result["sections"]) > 1:
        lines += ["", "## Sections", "", "| # | Section | Words | Score | Verdict |", "|---|---|---|---|---|"]
        for i, s in enumerate(result["sections"], 1):
            lines.append(f"| {i} | {s['title']} | {s['words']} | {s['score']} | {s['verdict']} |")
    if result["flagged_sentences"]:
        lines += ["", "## Most AI-sounding sentences", ""]
        for f in result["flagged_sentences"]:
            lines.append(f"- \"{f['sentence']}\"  \n  _{'; '.join(f['reasons'])}_")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--report", help="write a markdown report here")
    ap.add_argument("--json", help="write full results as JSON here")
    ap.add_argument("--compare", help="JSON from an earlier run, to show before/after")
    args = ap.parse_args()

    text = extract(args.input)
    result = build(text)
    before = json.loads(Path(args.compare).read_text(encoding="utf-8")) if args.compare else None
    report = render(result, args.input, before)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(report, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
