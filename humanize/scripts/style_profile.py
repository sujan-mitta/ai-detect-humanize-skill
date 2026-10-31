#!/usr/bin/env python3
"""Build a voice profile from someone's own writing, and measure a rewrite against it.

Usage:
  python style_profile.py SAMPLE [SAMPLE ...] [-o profile.json]
  python style_profile.py --profile profile.json --check REWRITE

SAMPLE is text the person really wrote (emails, old essays, posts) - the more
the better, ideally 500+ words. The profile lists measurable habits (sentence
length and spread, contractions, punctuation, openers, pet words) and turns
them into concrete targets for the rewrite. --check scores how close a
rewrite is to the profile, metric by metric.
"""
import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ai_score  # noqa: E402
from extract_text import extract  # noqa: E402

STOP = set("""a an the and or but so if of to in on at for with by from as is are was were be been being it its
this that these those i me my we our you your he she they them his her their there here what which who whom
when where why how not no do does did have has had will would can could should may might must than then
just also very about into over out up down more most some any all each other such only own same too s t
""".split())

# metric: (label, how to phrase the target)
METRICS = {
    "mean_sentence_length": "average sentence length (words)",
    "sentence_length_cv": "sentence-length variety (higher = more varied)",
    "contractions_per_1k": "contractions per 1,000 words",
    "first_person_per_1k": "I/me/my per 1,000 words",
    "avg_word_length": "average word length (letters)",
    "mattr": "vocabulary variety",
    "commas_per_sentence": "commas per sentence",
    "em_dashes_per_1k": "em dashes per 1,000 words",
    "semicolons_per_1k": "semicolons per 1,000 words",
    "parens_per_1k": "parentheses per 1,000 words",
    "questions_per_1k": "questions per 1,000 words",
    "exclaims_per_1k": "exclamation marks per 1,000 words",
}


def profile(text):
    a = ai_score.analyze(text)
    raw = a["raw"]
    sents = a["_sentences"]
    lengths = [len(s.split()) for s in sents]
    words = [w.lower() for w in re.findall(r"[A-Za-z']+", text)]
    content = Counter(w for w in words if w not in STOP and len(w) > 3)
    bigrams = Counter(" ".join(p) for p in zip(words, words[1:]) if not (p[0] in STOP and p[1] in STOP))
    openers = Counter(s.split()[0].strip("\"'(").lower() for s in sents if s.split())
    paras = [p for p in re.split(r"\n\s*\n", text) if len(p.split()) > 5]
    return {
        "words": raw["words"],
        "metrics": {k: raw[k] for k in METRICS},
        "sentence_length_range": [min(lengths), max(lengths)] if lengths else [0, 0],
        "short_sentence_share": round(sum(l <= 7 for l in lengths) / max(1, len(lengths)), 3),
        "long_sentence_share": round(sum(l >= 30 for l in lengths) / max(1, len(lengths)), 3),
        "paragraph_sentences": round(statistics.mean(len(ai_score.sentences(p)) or 1 for p in paras), 1) if paras else 0,
        "top_openers": openers.most_common(8),
        "pet_words": [w for w, c in content.most_common(40) if c >= 2][:15],
        "pet_phrases": [b for b, c in bigrams.most_common(60) if c >= 2 and not any(ch.isdigit() for ch in b)][:10],
        "starts_with_and_but": round(sum(o in ("and", "but", "so") for o in (s.split()[0].lower() for s in sents if s.split())) / max(1, len(sents)), 3),
    }


def guidance(p):
    m = p["metrics"]
    g = [f"Average sentence ~{m['mean_sentence_length']:.0f} words, ranging {p['sentence_length_range'][0]}-{p['sentence_length_range'][1]}; "
         f"{p['short_sentence_share']:.0%} short (<=7 words), {p['long_sentence_share']:.0%} long (30+)."]
    g.append(f"Paragraphs of about {p['paragraph_sentences']} sentences.")
    c = m["contractions_per_1k"]
    g.append("Uses contractions freely (don't, it's)." if c > 20 else "Uses some contractions." if c > 6 else "Rarely contracts - keep it formal.")
    fp = m["first_person_per_1k"]
    g.append("Writes in first person a lot." if fp > 25 else "Some first person." if fp > 5 else "Mostly impersonal - little first person.")
    for key, word in (("em_dashes_per_1k", "em dashes"), ("semicolons_per_1k", "semicolons"),
                      ("parens_per_1k", "parentheses"), ("questions_per_1k", "questions"), ("exclaims_per_1k", "exclamation marks")):
        v = m[key]
        if v == 0:
            g.append(f"Never uses {word} - don't add them.")
        elif v > 6:
            g.append(f"Uses {word} often ({v:.0f} per 1,000 words).")
    if p["starts_with_and_but"] >= 0.04:
        g.append(f"Starts {p['starts_with_and_but']:.0%} of sentences with And/But/So.")
    if p["pet_words"]:
        g.append("Favourite words: " + ", ".join(p["pet_words"][:10]) + ".")
    if p["pet_phrases"]:
        g.append("Recurring phrases: " + "; ".join(p["pet_phrases"][:6]) + ".")
    g.append("Common openers: " + ", ".join(f"{w} ({n})" for w, n in p["top_openers"][:6]) + ".")
    return g


def check(prof, text):
    got = profile(text)
    rows = []
    for k, label in METRICS.items():
        target, value = prof["metrics"][k], got["metrics"][k]
        scale = max(abs(target) * 0.35, {"mattr": 0.03, "avg_word_length": 0.25, "sentence_length_cv": 0.1,
                                          "commas_per_sentence": 0.3}.get(k, 2.0))
        rows.append((label, target, value, min(1.0, abs(value - target) / scale)))
    match = round(100 * (1 - statistics.mean(r[3] for r in rows)))
    return match, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", nargs="*")
    ap.add_argument("-o", "--output", help="save the profile as JSON")
    ap.add_argument("--profile", help="saved profile JSON (for --check)")
    ap.add_argument("--check", help="rewrite to compare against the profile")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.check:
        prof = json.loads(Path(args.profile).read_text(encoding="utf-8")) if args.profile else profile(
            "\n\n".join(extract(s) for s in args.samples))
        match, rows = check(prof, extract(args.check))
        print(f"# Voice match: {match}/100\n\n| Metric | Their writing | Rewrite | |\n|---|---|---|---|")
        for label, t, v, d in rows:
            print(f"| {label} | {t} | {v} | {'ok' if d < 0.5 else 'adjust'} |")
        return

    if not args.samples:
        ap.error("give at least one writing sample")
    text = "\n\n".join(extract(s) for s in args.samples)
    prof = profile(text)
    prof["guidance"] = guidance(prof)
    if prof["words"] < 300:
        print(f"NOTE: only {prof['words']} words of sample - the profile will be rough.\n")
    print("# Voice profile\n")
    for line in prof["guidance"]:
        print(f"- {line}")
    if args.output:
        Path(args.output).write_text(json.dumps(prof, indent=2), encoding="utf-8")
        print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
