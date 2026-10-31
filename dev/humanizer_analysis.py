#!/usr/bin/env python3
"""What does an AI actually do when asked to "humanize" text? Measure it.

Usage: python dev/humanizer_analysis.py [--models style,deep,max]

Pairs every dev/corpus/humanized_wild/<rewriter>__<src-folder>__<src-stem>__<style>.txt
(and the skill's own rewrites in dev/corpus/humanized/) with its AI original,
then reports:
  1. Techniques: how often each rewrite technique was applied (feature rose or
     fell beyond a small margin), overall and per request style
  2. Detection: share of rewrites the deployed models still call "likely AI"
Writes dev/runs/humanizer_analysis.json.
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "humanize" / "scripts"))
import ai_score  # noqa: E402

STYLES = {"A": "evade detectors", "B": "humanize", "C": "paraphrase", "D": "casual rewrite",
          "E": "sound like a student", "S": "this skill"}

# technique name, feature, direction, minimum change that counts
TECHNIQUES = [
    ("Adds contractions", "contractions_per_1k", +1, 5),
    ("Adds casual fillers (basically, honestly...)", "fillers_per_1k", +1, 2),
    ("Starts sentences with And/But/So", "conj_start_ratio", +1, 0.03),
    ("Adds first person (I/me/my)", "first_person_per_1k", +1, 5),
    ("Talks to the reader ('you')", "second_person_per_1k", +1, 5),
    ("Adds personal hedges (I think, probably)", "personal_hedges_per_1k", +1, 2),
    ("Adds rhetorical questions", "questions_per_1k", +1, 2),
    ("Adds exclamation marks", "exclaims_per_1k", +1, 2),
    ("Adds ellipses (...)", "ellipses_per_1k", +1, 1),
    ("Fake typos / dropped apostrophes", "dropped_apostrophes_per_1k", +1, 1),
    ("Adds parentheses", "parens_per_1k", +1, 2),
    ("Adds em dashes", "em_dashes_per_1k", +1, 2),
    ("Removes em dashes", "em_dashes_per_1k", -1, 2),
    ("Shorter sentences on average", "mean_sentence_length", -1, 2),
    ("More sentence-length variety", "sentence_length_cv", +1, 0.08),
    ("Forces short/long alternation", "length_alternation", +1, 0.15),
    ("Adds 1-3 word fragments", "fragment_ratio", +1, 0.04),
    ("Fake-casual tics ('Here's the thing', 'Honestly?')", "performed_voice_tics", +1, 1),
    ("Removes stock AI phrases", "stock_hits_per_1k", -1, 3),
    ("Removes Furthermore/Moreover openers", "transition_start_ratio", -1, 0.03),
    ("Removes bold lead-ins", "bold_leads_per_1k", -1, 2),
    ("Breaks up rule-of-three lists", "triplets_per_1k", -1, 2),
    ("Plainer, shorter words", "avg_word_length", -1, 0.12),
    ("Less vocabulary variety (repeats words)", "mattr", -1, 0.02),
]


def pairs():
    corpus = ROOT / "corpus"
    for f in sorted((corpus / "humanized_wild").glob("*.txt")):
        parts = f.stem.split("__")
        if len(parts) < 5:
            continue
        rewriter, folder, style = parts[0], parts[1], parts[-1]
        src = corpus / "ai" / folder / ("__".join(parts[2:-1]) + ".txt")
        if src.exists():
            yield rewriter, style, src, f
    for f in sorted((corpus / "humanized").glob("*.txt")):
        folder, stem = f.stem.split("__", 1)
        src = corpus / "ai" / folder / f"{stem}.txt"
        if src.exists():
            yield "skill", "S", src, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="style")
    args = ap.parse_args()
    kinds = args.models.split(",")
    models = json.loads((ROOT.parent / "humanize" / "scripts" / "model.json").read_text(encoding="utf-8"))

    data = []
    for rewriter, style, src, new in pairs():
        a = ai_score.analyze(src.read_text(encoding="utf-8"))["raw"]
        b_text = new.read_text(encoding="utf-8")
        b = ai_score.analyze(b_text)["raw"]
        row = {"file": new.name, "rewriter": rewriter, "style": style, "used": [], "scores": {}}
        for name, feat, sign, margin in TECHNIQUES:
            if sign * (b[feat] - a[feat]) >= margin:
                row["used"].append(name)
        for kind in kinds:
            extra = None
            if kind != "style":
                import perplexity
                extra = perplexity.features(b_text) or {}
                if kind == "max":
                    import neural
                    extra.update(neural.features(b_text) or {})
            cal = ai_score.model_score(b, extra, kind)
            row["scores"][kind] = cal["score"] if cal else None
        data.append(row)

    print(f"## Techniques used by AI 'humanizers' ({len(data)} rewrites)\n")
    by_style = defaultdict(list)
    for r in data:
        by_style[r["style"]].append(r)
    styles = [s for s in "ABCDES" if s in by_style]
    print("| Technique | All | " + " | ".join(STYLES[s] for s in styles) + " |")
    print("|---|---|" + "---|" * len(styles))
    for name, *_ in TECHNIQUES:
        allp = sum(name in r["used"] for r in data) / len(data)
        cells = [f"{sum(name in r['used'] for r in by_style[s]) / len(by_style[s]):.0%}" for s in styles]
        print(f"| {name} | **{allp:.0%}** | " + " | ".join(cells) + " |")

    print("\n## Still detected as 'likely AI' after humanizing")
    for kind in kinds:
        t_ai = models.get(kind, {}).get("thresholds", {}).get("ai", 60)
        line = []
        for s in styles:
            sc = [r["scores"][kind] for r in by_style[s] if r["scores"].get(kind) is not None]
            if sc:
                line.append(f"{STYLES[s]} {sum(x >= t_ai for x in sc) / len(sc):.0%} (n={len(sc)})")
        allsc = [r["scores"][kind] for r in data if r["scores"].get(kind) is not None]
        print(f"  {kind:5s}: overall {sum(x >= t_ai for x in allsc) / len(allsc):.0%} | " + " | ".join(line))

    (ROOT / "runs" / "humanizer_analysis.json").write_text(json.dumps(data, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
