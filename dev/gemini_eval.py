#!/usr/bin/env python3
"""Score Gemini-family texts: dev/corpus/gemini/<model>__<condition>/<genre>__NN.txt (max mode).
Usage: python dev/gemini_eval.py [--mode max|deep|style]"""
import argparse, collections, statistics, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "ai-detect-humanize" / "scripts"))
import ai_score  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="max"); a = ap.parse_args()
by = collections.defaultdict(list)
for f in sorted((ROOT / "corpus" / "gemini").glob("*/*.txt")):
    t = f.read_text(encoding="utf-8")
    if len(t.split()) < 100:
        print("skip (under 100 words):", f); continue
    r = ai_score.build(t, deep=a.mode != "style", max_mode=a.mode == "max")
    by[f.parent.name].append((f.stem, r["score"]))
    print(f"{f.parent.name:28s} {f.stem:16s} {r['score']:3.0f}", flush=True)
print("\nSUMMARY (verdict: >=60 likely AI, <40 likely human)")
for k, v in by.items():
    s = [x for _, x in v]
    print(f"{k:28s} n={len(s):3d} mean {statistics.mean(s):3.0f} | flagged AI (>=60) {sum(x >= 60 for x in s) / len(s):.0%} | read as human (<40) {sum(x < 40 for x in s) / len(s):.0%}")
    g = collections.defaultdict(list)
    for n, x in v: g[n.split("__")[0]].append(x)
    print("   by genre: " + " | ".join(f"{n} {statistics.mean(x):.0f}" for n, x in sorted(g.items())))
