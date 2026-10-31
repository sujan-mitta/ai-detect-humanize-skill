#!/usr/bin/env python3
"""False-positive rate of the shipped model on human genres it never trained on.
Usage: python dev/newgenre_eval.py [--mode max|deep|style]"""
import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "plugin" / "skills" / "humanize" / "scripts"))
import ai_score  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="max")
a = ap.parse_args()
rows = [json.loads(l) for l in (ROOT / "corpus" / "newgenre_human.jsonl").read_text(encoding="utf-8").splitlines()]
by = collections.defaultdict(list)
for r in rows:
    s = ai_score.build(r["text"], deep=a.mode != "style", max_mode=a.mode == "max")["score"]
    by[r["genre"]].append(s)
    print(r["genre"], r["source"][:30], s, flush=True)
print(f"\nhuman texts from UNSEEN genres, {a.mode} mode (flagged = score >= 60)")
for g, v in by.items():
    print(f"  {g:8s} n={len(v)} flagged {sum(x >= 60 for x in v) / len(v):.0%} | read as human {sum(x < 40 for x in v) / len(v):.0%} | mean {statistics.mean(v):.0f}")
allv = [x for v in by.values() for x in v]
print(f"  ALL      n={len(allv)} flagged {sum(x >= 60 for x in allv) / len(allv):.0%}")
