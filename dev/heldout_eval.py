#!/usr/bin/env python3
"""End-to-end humanizer test on texts the model never trained on.

dev/corpus/heldout/ai/<genre>__NN.txt           fresh AI texts (never in any training set)
dev/corpus/heldout/humanized/<same name>        rewrites made by agents following the skill
Reports (max mode): score before -> after, verdicts, verify_rewrite result, length ratio.
Usage: python dev/heldout_eval.py [--dir heldout/humanized] [--mode max]"""
import argparse
import collections
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "skills" / "humanize" / "scripts"))
import ai_score  # noqa: E402
import verify_rewrite  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="humanized")
    ap.add_argument("--mode", default="max")
    a = ap.parse_args()
    base = ROOT / "corpus" / "heldout"
    rows, by = [], collections.defaultdict(list)
    for f in sorted((base / "ai").glob("*.txt")):
        h = base / a.dir / f.name
        if not h.exists():
            continue
        o, n = f.read_text(encoding="utf-8"), h.read_text(encoding="utf-8")
        so = ai_score.build(o, deep=True, max_mode=a.mode == "max")["score"]
        sn = ai_score.build(n, deep=True, max_mode=a.mode == "max")["score"]
        ratio = len(n.split()) / max(1, len(o.split()))
        rows.append((f.stem, so, sn, ratio))
        by[f.stem.split("__")[0]].append((so, sn))
        print(f"{f.stem:14s} {so:3.0f} -> {sn:3.0f}  len {ratio:.0%}", flush=True)
    print(f"\n{len(rows)} texts ({a.mode} mode, shown scale: >=60 likely AI, <40 likely human)")
    print(f"mean score {statistics.mean(r[1] for r in rows):.0f} -> {statistics.mean(r[2] for r in rows):.0f}")
    print(f"read as AI (>=60): {sum(r[1] >= 60 for r in rows)} -> {sum(r[2] >= 60 for r in rows)} | "
          f"read as human (<40): {sum(r[1] < 40 for r in rows)} -> {sum(r[2] < 40 for r in rows)}")
    print(f"mean length kept {statistics.mean(r[3] for r in rows):.0%}")
    for g, v in sorted(by.items()):
        print(f"  {g:10s} {statistics.mean(x for x, _ in v):.0f} -> {statistics.mean(y for _, y in v):.0f}")


if __name__ == "__main__":
    main()
