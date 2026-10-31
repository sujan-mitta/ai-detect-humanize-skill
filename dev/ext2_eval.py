#!/usr/bin/env python3
"""Per-generator recall / human false positives of a model on the ext2 texts (cached features).
Usage: python dev/ext2_eval.py [MODEL.json] [--split a|b]   (split filters the rows evaluated)"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "humanize" / "scripts"))
import external_eval as ee  # noqa: E402
import ai_score  # noqa: E402
args = sys.argv[1:]
split = args[args.index("--split") + 1] if "--split" in args else None
paths = [a for a in args if a.endswith(".json")]
if paths:
    ai_score.MODEL_PATH = Path(paths[0]); ai_score._MODELS = None
rows = json.load(open(ROOT / "runs" / "ext2_rows.json", encoding="utf-8"))
if split:
    rows = [r for r in rows if r["split"] == split]
cache = ee.load_cache()
thr = ai_score.load_models()["max"]["thresholds"]["ai"] / 100
by = {}
for r in rows:
    by.setdefault((r["label"], r["generator"]), []).append(ee.score(r["text"], cache[ee.key(r["text"])], "max") or 0)
ai_all, hu_all = [], []
for (lab, g), s in sorted(by.items()):
    s = np.array(s); (ai_all if lab else hu_all).extend(s)
    print(f"{'AI ' if lab else 'HUM'} {g:34s} n={len(s):2d} verdict {np.mean(s >= thr):4.0%}  >=60 {np.mean(s >= .6):4.0%}")
a, h = np.array(ai_all), np.array(hu_all)
print(f"ALL AI n={len(a)} verdict {np.mean(a >= thr):.1%} >=60 {np.mean(a >= .6):.1%} | ALL human n={len(h)} verdict {np.mean(h >= thr):.1%} >=60 {np.mean(h >= .6):.1%}")
