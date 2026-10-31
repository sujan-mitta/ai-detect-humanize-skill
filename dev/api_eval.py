#!/usr/bin/env python3
"""Recall of a model on the API-generated texts, by model and by prompt style (cached features).
Usage: python dev/api_eval.py [MODEL.json] [--exclude-models a,b]   (rows of excluded models are skipped)"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "humanize" / "scripts"))
import external_eval as ee  # noqa: E402
import ai_score  # noqa: E402
args = sys.argv[1:]
ex = set(args[args.index("--exclude-models") + 1].split(",")) if "--exclude-models" in args else set()
only = set(args[args.index("--only-models") + 1].split(",")) if "--only-models" in args else None
paths = [a for a in args if a.endswith(".json")]
if paths:
    ai_score.MODEL_PATH = Path(paths[0]); ai_score._MODELS = None
rows = json.load(open(ROOT / "runs" / "api_rows.json", encoding="utf-8"))
cache = ee.load_cache()
thr = ai_score.load_models()["max"]["thresholds"]["ai"] / 100
by, bystyle = {}, {}
for r in rows:
    if r["generator"] in ex or (only and r["generator"] not in only):
        continue
    k = ee.key(r["text"])
    if k not in cache:
        continue
    s = (ee.score(r["text"], cache[k], "max") or 0)
    by.setdefault(r["generator"], []).append(s); bystyle.setdefault(r["split"], []).append(s)
allv = []
for title, d in (("by model", by), ("by prompt style", bystyle)):
    print(title)
    for k, v in sorted(d.items()):
        v = np.array(v); allv += list(v) if title == "by model" else []
        print(f"  {k:24s} n={len(v):3d} verdict {np.mean(v >= thr):4.0%}  >=60 {np.mean(v >= .6):4.0%}  mean {v.mean()*100:4.0f}")
a = np.array(allv)
print(f"ALL n={len(a)} verdict {np.mean(a >= thr):.1%}  >=60 {np.mean(a >= .6):.1%}")
