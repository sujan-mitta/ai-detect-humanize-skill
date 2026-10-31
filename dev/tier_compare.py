#!/usr/bin/env python3
"""Style / deep / max tier recall and human false positives of two model files on the held-out sets (cached features).
Usage: python dev/tier_compare.py OLD.json NEW.json"""
import json, math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "ai-detect-humanize" / "scripts"))
import external_eval as ee  # noqa: E402
import ai_score  # noqa: E402
import regression_floor as rf  # noqa: E402
cache = ee.load_cache()
ext = ee.load_texts()
sets = {"ext AI": [t["text"] for t in ext if t["label"] == 1], "ext human": [t["text"] for t in ext if t["label"] == 0]}
g1 = json.load(open(ROOT / "runs" / "gemini_rows.json", encoding="utf-8")); g2 = json.load(open(ROOT / "runs" / "gemini2_rows.json", encoding="utf-8"))
api = json.load(open(ROOT / "runs" / "api_rows.json", encoding="utf-8")); hw = json.load(open(ROOT / "runs" / "humanweb_rows.json", encoding="utf-8"))
sets.update({"gemini r1 test": [r["text"] for r in g1 if r["split"] == "test"], "gemini r2": [r["text"] for r in g2], "api texts": [r["text"] for r in api],
             "human web": [r["text"] for r in hw]})
for k, v in rf.load_data()[3].items():
    sets[f"fam {k}"] = [r["text"] for r in v]
sets["fresh human"] = [r["text"] for r in rf.load_data()[4]]
HUMAN = {"ext human", "human web", "fresh human"}
for path in sys.argv[1:3]:
    rf._enable_probe(path); ai_score.MODEL_PATH = Path(path); ai_score._MODELS = None
    models = ai_score.load_models()
    print(f"\n=== {Path(path).name}")
    for tier in ("style", "deep", "max"):
        thr = models[tier]["thresholds"]["ai"] / 100
        out = []
        for k, ts in sets.items():
            ok = [t for t in ts if ee.key(t) in cache and (tier == "style" or (cache[ee.key(t)].get("deep") and (tier != "max" or cache[ee.key(t)].get("neural"))))]
            s = np.array([ee.score(t, cache[ee.key(t)], tier) or 0 for t in ok])
            if len(s):
                out.append(f"{k} {np.mean(s >= thr):.0%}")
        print(f"  {tier:5s} (thr {thr:.2f}) flagged: " + " | ".join(out))
