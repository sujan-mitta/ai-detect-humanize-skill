#!/usr/bin/env python3
"""Round-2 Gemini recall per prompt condition for a model file, plus the regression-floor metrics.
Usage: python dev/gemini2_eval.py MODEL.json [MODEL2.json ...]"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "skills" / "humanize" / "scripts"))
import ai_score, external_eval as ee, regression_floor as rf  # noqa: E402

rows = json.load(open(ROOT / "runs" / "gemini2_rows.json", encoding="utf-8"))
cache = ee.load_cache()
data = rf.load_data()
for path in sys.argv[1:]:
    ai_score.MODEL_PATH = Path(path); ai_score._MODELS = None
    thr = ai_score.load_models()["max"]["thresholds"]["ai"] / 100
    out = []
    for cond in ("casual", "persona"):
        s = np.array([ee.score(r["text"], cache[ee.key(r["text"])], "max") or 0 for r in rows if r["split"] == cond])
        out.append(f"{cond}: >=60 {np.mean(s >= .6):.0%} >=verdict {np.mean(s >= thr):.0%}")
    m = rf.metrics(path, data)
    print(f"{Path(path).name:28s} " + " | ".join(out) +
          f" || ext AI {m['ext_ai_verdict']:.1%} ext human>=60 {m['ext_human_60']:.1%} verdict {m['ext_human_verdict']:.1%} "
          f"fresh human>=60 {m['fresh_human_60']:.1%} gem1 {m['gemini_verdict']:.1%} llama3 {m['llama3_verdict']:.0%} gemma {m['gemma_verdict']:.0%}")
    bad = rf.check(m)
    print("   floors:", "ok" if not bad else "; ".join(bad))
