#!/usr/bin/env python3
"""Score = max(model A, model B) for every benchmark script (cached features). Usage: python dev/ensemble_eval.py A.json B.json
Patches external_eval.score so regression_floor / api_eval / gemini2_eval / humanweb_eval see the ensemble."""
import json, runpy, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "skills" / "humanize" / "scripts"))
import external_eval as ee  # noqa: E402
import ai_score  # noqa: E402
A, B = sys.argv[1], sys.argv[2]
sys.argv = sys.argv[:1]
ai_score.MODEL_PATH = Path(A); ai_score._MODELS = None; MA = ai_score.load_models()
ai_score.MODEL_PATH = Path(B); ai_score._MODELS = None; MB = ai_score.load_models()
orig = ee.score
def ens(text, rec, mode):
    ai_score._MODELS = MA; a = orig(text, rec, mode) or 0
    ai_score._MODELS = MB; b = orig(text, rec, mode) or 0
    return max(a, b)
ee.score = ens
ai_score._MODELS = MA
import regression_floor as rf  # noqa: E402
rf.ai_score._MODELS = MA
def metrics_patched(model_path=None, data=None):
    cache, ext, gem, fam, hum = data or rf.load_data()
    import numpy as np
    thr = MA["max"]["thresholds"]["ai"] / 100
    sc = lambda rs: np.array([ens(t["text"], cache[ee.key(t["text"])], "max") for t in rs])
    h, a, g, fh = (sc([t for t in ext if t["label"] == 0]), sc([t for t in ext if t["label"] == 1]), sc(gem), sc(hum))
    m = {"ext_ai_verdict": np.mean(a >= thr), "ext_ai_60": np.mean(a >= .6), "ext_human_verdict": np.mean(h >= thr), "ext_human_60": np.mean(h >= .6),
         "gemini_verdict": np.mean(g >= thr), "gemini_60": np.mean(g >= .6), "fresh_human_60": np.mean(fh >= .6), "fresh_human_verdict": np.mean(fh >= thr)}
    for k, v in fam.items():
        if v:
            s = sc(v); m[f"{k}_verdict"], m[f"{k}_60"] = np.mean(s >= thr), np.mean(s >= .6)
    return m
m = metrics_patched()
print("== floor metrics (ensemble = max of shipped and candidate)")
for k, v in m.items():
    print(f"  {k:22s} {v:.3f}")
bad = rf.check(m); print("FAIL:", bad if bad else "none")
for name in ("api_eval.py", "gemini2_eval.py", "humanweb_eval.py"):
    print("\n==", name)
    runpy.run_path(str(ROOT / name), run_name="__main__")
