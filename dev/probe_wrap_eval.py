#!/usr/bin/env python3
"""Run the benchmark scripts on a model that has a 'probe' feature, injecting the probe value from cached embeddings
into each text's neural record (the feature cache predates the probe). Usage: python dev/probe_wrap_eval.py MODEL.json"""
import json, math, runpy, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "humanize" / "scripts"))
import external_eval as ee  # noqa: E402
import ai_score  # noqa: E402
MODEL = sys.argv[1]
w = json.load(open(MODEL, encoding="utf-8")).get("probe")
emb = dict(np.load(ROOT / "runs" / "emb_fakespot.npz"))
mean, scale, coef = np.array(w["mean"]), np.array(w["scale"]), np.array(w["coef"])
orig = ee.score
def with_probe(text, rec, mode):
    if w and mode == "max" and rec.get("neural") is not None:
        e = emb[ee.key(text)].astype(np.float64)
        z = w["intercept"] + float(np.sum(coef * (e - mean) / scale))
        rec = {**rec, "neural": {**rec["neural"], "probe": 1 / (1 + math.exp(-max(-30, min(30, z))))}}
    return orig(text, rec, mode)
ee.score = with_probe
ai_score.MODEL_PATH = Path(MODEL); ai_score._MODELS = None
import regression_floor as rf  # noqa: E402
m = rf.metrics(MODEL)
print("== floor metrics")
for k, v in m.items():
    print(f"  {k:22s} {v:.3f}")
bad = rf.check(m); print("FAIL:", bad if bad else "none")
for name in ("api_eval.py", "gemini2_eval.py", "humanweb_eval.py"):
    print("\n==", name)
    sys.argv = [name, MODEL]
    runpy.run_path(str(ROOT / name), run_name="__main__")
