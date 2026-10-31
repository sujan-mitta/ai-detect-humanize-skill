#!/usr/bin/env python3
"""Run any dev script with the probe feature injected into cached neural records.
Usage: python dev/run_with_probe.py MODEL.json SCRIPT.py [script args]"""
import json, math, runpy, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "humanize" / "scripts"))
import external_eval as ee  # noqa: E402
import ai_score  # noqa: E402
model, script, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
w = json.load(open(model, encoding="utf-8")).get("probe")
emb = dict(np.load(ROOT / "runs" / "emb_fakespot.npz"))
mean, scale, coef = np.array(w["mean"]), np.array(w["scale"]), np.array(w["coef"])
orig = ee.score
def with_probe(text, rec, mode):
    if mode == "max" and rec.get("neural") is not None and ee.key(text) in emb:
        e = emb[ee.key(text)].astype(np.float64)
        z = w["intercept"] + float(np.sum(coef * (e - mean) / scale))
        rec = {**rec, "neural": {**rec["neural"], "probe": 1 / (1 + math.exp(-max(-30, min(30, z))))}}
    return orig(text, rec, mode)
ee.score = with_probe
ai_score.MODEL_PATH = Path(model); ai_score._MODELS = None
sys.argv = [script] + rest
runpy.run_path(str(ROOT / script), run_name="__main__")
