#!/usr/bin/env python3
"""GPT-2-features-only legacy head for the --deep tier (no neural models needed). Writes "legacy_deep".
Same recipe as legacy_head.py: train on our rows + benchmark/Gemini rows, rescale so its 1%-human threshold
lands on the deep model's verdict threshold, gate at 80."""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "skills" / "humanize" / "scripts"))
import calibrate as c, external_eval as ee, ai_score, legacy_head as lh  # noqa: E402

NAMES = c.DEEP_FEATURES
ours = [r for r in c.extract_features(c.load_corpus(), True) if r["genre"] != "extra" and r.get("deep")]
ext = [r for r in lh.load_ext()]
for r in ext: r["neural"] = r.get("neural") or {}
allr = ours + ext
c.GROUPS = np.array([r["group"] for r in allr]); c.GENRES = np.array([r["genre"] for r in allr])
y = np.array([r["label"] for r in allr])
X = c.matrix(allr, NAMES)
p = c.cv_predict_gb(X, y, NAMES)
n = len(ours); yo = y[:n]
t_leg = float(np.quantile(p[:n][yo == 0], 0.99))
cur = ai_score.load_models()
th_deep = cur["deep"]["thresholds"]["ai"] / 100
scale = th_deep / max(t_leg, 1e-6)
print(f"LOGO legacy-deep AUC ours+ext {roc_auc_score(y, p):.3f}; ext-only {roc_auc_score(y[n:], p[n:]):.3f}; scale {scale:.2f}")
leg = c.fit_gb(X, y, NAMES); leg["threshold"] = round(t_leg * 100, 2); leg["scale"] = round(scale, 4); leg["gate"] = 80
if "--write" in sys.argv:
    path = ai_score.MODEL_PATH
    m = json.loads(path.read_text(encoding="utf-8")); m["legacy_deep"] = leg
    path.write_text(json.dumps(m, separators=(",", ":")), encoding="utf-8"); print("wrote legacy_deep")
