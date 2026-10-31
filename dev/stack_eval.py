#!/usr/bin/env python3
"""Does the fakespot-embedding probe add to the max-tier model? LOGO comparison on rows with deep+neural features.
probe = out-of-fold (leave-one-genre-out) probability of a logistic probe on fakespot embeddings."""
import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "skills" / "humanize" / "scripts"))
import external_eval as ee, calibrate as c, calibrate2 as c2, regression_floor as rf  # noqa: E402
emb = dict(np.load(ROOT / "runs" / "emb_fakespot.npz"))
rows = c.extract_features(c2.load_rows(True, True), True)
c2.attach_modern_features(rows)
rows = [r for r in rows if r["genre"] != "extra"]
_, _, _, fam, hum = rf.load_data()
floor_keys = {ee.key(r["text"]) for v in fam.values() for r in v} | {ee.key(r["text"]) for r in hum}
rows = [r for r in rows if ee.key(r["text"]) in emb and ee.key(r["text"]) not in floor_keys]
y = np.array([r["label"] for r in rows]); G = np.array([r["genre"] for r in rows])
X = np.stack([emb[ee.key(r["text"])].astype(np.float32) for r in rows]); Xs = StandardScaler().fit_transform(X)
oof = np.zeros(len(y))
for g in sorted(set(G)):
    te = G == g
    oof[te] = LogisticRegression(C=0.05, max_iter=400, class_weight="balanced").fit(Xs[~te], y[~te]).predict_proba(Xs[te])[:, 1]
for r, p in zip(rows, oof):
    if r.get("neural"):
        r["neural"] = {**r["neural"], "probe": float(p)}
sub = [r for r in rows if r.get("deep") and r.get("neural")]
ys = np.array([r["label"] for r in sub]); Gs = np.array([r["genre"] for r in sub]); Gr = np.array([r.get("group", r["id"]) for r in sub])
print(len(sub), "max-tier rows,", int(ys.sum()), "AI,", len(set(Gs)), "genres", flush=True)
c.NEURAL_FEATURES = list(c.NEURAL_FEATURES)
base = c.STYLE_FEATURES + c.DEEP_FEATURES + c.NEURAL_FEATURES
import calibrate2
calibrate2.template_features(sub, [])
c.GROUPS, c.GENRES = Gr, Gs
def run(names, label):
    p = c.cv_predict_gb(c.matrix(sub, names), ys, names)
    h = p[ys == 0]
    out = [f"{label:14s} AUC {roc_auc_score(ys, p):.3f}"]
    for fp in (0.01, 0.02, 0.05):
        out.append(f"recall@{fp:.0%}FP {np.mean(p[ys == 1] > np.quantile(h, 1 - fp)):.1%}")
    by = {g: np.mean(p[(ys == 0) & (Gs == g)] > np.quantile(h, .98)) for g in sorted(set(Gs[ys == 0]))}
    print(" | ".join(out)); print("   human flagged at the 2%-FP cut by genre: " + " ".join(f"{g}:{v:.0%}" for g, v in by.items() if v > 0))
    return p
run(base, "baseline")
run(base + ["probe"], "with probe")
