#!/usr/bin/env python3
"""Linear probe on fakespot RoBERTa embeddings, trained on all diverse labelled data.
Leave-one-genre-out OOF AUC on the training rows, then held-out tests (external benchmark, Gemini round 1/2,
API texts, human web) at the threshold that flags 1% of the OOF humans. Compare with the shipped max score.
Usage: python dev/probe_eval.py [C]"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "skills" / "humanize" / "scripts"))
import external_eval as ee, calibrate as c, calibrate2 as c2, regression_floor as rf  # noqa: E402
Cval = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
emb = dict(np.load(ROOT / "runs" / "emb_fakespot.npz"))
V = lambda t: emb[ee.key(t)].astype(np.float32)
# exclude the regression-floor pools from training (they must stay unseen)
_, _, _, fam, hum = rf.load_data()
floor_keys = {ee.key(r["text"]) for v in fam.values() for r in v} | {ee.key(r["text"]) for r in hum}
rows = [r for r in c2.load_rows(True, True) if ee.key(r["text"]) in emb and ee.key(r["text"]) not in floor_keys and r["genre"] != "extra"]
X = np.stack([V(r["text"]) for r in rows]); y = np.array([r["label"] for r in rows]); G = np.array([r["genre"] for r in rows])
print(len(rows), "train rows,", int(y.sum()), "AI,", len(set(G)), "genres", flush=True)
sc = StandardScaler().fit(X); Xs = sc.transform(X)
oof = np.zeros(len(y))
for g in sorted(set(G)):
    te = G == g
    m = LogisticRegression(C=Cval, max_iter=400, class_weight="balanced").fit(Xs[~te], y[~te])
    oof[te] = m.predict_proba(Xs[te])[:, 1]
print(f"LOGO OOF AUC {roc_auc_score(y, oof):.3f}; human flagged at 0.5: {np.mean(oof[y==0] >= .5):.1%}, AI caught {np.mean(oof[y==1] >= .5):.1%}")
t1 = np.quantile(oof[y == 0], .99); t5 = np.quantile(oof[y == 0], .95)
print(f"OOF AI recall at 1% human FP {np.mean(oof[y==1] > t1):.0%}, at 5% {np.mean(oof[y==1] > t5):.0%}")
full = LogisticRegression(C=Cval, max_iter=400, class_weight="balanced").fit(Xs, y)
P = lambda ts: full.predict_proba(sc.transform(np.stack([V(t) for t in ts])))[:, 1]
cache = ee.load_cache()
ext = ee.load_texts(); eh = [t["text"] for t in ext if t["label"] == 0]; ea = [t["text"] for t in ext if t["label"] == 1]
g2 = json.load(open(ROOT / "runs" / "gemini2_rows.json", encoding="utf-8")); api = json.load(open(ROOT / "runs" / "api_rows.json", encoding="utf-8"))
g1 = [r for r in json.load(open(ROOT / "runs" / "gemini_rows.json", encoding="utf-8")) if r["split"] == "test"]
hw = [r["text"] for r in json.load(open(ROOT / "runs" / "humanweb_rows.json", encoding="utf-8"))]
sets = {"ext AI": ea, "gemini r1 test": [r["text"] for r in g1], "gemini r2 casual": [r["text"] for r in g2 if r["split"] == "casual"],
        "gemini r2 persona": [r["text"] for r in g2 if r["split"] == "persona"], "api general": [r["text"] for r in api if r["source"] == "api"],
        "api science": [r["text"] for r in api if r["source"] == "api_sci"]}
for k, v in fam.items():
    if v: sets[f"fam {k}"] = [r["text"] for r in v]
ph = {"ext human": P(eh), "human web": P(hw), "fresh human": P([r["text"] for r in hum])}
print("\nprobe at the 1%-OOF-FP threshold (humans flagged):  " + " | ".join(f"{k} {np.mean(v > t1):.1%}" for k, v in ph.items()))
print("probe recall:  " + " | ".join(f"{k} {np.mean(P(v) > t1):.0%}" for k, v in sets.items()))
