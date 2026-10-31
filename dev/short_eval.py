#!/usr/bin/env python3
"""Accuracy on short texts: cut held-out texts into 35-99 word chunks of whole sentences and score each.
Humans: external eval humans (COLING dev, RAID) - never trained on. AI: held-out Claude texts, external eval AI,
Gemini round-1 files 04-05. Usage: python dev/short_eval.py [--mode style|max] [--n 200]"""
import argparse, json, random, re, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "skills" / "humanize" / "scripts"))
import external_eval as ee, ai_score  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="style"); ap.add_argument("--n", type=int, default=200); a = ap.parse_args()
random.seed(3)

def chunks(text, lo, hi):
    sents = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    out, cur = [], []
    for s in sents:
        cur.append(s)
        n = sum(len(x.split()) for x in cur)
        if n >= lo:
            if n <= hi: out.append(" ".join(cur))
            cur = []
    return out

hum = [t["text"] for t in ee.load_texts() if t["label"] == 0]
ai = [t["text"] for t in ee.load_texts() if t["label"] == 1]
ai += [f.read_text(encoding="utf-8") for f in sorted((ROOT / "corpus" / "heldout" / "ai").glob("*.txt"))]
ai += [f.read_text(encoding="utf-8") for f in sorted((ROOT / "corpus" / "gemini").glob("*/*__0[45].txt"))]
bins = [(35, 60), (60, 80), (80, 99)]
models = ai_score.load_models(); kind = {"style": "style", "deep": "deep", "max": "max"}[a.mode]
thr = models[kind]["thresholds"]["ai"]
print(f"mode {a.mode}; verdict threshold on model scale {thr}")
for lo, hi in bins:
    H = [c for t in hum for c in chunks(t, lo, hi)]; A = [c for t in ai for c in chunks(t, lo, hi)]
    H = random.sample(H, min(a.n, len(H))); A = random.sample(A, min(a.n, len(A)))
    sc = lambda c: ai_score.build(c, deep=a.mode != "style", max_mode=a.mode == "max")["score"]
    h = np.array([sc(c) for c in H]); x = np.array([sc(c) for c in A])
    y = np.r_[np.zeros(len(h)), np.ones(len(x))]; p = np.r_[h, x]
    print(f"{lo}-{hi} words: n={len(h)}H/{len(x)}A AUC {roc_auc_score(y, p):.3f} | >=60: human {np.mean(h >= 60):.0%} AI {np.mean(x >= 60):.0%} | >=85: human {np.mean(h >= 85):.0%} AI {np.mean(x >= 85):.0%}", flush=True)
