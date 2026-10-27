#!/usr/bin/env python3
"""Fair comparison of two training sets on the SAME held-out test texts.

Every retrain so far was scored on a test set that changed with the data, so small
differences (2-3 points) were not comparable. Here a fixed share of whole groups (source /
author / original text) is held out; model A trains on the rest of the base corpus, model B on
that plus the extra data. Both are scored on the identical test rows, repeated over several
random splits, and reported with the spread.

Usage: python dev/frozen_compare.py [--splits 5] [--tier max|style]
"base" = everything except the round-v5 additions (techdoc2 + govreport4 human; ai/*6 AI)."""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import calibrate as c  # noqa: E402
import calibrate2  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--splits", type=int, default=5)
ap.add_argument("--tier", default="max")
args = ap.parse_args()


def is_extra(r):
    return r["source"].startswith(("techdoc2:", "govreport4")) or r["source"] in {"ai:sonnet6", "ai:haiku6", "ai:opus6"} \
        or r["id"].split("/")[0] in {"sonnet6", "haiku6", "opus6"}


def tpr_at(p, y, fpr):
    cut = np.quantile(p[y == 0], 1 - fpr)
    return float((p[y == 1] > cut).mean())


def main():
    rows = c.extract_features(c.load_corpus(), args.tier != "style")
    rows = [r for r in rows if r["genre"] != "extra"]
    if args.tier != "style":
        rows = [r for r in rows if r.get("deep") and (args.tier != "max" or r.get("neural"))]
    names = c.STYLE_FEATURES + (c.DEEP_FEATURES + (c.NEURAL_FEATURES if args.tier == "max" else []) if args.tier != "style" else [])
    base = [r for r in rows if not is_extra(r)]
    extra = [r for r in rows if is_extra(r)]
    print(f"{len(base)} base rows, {len(extra)} extra rows ({sum(r['label'] for r in extra)} AI)")
    groups = sorted({r["group"] for r in base})
    res = {"A": [], "B": []}
    for s in range(args.splits):
        rng = np.random.RandomState(100 + s)
        test_g = set(rng.choice(groups, int(0.2 * len(groups)), replace=False))
        test = [r for r in base if r["group"] in test_g]
        trainA = [r for r in base if r["group"] not in test_g]
        trainB = trainA + extra
        yt = np.array([r["label"] for r in test])
        out = {}
        for tag, train in (("A", trainA), ("B", trainB)):
            c.GROUPS = np.array([r["group"] for r in train])
            c.GENRES = np.array([r["genre"] for r in train])
            tr = [dict(r, feat=dict(r["feat"])) for r in train]
            te = [dict(r, feat=dict(r["feat"])) for r in test]
            calibrate2.template_features(tr, te)
            Xtr, Xte = c.matrix(tr, names), c.matrix(te, names)
            ytr = np.array([r["label"] for r in tr])
            gb = c.make_gb(names).fit(Xtr, ytr)
            p = gb.decision_function(Xte)
            fp = {g: None for g in ()}
            res[tag].append({"auc": roc_auc_score(yt, p), "t1": tpr_at(p, yt, 0.01), "t5": tpr_at(p, yt, 0.05),
                             "tech": float(np.mean([pi > np.quantile(p[yt == 0], 0.99) for pi, r in zip(p, test) if r["label"] == 1 and r["genre"] == "technical"] or [np.nan])),
                             "rep": float(np.mean([pi > np.quantile(p[yt == 0], 0.99) for pi, r in zip(p, test) if r["label"] == 1 and r["genre"] == "report"] or [np.nan]))})
        print(f"split {s}: test {len(test)} rows | A auc {res['A'][-1]['auc']:.3f} tpr@1% {res['A'][-1]['t1']:.2f} | "
              f"B auc {res['B'][-1]['auc']:.3f} tpr@1% {res['B'][-1]['t1']:.2f}", flush=True)
    print(f"\n{args.tier} tier, identical test rows, {args.splits} random group splits (mean +- sd)")
    for k, label in (("auc", "AUC"), ("t1", "AI caught at 1% human FP"), ("t5", "AI caught at 5% human FP"),
                     ("tech", "technical AI caught at 1%"), ("rep", "report AI caught at 1%")):
        a, b = np.array([r[k] for r in res["A"]]), np.array([r[k] for r in res["B"]])
        print(f"  {label:32s} A (v4 data) {np.nanmean(a):.3f} +- {np.nanstd(a):.3f} | B (v5 data) {np.nanmean(b):.3f} +- {np.nanstd(b):.3f} | "
              f"B-A {np.nanmean(b - a):+.3f} (paired sd {np.nanstd(b - a):.3f})")


if __name__ == "__main__":
    main()
