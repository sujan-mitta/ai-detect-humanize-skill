#!/usr/bin/env python3
"""Compute neural detector features for the whole corpus (cached) and report
each detector's stand-alone accuracy.

Usage: python dev/neural_features.py [--limit N]

Writes/extends dev/runs/neural.jsonl ({"id", "neural": {...}}). Safe to stop
and restart: rows already computed are skipped.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "ai-detect-humanize" / "scripts"))
import calibrate as c  # noqa: E402
import neural  # noqa: E402

CACHE = ROOT / "runs" / "neural.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rows = c.load_corpus()
    done = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            done[d["id"]] = d["neural"]
    todo = [r for r in rows if r["id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    t0 = time.time()
    with CACHE.open("a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            feats = neural.features(r["text"])
            done[r["id"]] = feats
            f.write(json.dumps({"id": r["id"], "neural": feats}) + "\n")
            f.flush()
            if i % 50 == 0:
                rate = (time.time() - t0) / i
                print(f"  {i}/{len(todo)}  {rate:.1f}s per text, ~{rate * (len(todo) - i) / 60:.0f} min left", flush=True)

    scored = [r for r in rows if done.get(r["id"])]
    y = np.array([r["label"] for r in scored])
    print(f"\n## Stand-alone accuracy on {len(scored)} texts")
    for name, sign in (("fakespot", 1), ("desklib", 1), ("binoculars", -1)):
        v = np.array([done[r["id"]].get(name, np.nan) for r in scored], dtype=float)
        ok = ~np.isnan(v)
        auc = roc_auc_score(y[ok], sign * v[ok])
        # best single threshold accuracy (optimistic, for comparison only)
        best = max(((sign * v[ok] >= t) == y[ok]).mean() for t in np.quantile(sign * v[ok], np.linspace(0.02, 0.98, 97)))
        print(f"  {name:10s} AUC {auc:.3f} | best-threshold accuracy {best:.1%}")


if __name__ == "__main__":
    main()
