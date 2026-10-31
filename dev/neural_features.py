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
sys.path.insert(0, str(ROOT.parent / "humanize" / "scripts"))
import calibrate as c  # noqa: E402
import neural  # noqa: E402

CACHE = ROOT / "runs" / "neural.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import ai_score
    rows = c.load_corpus()
    loaded = c.load_neural(CACHE) if CACHE.exists() else {}
    done = {k: v[0] for k, v in loaded.items()}

    # reusable only if computed with the fixed Binoculars on the text as canonicalised NOW
    # (the record's 'cv' stamp, or for older records its position in the file; see
    # calibrate.neural_is_valid). Fresh records are stamped with the current version.
    def valid(r):
        n, idx = loaded.get(r["id"], (None, -1))
        return c.neural_is_valid(r["text"], n, idx)
    todo = [r for r in rows if not valid(r)]
    print(f"{len(todo)} of {len(rows)} texts to compute", flush=True)
    if args.limit:
        todo = todo[:args.limit]
    t0 = time.time()
    with CACHE.open("a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            feats = neural.features(r["text"])
            if feats:
                feats["binoculars_v2"] = True
                feats["cv"] = ai_score.CANON_VERSION
            done[r["id"]] = feats
            loaded[r["id"]] = (feats, 10**9)
            f.write(json.dumps({"id": r["id"], "neural": feats}) + "\n")
            f.flush()
            if i % 50 == 0:
                rate = (time.time() - t0) / i
                print(f"  {i}/{len(todo)}  {rate:.1f}s per text, ~{rate * (len(todo) - i) / 60:.0f} min left", flush=True)

    scored = [r for r in rows if valid(r)]
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
