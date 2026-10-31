#!/usr/bin/env python3
"""Is our Binoculars score implemented the way the original paper/code does it?

Compares orientations on a balanced sample of the corpus (first ~400 words of
each text, 512 tokens), reporting AUROC for "lower score = machine":

  mine      log-ppl under the OBSERVER (base) / cross-entropy of softmax(PERFORMER) . log_softmax(OBSERVER)
  official  log-ppl under the PERFORMER (instruct) / cross-entropy of softmax(OBSERVER) . log_softmax(PERFORMER)
  ppl_obs / ppl_perf   perplexity alone, for reference

Usage: python dev/binoculars_check.py [--n 120]
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "skills" / "humanize" / "scripts"))
import calibrate as c  # noqa: E402
import neural  # noqa: E402


def scores(text):
    import torch
    tok, obs, perf = neural._binoculars()
    ids = tok(text, truncation=True, max_length=512, return_tensors="pt").input_ids
    if ids.shape[1] < 16:
        return None
    with torch.inference_mode():
        lo = obs(ids).logits[0, :-1].float()
        lp = perf(ids).logits[0, :-1].float()
        t = ids[0, 1:]
        lso, lsp = torch.log_softmax(lo, -1), torch.log_softmax(lp, -1)
        ppl_obs = -lso.gather(1, t.unsqueeze(1)).mean()
        ppl_perf = -lsp.gather(1, t.unsqueeze(1)).mean()
        x_mine = -(torch.softmax(lp, -1) * lso).sum(-1).mean()       # softmax(perf) . log_softmax(obs)
        x_off = -(torch.softmax(lo, -1) * lsp).sum(-1).mean()        # softmax(obs) . log_softmax(perf)
        return {"mine": float(ppl_obs / x_mine), "official": float(ppl_perf / x_off),
                "ppl_obs": float(ppl_obs), "ppl_perf": float(ppl_perf)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    args = ap.parse_args()
    rows = [r for r in c.load_corpus() if r["genre"] not in {"extra", "report"}]
    random.seed(2)
    hum = random.sample([r for r in rows if r["label"] == 0], args.n // 2)
    ai = random.sample([r for r in rows if r["label"] == 1], args.n // 2)
    ys, out = [], {k: [] for k in ("mine", "official", "ppl_obs", "ppl_perf")}
    for r in hum + ai:
        s = scores(" ".join(r["text"].split()[:400]))
        if s is None:
            continue
        ys.append(r["label"])
        for k in out:
            out[k].append(s[k])
    y = np.array(ys)
    print(f"n={len(y)} ({int(y.sum())} AI)")
    for k, v in out.items():
        v = np.array(v)
        print(f"  {k:9s} AUROC when LOWER = machine: {roc_auc_score(y, -v):.3f}   (higher = machine: {roc_auc_score(y, v):.3f})")


if __name__ == "__main__":
    main()
