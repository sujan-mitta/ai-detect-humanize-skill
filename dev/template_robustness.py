#!/usr/bin/env python3
"""Is the template model learning AI templates, or shortcuts from where the
human texts came from? Compare settings on what matters: genres it never saw.

Usage: python dev/template_robustness.py

For each setting: leave-one-genre-out AUC (train on all other genres, test on
the held-out one, averaged) and the share of held-out-genre AI texts (emails,
cover letters, LinkedIn, reports) it catches. Ordinary CV AUC is shown too,
but it is the number shortcuts inflate.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "ai-detect-humanize" / "scripts"))
import calibrate as c  # noqa: E402
import templates  # noqa: E402

SETTINGS = {
    "current (n=2-4, C=2, raw text)": dict(clean=False, n_max=4, C=2.0, min_df=5),
    "cleaned text, n=2-4, C=2": dict(clean=True, n_max=4, C=2.0, min_df=5),
    "cleaned, n=2-3, C=0.5": dict(clean=True, n_max=3, C=0.5, min_df=10),
    "cleaned, n=2-3, C=0.2": dict(clean=True, n_max=3, C=0.2, min_df=10),
    "cleaned, n=2-4, C=0.2": dict(clean=True, n_max=4, C=0.2, min_df=10),
}


def run(train, extra, cfg):
    templates.CLEAN = cfg["clean"]
    templates.N_MAX = cfg["n_max"]
    y = np.array([r["label"] for r in train])
    genres = sorted({r["genre"] for r in train})
    aucs = []
    for g in genres:
        tr = [i for i, r in enumerate(train) if r["genre"] != g]
        te = [i for i, r in enumerate(train) if r["genre"] == g]
        if len({train[i]["label"] for i in te}) < 2:
            continue
        m = c.fit_templates([train[i]["text"] for i in tr], y[tr], min_df=cfg["min_df"], C=cfg["C"])
        p = [templates.score(train[i]["text"], m) for i in te]
        aucs.append(roc_auc_score(y[te], p))
    full = c.fit_templates([r["text"] for r in train], y, min_df=cfg["min_df"], C=cfg["C"])
    caught = np.mean([templates.score(r["text"], full) >= 0.5 for r in extra])
    return float(np.mean(aucs)), float(caught)


def main():
    rows = c.load_corpus()
    train = [r for r in rows if r["genre"] not in {"extra", "report"}]
    extra = [r for r in rows if r["genre"] in {"extra", "report"}]
    print("| Setting | Leave-one-genre-out AUC | Unseen-genre AI caught |")
    print("|---|---|---|")
    for name, cfg in SETTINGS.items():
        auc, caught = run(train, extra, cfg)
        print(f"| {name} | {auc:.3f} | {caught:.0%} |", flush=True)


if __name__ == "__main__":
    main()
