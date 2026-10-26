#!/usr/bin/env python3
"""Why do formal human reports look like AI, and why do some AI technical texts look human?

Trains the max tier leave-one-genre-out (out-of-fold probabilities), then for a genre lists
the worst false positives / misses with the features that make them look like the other class
(z-score against the genre's own human and AI texts)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import calibrate as c  # noqa: E402
import calibrate2  # noqa: E402

GENRES = sys.argv[1:] or ["report", "technical"]


def main():
    rows = c.extract_features(c.load_corpus(), True)
    train = [r for r in rows if r["genre"] != "extra" and r.get("deep") and r.get("neural")]
    c.GROUPS = np.array([r["group"] for r in train])
    c.GENRES = np.array([r["genre"] for r in train])
    y = np.array([r["label"] for r in train])
    calibrate2.template_features(train, [])
    names = c.STYLE_FEATURES + c.DEEP_FEATURES + c.NEURAL_FEATURES
    X = c.matrix(train, names)
    p = c.cv_predict_gb(X, y, names)
    thr = c.verdict_thresholds(p, y)["ai"] / 100
    g = np.array([r["genre"] for r in train])
    print(f"\nthreshold {thr:.2f}")
    for genre in GENRES:
        h, a = (g == genre) & (y == 0), (g == genre) & (y == 1)
        mu_h, sd_h = X[h].mean(0), X[h].std(0) + 1e-9
        mu_a, sd_a = X[a].mean(0), X[a].std(0) + 1e-9
        print(f"\n===== {genre}: {h.sum()} human (flagged {np.mean(p[h] >= thr):.0%}), {a.sum()} AI (caught {np.mean(p[a] >= thr):.0%})")
        # which features separate the genre's two classes most (standardised gap)
        gap = (mu_a - mu_h) / np.sqrt((sd_a ** 2 + sd_h ** 2) / 2)
        order = np.argsort(-np.abs(gap))[:10]
        print("  strongest class gaps in this genre (AI minus human, in SDs): " + ", ".join(f"{names[i]} {gap[i]:+.1f}" for i in order))
        for label, mask, sel in (("FALSE POSITIVES (human, high score)", h, np.argsort(-np.where(h, p, -1))[:8]),
                                 ("MISSES (AI, low score)", a, np.argsort(np.where(a, p, 2))[:8])):
            print(f"\n  {label}")
            for i in sel:
                if not mask[i]:
                    continue
                other_mu, other_sd = (mu_a, sd_a) if label.startswith("FALSE") else (mu_h, sd_h)
                z = (X[i] - other_mu) / other_sd
                near = np.argsort(np.abs(z))[:5]
                txt = train[i]["text"].replace("\n", " ")[:130]
                print(f"   p={p[i]:.2f} [{train[i]['source'][:28]}] {txt}")
                print("      closest to the other class on: " + ", ".join(f"{names[k]}({X[i][k]:.2f})" for k in near))


if __name__ == "__main__":
    main()
