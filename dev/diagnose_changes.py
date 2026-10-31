#!/usr/bin/env python3
"""Why did AI recall on unseen genres fall (76-86% -> 61-63%) while CV AUC rose?

Re-adds the changes one at a time and measures, for the style model:
  - author-grouped 5-fold CV AUC            (the number that went UP)
  - AI recall on genres with no training data: emails, cover letters, LinkedIn,
    blogs, business reports ("extra"/"report")        (the number that went DOWN)
  - recall on humanized AI text (out-of-fold)

Variants
  A  old features (no humanizer fingerprints, no template model), no humanized rows
  B  A + humanizer-fingerprint features
  C  B + learned template probability
  D  C + humanized AI texts added to training      (= current model)
  E  A + humanized rows                            (is it the extra rows alone?)
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "skills" / "humanize" / "scripts"))
import calibrate as c  # noqa: E402

NEW7 = ["fillers_per_1k", "dropped_apostrophes_per_1k", "ellipses_per_1k", "personal_hedges_per_1k",
        "second_person_per_1k", "conj_start_ratio", "length_alternation"]


def run(train, extra, names, use_humanized, label):
    tr_rows = [r for r in train if use_humanized or r["style"] != "humanized"]
    y = np.array([r["label"] for r in tr_rows])
    c.GROUPS = np.array([r["group"] for r in tr_rows])
    X = c.matrix(tr_rows, names)
    p = c.cv_predict_gb(X, y, names)
    auc = roc_auc_score(y, p)
    hum = np.array([r["style"] == "humanized" for r in tr_rows])
    gb = c.make_gb(names).fit(X, y)
    pe = gb.predict_proba(c.matrix(extra, names))[:, 1]
    by_genre = {}
    for r, v in zip(extra, pe):
        by_genre.setdefault(r["genre"], []).append(v >= 0.5)
    print(f"{label:34s} CV AUC {auc:.3f} | unseen-genre AI recall {np.mean(pe >= 0.5):4.0%} "
          f"(extra {np.mean(by_genre.get('extra', [0])):.0%}, report {np.mean(by_genre.get('report', [0])):.0%}) | "
          f"humanized recall {np.mean(p[hum] >= 0.5) if hum.any() else float('nan'):.0%} | mean P(AI) on unseen {pe.mean():.2f}",
          flush=True)


def main():
    rows = c.extract_features(c.load_corpus(), deep=False)
    train = [r for r in rows if r["genre"] not in {"extra", "report"}]
    extra = [r for r in rows if r["genre"] in {"extra", "report"}]
    y = np.array([r["label"] for r in train])
    c.GROUPS = np.array([r["group"] for r in train])
    print(f"{len(train)} training rows ({int(y.sum())} AI), {len(extra)} unseen-genre AI texts\n")
    c.add_template_feature(train, extra, y)
    base = [f for f in c.STYLE_FEATURES if f not in NEW7 and f != "template_prob"]
    run(train, extra, base, False, "A  old features")
    run(train, extra, base + NEW7, False, "B  + humanizer fingerprints")
    run(train, extra, base + NEW7 + ["template_prob"], False, "C  + learned templates")
    run(train, extra, base + NEW7 + ["template_prob"], True, "D  + humanized rows  (current)")
    run(train, extra, base, True, "E  old features + humanized rows")
    run(train, extra, base + ["template_prob"], True, "F  old features + templates + humanized")


if __name__ == "__main__":
    main()
