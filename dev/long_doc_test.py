#!/usr/bin/env python3
"""How accurate is the scorer on LONG documents? (style model, CPU only)

Usage: python dev/long_doc_test.py

The calibration texts are 150-450 words. To test long documents, held-out
texts of the same label and genre are joined into ~1,000 / ~3,000 / ~6,000-word
documents and scored by a model that never saw any of them (5-fold).
Reports accuracy, false-positive rate and scoring time per length.
"""
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "plugin" / "skills" / "humanize" / "scripts"))
import calibrate as c  # noqa: E402
import ai_score  # noqa: E402

random.seed(3)


def main():
    rows = c.extract_features(c.load_corpus(), deep=False)
    rows = [r for r in rows if r["genre"] not in {"extra", "report"}]
    y = np.array([r["label"] for r in rows])
    X = c.matrix(rows, c.STYLE_FEATURES, False)
    results = defaultdict(lambda: {"correct": 0, "n": 0, "fp": 0, "humans": 0, "secs": []})
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        model = c.make_gb(c.STYLE_FEATURES).fit(X[tr], y[tr])
        groups = defaultdict(list)
        for i in te:
            groups[(rows[i]["label"], rows[i]["genre"])].append(rows[i])
        for target in (1000, 3000, 6000):
            for (label, genre), items in groups.items():
                items = items[:]
                random.shuffle(items)
                doc, count, docs = [], 0, []
                for r in items:
                    doc.append(r["text"])
                    count += len(r["text"].split())
                    if count >= target:
                        docs.append("\n\n".join(doc))
                        doc, count = [], 0
                for text in docs:
                    t = time.time()
                    raw = ai_score.analyze(text)["raw"]
                    x = c.matrix([{"feat": {k: raw[k] for k in c.STYLE_FEATURES}}], c.STYLE_FEATURES, False)
                    p = model.predict_proba(x)[0, 1]
                    res = results[target]
                    res["secs"].append(time.time() - t)
                    res["n"] += 1
                    res["correct"] += int((p >= 0.5) == label)
                    if label == 0:
                        res["humans"] += 1
                        res["fp"] += int(p >= 0.5)
    print("## Long documents (style model, out-of-fold, CPU)")
    for target, r in sorted(results.items()):
        print(f"  ~{target:5d} words ({target // 300:2d} pages): accuracy {r['correct'] / r['n']:.1%} on {r['n']} docs | "
              f"humans wrongly flagged {r['fp'] / max(1, r['humans']):.1%} | scoring {np.mean(r['secs']):.1f}s per doc")


if __name__ == "__main__":
    main()
