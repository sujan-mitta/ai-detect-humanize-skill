#!/usr/bin/env python3
"""Shortcut audit: properties that should NOT tell human from AI, but might in our corpus.

For each non-linguistic property, the AUC of that property ALONE (max(auc, 1-auc), so 0.5 =
no information), overall and within each genre. A value well above 0.5 means a model can
cheat on it; those properties are either neutralised (typography) or must be balanced in
the data (length, paragraph layout).
"""
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "ai-detect-humanize" / "scripts"))
import calibrate as c  # noqa: E402
import ai_score  # noqa: E402


def props(text):
    blocks = ai_score.prose_blocks(text)
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    words = text.split()
    lines = [l for l in text.splitlines() if l.strip()]
    return {
        "word_count": len(words),
        "paragraph_count": len(paras),
        "mean_paragraph_words": len(words) / max(1, len(paras)),
        "mean_line_length": np.mean([len(l) for l in lines]) if lines else 0,
        "has_markdown": float(bool(re.search(r"\*\*|^#{1,6} |^\s*[-*] ", text, re.M))),
        "digit_share": sum(ch.isdigit() for ch in text) / max(1, len(text)),
        "uppercase_share": sum(ch.isupper() for ch in text) / max(1, sum(ch.isalpha() for ch in text)),
        "non_ascii_share": sum(ord(ch) > 127 for ch in text) / max(1, len(text)),
        "ends_with_punct": float(text.strip()[-1:] in ".!?\"'"),
        "starts_uppercase": float(text.strip()[:1].isupper()),
        "avg_word_len": np.mean([len(w) for w in words]) if words else 0,
        "newline_share": text.count("\n") / max(1, len(text)),
        "double_space_after_stop": float(bool(re.search(r"[.!?]  +\S", text))),
        "curly_quotes": float(bool(re.search(r"[‘’“”]", text))),
        "typography_changed": float(ai_score.typography_changed(text)),
    }


def main():
    rows = [r for r in c.load_corpus() if r["genre"] != "extra"]
    y = np.array([r["label"] for r in rows])
    P = [props(r["text"]) for r in rows]
    genres = np.array([r["genre"] for r in rows])
    print(f"{len(rows)} texts ({int(y.sum())} AI). AUC of each property alone (0.5 = no information)\n")
    print(f"{'property':26s} {'overall':>8s} {'median within genre':>20s} {'worst genre':>14s}")
    flagged = []
    for k in P[0]:
        v = np.array([p[k] for p in P], dtype=float)
        if v.std() == 0:
            continue
        auc = roc_auc_score(y, v)
        overall = max(auc, 1 - auc)
        per = []
        for g in sorted(set(genres)):
            m = genres == g
            if len(set(y[m])) == 2 and v[m].std() > 0:
                a = roc_auc_score(y[m], v[m])
                per.append((max(a, 1 - a), g))
        med = float(np.median([a for a, _ in per])) if per else float("nan")
        worst = max(per) if per else (float("nan"), "-")
        mark = "  <-- shortcut" if overall > 0.65 or med > 0.65 else ""
        print(f"{k:26s} {overall:8.3f} {med:20.3f} {worst[0]:9.3f} ({worst[1]}){mark}")
        if mark:
            flagged.append(k)
    print("\nflagged:", flagged or "none")


if __name__ == "__main__":
    main()
