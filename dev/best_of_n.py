#!/usr/bin/env python3
"""Sentence-level best-of-N rewrite selection (experiment).

Input JSON: {"units": [{"kind": "para", "sentences": [[orig, alt1, ...], ...]}, {"kind": "keep", "text": "..."}]}
For each sentence pick the candidate that (1) keeps the facts of the original sentence (numbers, quotes,
names, acronyms must match exactly), (2) stays close in meaning (MiniLM cosine to the original >= --sim),
(3) has a sane length, and among those (4) is the LEAST predictable to GPT-2 given the previous chosen
sentence (highest mean token surprisal). Writes the assembled text.
Usage: python dev/best_of_n.py CANDS.json OUT.txt [--sim 0.78] [--stats]"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "ai-detect-humanize" / "scripts"))
import perplexity as px  # noqa: E402
import verify_rewrite as vr  # noqa: E402


def surprisal(prev, cand):
    text = (prev + " " + cand) if prev else cand
    off = len(prev) + 1 if prev else 0
    s = [-lp for o, lp, *_ in px.token_scores(text) if o >= off]
    return float(np.mean(s)) if s else 0.0


def fact_key(s):
    f = vr.facts(s)
    return tuple(sorted(tuple(sorted(f[k])) for k in ("numbers", "quotes", "dates", "code")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cands"); ap.add_argument("out")
    ap.add_argument("--sim", type=float, default=0.78)
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    data = json.loads(Path(a.cands).read_text(encoding="utf-8"))
    out, changed, total = [], 0, 0
    for u in data["units"]:
        if u["kind"] == "keep":
            out.append(u["text"]); continue
        chosen, prev = [], ""
        for cands in u["sentences"]:
            orig = cands[0]
            ok = [orig]
            embs = vr._embed(cands).numpy() if len(cands) > 1 else None
            for i, c in enumerate(cands[1:], 1):
                if not c.strip() or fact_key(c) != fact_key(orig):
                    continue
                if not (0.5 <= len(c.split()) / max(1, len(orig.split())) <= 1.7):
                    continue
                if float(embs[0] @ embs[i]) < a.sim:
                    continue
                ok.append(c)
            best = max(ok, key=lambda c: surprisal(prev, c))
            changed += best != orig; total += 1
            chosen.append(best); prev = best
        out.append(" ".join(chosen))
    Path(a.out).write_text("\n\n".join(out) + "\n", encoding="utf-8")
    if a.stats:
        print(f"{changed}/{total} sentences replaced")


if __name__ == "__main__":
    main()
