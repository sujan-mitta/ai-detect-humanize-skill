#!/usr/bin/env python3
"""Which sentences should a rewrite change first?

Usage: python rewrite_targets.py INPUT [--top 12] [--deep]

Ranks the sentences of a text by how much they add to its AI-likeness and says why, so a
revision round can concentrate on the few sentences that matter instead of re-wording
everything. Three signals per sentence:
  template   how strongly its skeleton (function words and punctuation, content words masked)
             matches sentence shapes the learned template model associates with AI
  joins      comma + conjunction chains (", and", ", so", ", which", ", since", ", but"...);
             in rewrite experiments, replacing these with a full stop, semicolon, colon or
             parentheses lowered scores more than any word-level change
  predictable (--deep) the sentence's GPT-2 perplexity is low
Also prints the overall score and, when the score is saturated near 100, the model margin
to track between rounds. Edits should keep facts and read well; this only points at targets."""
import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ai_score  # noqa: E402
import templates  # noqa: E402
from extract_text import extract  # noqa: E402

SPLIT_RE = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[*]?[A-Z0-9])")
JOIN_RE = re.compile(r",\s+(and|so|which|since|but|while|because|as|although|whereas|allowing|making|ensuring)\b", re.I)


def split_sentences(text):
    out = []
    for block in re.split(r"\n\s*\n", text):
        b = block.strip()
        if b and not re.match(r"^#{1,6}\s", b):
            out += [s.strip() for s in SPLIT_RE.split(b) if len(s.split()) >= 4]
    return out


def template_heat(sentences, model):
    """Mean learned weight (toward AI, positive) of the skeleton n-grams in each sentence."""
    vocab = {g: i for i, g in enumerate(model["vocab"])}
    heat = []
    for s in sentences:
        counts = templates.ngram_counts(s)
        tot = sum(model["coef"][vocab[g]] * model["idf"][vocab[g]] * n for g, n in counts.items() if g in vocab)
        heat.append(tot / max(1, len(s.split())))
    return heat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--deep", action="store_true")
    a = ap.parse_args()
    text = extract(a.input)
    sents = split_sentences(text)
    models = ai_score.load_models()
    tm = models.get("templates")
    heat = template_heat(sents, tm) if tm else [0.0] * len(sents)
    mu, sd = (sum(heat) / len(heat), (sum((h - sum(heat) / len(heat)) ** 2 for h in heat) / len(heat)) ** 0.5 or 1.0) if heat else (0, 1)
    ppl = {}
    if a.deep:
        import perplexity
        f = perplexity.features(text) or {}
        ppl = {" ".join(s.split())[:60]: p for s, p in f.get("sentence_ppl", [])}
    rows = []
    for s, h in zip(sents, heat):
        joins = len(JOIN_RE.findall(s))
        p = ppl.get(" ".join(s.split())[:60])
        score = (h - mu) / sd + 0.8 * joins + (0.6 if p is not None and p < 25 else 0)
        why = []
        if (h - mu) / sd > 0.8:
            why.append("AI-typical sentence shape")
        if joins:
            why.append(f"{joins} comma+conjunction join(s)")
        if p is not None and p < 25:
            why.append(f"very predictable (ppl {p:.0f})")
        rows.append((score, s, why))
    res = ai_score.analyze(text)
    z = None
    if a.deep:
        r = ai_score.build(text, deep=True)
        print(f"Score {r['score']}/100 ({r['verdict']})" + (f"; model margin {r['calibrated']['z']:+.2f}" if r.get("calibrated") and r["calibrated"].get("z") is not None else ""))
    else:
        print(f"Score {res['score']}/100 ({res['verdict']}) [style model]")
    print(f"{len(sents)} sentences; showing the {min(a.top, len(rows))} to change first (highest first)\n")
    for score, s, why in sorted(rows, key=lambda t: -t[0])[:a.top]:
        if not why:
            continue
        print(f"- [{score:+.1f}] {'; '.join(why)}\n    {s[:220]}")
    n_join = sum(len(JOIN_RE.findall(s)) for s in sents)
    print(f"\nTotals: {n_join} comma+conjunction joins in {len(sents)} sentences ({n_join / max(1, len(sents)):.2f} per sentence).")


if __name__ == "__main__":
    main()
