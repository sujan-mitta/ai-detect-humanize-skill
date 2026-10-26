#!/usr/bin/env python3
"""Does the score change when only formatting changes? (it must not)

Usage: python dev/invariance_test.py [--model PATH] [--n 100]

Takes human and AI texts and applies edits that carry NO information about authorship:
  single_block   join all paragraphs into one block
  split          insert a paragraph break every ~4 sentences
  curly          straight quotes/apostrophes -> curly
  straight       curly -> straight
  double_space   ". " -> ".  "
  hard_wrap      wrap at 70 columns with single newlines
  crlf_trailing  Windows line endings and trailing spaces
and reports, per edit and class, the mean absolute change in the 0-100 style-tier score
and the share of texts whose verdict flips. A model that reads layout or typography instead
of writing shows up here immediately. Style tier only (GPT-2/neural features need slow
recomputation); the typography/layout handling is shared by all tiers.
"""
import argparse
import random
import re
import sys
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "ai-detect-humanize" / "scripts"))
import calibrate as c  # noqa: E402
import ai_score  # noqa: E402


def single_block(t):
    return " ".join(t.split())


def split(t):
    sents = re.split(r"(?<=[.!?])\s+", " ".join(t.split()))
    return "\n\n".join(" ".join(sents[i:i + 4]) for i in range(0, len(sents), 4))


def curly(t):
    t = re.sub(r"(?<=\w)'(?=\w)", "’", t)
    t = re.sub(r'"([^"]*)"', "“\\1”", t)
    return t


def straight(t):
    return t.translate({0x2018: "'", 0x2019: "'", 0x201c: '"', 0x201d: '"'})


def double_space(t):
    return re.sub(r"([.!?]) (?=[A-Z])", r"\1  ", t)


def hard_wrap(t):
    return "\n\n".join("\n".join(textwrap.wrap(p, 70)) for p in re.split(r"\n\s*\n", t))


def crlf_trailing(t):
    return "\r\n".join(l + "  " for l in t.split("\n"))


def tokenize(t):
    """PTB-style pre-tokenised text, as in the HC3 human answers: 'word , word .'"""
    t = re.sub(r"([,.;:!?%)\]])", r" \1", t)
    t = re.sub(r"([(\[])", r"\1 ", t)
    return re.sub(r"(\w)n't\b", r"\1 n't", t)


EDITS = {"single_block": single_block, "split": split, "curly": curly, "straight": straight,
         "double_space": double_space, "hard_wrap": hard_wrap, "crlf_trailing": crlf_trailing,
         "tokenize": tokenize}


def score(text):
    a = ai_score.analyze(text)
    cal = ai_score.model_score(a["raw"], None, "style")
    p, _ = ai_score.apply_tell_net(cal["score"], a["raw"]["tells_per_100w"])
    return ai_score.display(p, "style")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT.parent / "ai-detect-humanize" / "scripts" / "model.json"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--plain", action="store_true",
                    help="only texts without markdown: flattening a list or bold label changes the "
                         "content (those ARE AI tells), so layout-only invariance is judged on plain prose")
    args = ap.parse_args()
    ai_score.MODEL_PATH = Path(args.model)
    ai_score._MODELS = None
    random.seed(8)
    rows = [r for r in c.load_corpus() if r["genre"] != "extra" and r["style"] != "humanized"]
    if args.plain:
        rows = [r for r in rows if not re.search(r"\*\*|^#{1,6} |^\s*[-*•] |^\s*\d+[.)] ", r["text"], re.M)]
    pick = random.sample([r for r in rows if r["label"] == 0], args.n // 2) + \
        random.sample([r for r in rows if r["label"] == 1], args.n // 2)
    base = np.array([score(r["text"]) for r in pick])
    label = np.array([r["label"] for r in pick])
    print(f"{len(pick)} texts; verdict threshold = {ai_score.SCORE_AI}\n")
    print(f"{'edit':16s} {'mean |change| human':>20s} {'AI':>8s} {'verdict flips':>15s}")
    worst = 0.0
    for name, fn in EDITS.items():
        new = np.array([score(fn(r["text"])) for r in pick])
        d = np.abs(new - base)
        flips = ((base >= ai_score.SCORE_AI) != (new >= ai_score.SCORE_AI)).mean()
        print(f"{name:16s} {d[label == 0].mean():17.1f} pts {d[label == 1].mean():6.1f}  {flips:13.0%}")
        worst = max(worst, d.mean())
    print(f"\nworst mean change {worst:.1f} points (goal: under 2)")


if __name__ == "__main__":
    main()
