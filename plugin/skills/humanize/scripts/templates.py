#!/usr/bin/env python3
"""Learned sentence-template detector ("skeleton n-grams").

Hand-written phrase lists only catch wordings someone thought of. This module
reduces text to its *skeleton* - function words and punctuation kept, content
words replaced by placeholders - and scores n-grams of that skeleton with a
model learned from labelled human and AI text. So

    "It's not just a tool - it's a mindset."  ->  it's not just a w - it's a w .
    "By leveraging data, companies can..."      ->  by w w , w can ...

and the model learns which grammatical templates AI uses, without memorising
topics. Pure Python at scoring time; the model lives in model.json["templates"].

Usage: python templates.py INPUT   (prints P(AI) from templates alone)
"""
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

FUNCTION_WORDS = set("""
a an the this that these those it its it's they them their there here what which who whom whose
i me my we us our you your he him his she her one ones
is are was were be been being am do does did done have has had having will would shall should can could
may might must not no nor n't 's 're 've 'll 'd
and or but so yet for because if then than when while although though since unless until whether
as at by from in into of off on onto out over through to under up upon with within without about
above after against along among around before behind below between beyond during except like near past
toward towards via
also just only even still already always never often sometimes usually really very quite rather too more
most less least much many few several some any each every all both either neither other another such
same own enough how why where whereas however therefore thus hence moreover furthermore additionally
instead indeed perhaps maybe probably actually basically simply
not only but also whether
make makes made get gets got go goes went take takes took see seen use used using
can't won't don't doesn't didn't isn't aren't wasn't weren't i'm i've i'd you're we're they're that's
what's let's here's there's who's
""".split())

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*|[.,;:!?()\-—–\"'“”…*#]")
N_MAX = 4
CLEAN = False  # cleaning measured slightly worse on unseen genres (dev/template_robustness.py)


def clean(text):
    """Remove source-specific typography so the model learns grammar, not where a
    text came from: markdown/bold/headings, Usenet '--' and '>' quoting,
    Gutenberg _italics_, LaTeX, list markers, odd quotes and dash styles."""
    t = re.sub(r"^\s*(#{1,6}|[-*•>]|\d+[.)])\s+", "", text, flags=re.M)
    t = re.sub(r"\*\*|__|(?<!\w)_|_(?!\w)|`", "", t)
    t = re.sub(r"\$[^$\n]{1,80}\$", "x", t)
    t = re.sub(r"\s*(--|—|–)\s*", " - ", t)
    t = t.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    t = t.replace("…", "...")
    return re.sub(r"\s+", " ", t)


def skeleton(text):
    """Text -> list of skeleton tokens (function words, punctuation, w/W/0 placeholders)."""
    out = []
    from ai_score import canon_typography   # same typography/detokenisation as every other feature
    text = canon_typography(text)
    if CLEAN:
        text = clean(text)
    for t in TOKEN_RE.findall(text.replace("’", "'")):
        low = t.lower()
        if low in FUNCTION_WORDS:
            out.append(low)
        elif t[0].isdigit():
            out.append("0")
        elif t[0].isalpha():
            out.append("W" if t[0].isupper() and out and out[-1] not in (".", "!", "?") else "w")
        else:
            out.append({"—": "-", "–": "-", "“": '"', "”": '"', "…": "..."}.get(t, t))
    return out


def ngram_counts(text):
    toks = skeleton(text)
    c = Counter()
    for n in range(2, N_MAX + 1):
        for i in range(len(toks) - n + 1):
            gram = toks[i:i + n]
            if sum(g not in ("w", "W", "0") for g in gram) >= 1:  # skip all-placeholder grams
                c[" ".join(gram)] += 1
    return c


def vectorize(counts, vocab, idf):
    """Sparse tf-idf (sublinear tf, l2-normalised), as {index: value}."""
    v = {}
    for g, n in counts.items():
        j = vocab.get(g)
        if j is not None:
            v[j] = (1 + math.log(n)) * idf[j]
    norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {j: x / norm for j, x in v.items()}


def score(text, model):
    """P(AI) from skeleton templates. model = model.json['templates']."""
    vocab = model["_vocab_index"] if "_vocab_index" in model else {g: i for i, g in enumerate(model["vocab"])}
    model["_vocab_index"] = vocab
    v = vectorize(ngram_counts(text), vocab, model["idf"])
    z = model["intercept"] + sum(model["coef"][j] * x for j, x in v.items())
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


def score_counts(counts, model):
    """P(AI) from pre-computed n-gram counts (lets calibration reuse counts across folds)."""
    vocab = model.get("_vocab_index")
    if vocab is None:
        vocab = model["_vocab_index"] = {g: i for i, g in enumerate(model["vocab"])}
    v = vectorize(counts, vocab, model["idf"])
    z = model["intercept"] + sum(model["coef"][j] * x for j, x in v.items())
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


def top_templates(text, model, k=8):
    """The skeleton n-grams in this text that push hardest toward AI."""
    vocab = {g: i for i, g in enumerate(model["vocab"])}
    counts = ngram_counts(text)
    scored = [(model["coef"][vocab[g]] * (1 + math.log(n)), g) for g, n in counts.items() if g in vocab]
    return [g for s, g in sorted(scored, reverse=True)[:k] if s > 0]


def main():
    import json
    from extract_text import extract
    import ai_score
    m = ai_score.load_models().get("templates")
    if not m:
        sys.exit("model.json has no template model yet (run dev/calibrate.py --write)")
    t = extract(sys.argv[1])
    print(f"P(AI) from templates: {score(t, m):.2f}")
    print("most AI-typical templates:", " | ".join(top_templates(t, m)))


if __name__ == "__main__":
    main()
