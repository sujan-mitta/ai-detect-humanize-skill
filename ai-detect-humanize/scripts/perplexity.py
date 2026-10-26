#!/usr/bin/env python3
"""Deep detection features from a small local language model (GPT-2).

Usage: python perplexity.py INPUT [--model gpt2] [--json OUT.json]

AI text is unusually *predictable* to a language model: low perplexity, many
tokens that were the model's top guess, and little variation in perplexity
from sentence to sentence. This script measures that (the same idea behind
GPTZero's perplexity/burstiness and the GLTR tool).

Needs:  pip install torch transformers   (CPU is fine, no graphics card needed)
First run downloads the model (~500 MB for gpt2) into the Hugging Face cache.

Speed: the text is scored in ONE pass of 1,024-token windows and each token is
mapped back to its sentence, so sentence-level perplexity costs nothing extra.
Documents longer than MAX_TOKENS are scored on evenly spaced windows that
together cover MAX_TOKENS tokens - the features are averages, so a large even
sample gives the same answer as reading every token.

ai_score.py calls features() when run with --deep; you can also run this
directly to see the numbers.
"""
import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

WINDOW = 1024
MAX_TOKENS = 12000   # ~9,000 words scored in full; longer texts are sampled evenly
_MODEL = {}


def available():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def load(name="gpt2"):
    if name not in _MODEL:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name)
        model.eval()
        _MODEL[name] = (tok, model)
    return _MODEL[name]


def _windows(n_tokens, max_tokens=MAX_TOKENS):
    """Start offsets of non-overlapping windows; evenly spaced if the text is long."""
    if n_tokens <= max_tokens:
        return list(range(0, max(1, n_tokens - 1), WINDOW))
    k = max(1, max_tokens // WINDOW)
    step = (n_tokens - WINDOW) / max(1, k - 1)
    return sorted({int(i * step) for i in range(k)})


def token_scores(text, name="gpt2", max_tokens=MAX_TOKENS):
    """Per-token (char_offset, logprob, rank) for the scored windows."""
    import torch
    tok, model = load(name)
    enc = tok(text, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    out = []
    with torch.inference_mode():
        for start in _windows(len(ids), max_tokens):
            chunk = ids[start:start + WINDOW]
            if len(chunk) < 2:
                continue
            x = torch.tensor([chunk])
            logp = torch.log_softmax(model(x).logits[0, :-1].float(), dim=-1)
            target = x[0, 1:]
            true_lp = logp.gather(1, target.unsqueeze(1)).squeeze(1)
            rank = (logp > true_lp.unsqueeze(1)).sum(dim=1)
            # Fast-DetectGPT (Bao et al., 2024) needs, per position, the mean and variance of
            # log p(x') for x' drawn from the model itself: mean = sum p log p (negative
            # entropy), variance = sum p (log p)^2 - mean^2. Free from the logits we have.
            p = logp.exp()
            mean_lp = (p * logp).sum(-1)
            var_lp = ((p * logp * logp).sum(-1) - mean_lp ** 2).clamp(min=1e-9)
            for j, (lp, rk, mu, var) in enumerate(zip(true_lp.tolist(), rank.tolist(),
                                                      mean_lp.tolist(), var_lp.tolist())):
                out.append((offsets[start + j + 1][0], lp, rk, mu, var))
    return out


def features(text, name="gpt2", max_tokens=MAX_TOKENS):
    """Document-level deep features plus per-sentence perplexities, in one pass."""
    from ai_score import prose_blocks, sentences
    blocks = prose_blocks(text)
    prose = "\n\n".join(blocks)
    if not prose.strip():
        return None
    scores = token_scores(prose, name, max_tokens)
    if len(scores) < 20:
        return None
    # sentence character spans inside `prose`
    spans, pos = [], 0
    for b in blocks:
        for s in sentences(b):
            i = prose.find(s[:40], pos)
            if i < 0:
                continue
            spans.append((i, i + len(s), s))
            pos = i + 1
    per_sent = [[] for _ in spans]
    si = 0
    for off, lp, *_ in scores:
        while si < len(spans) and off >= spans[si][1]:
            si += 1
        if si < len(spans) and spans[si][0] <= off < spans[si][1]:
            per_sent[si].append(lp)
    sent_ppl = [(spans[i][2], math.exp(-statistics.mean(lps))) for i, lps in enumerate(per_sent) if len(lps) >= 4]
    lps = [s[1] for s in scores]
    ranks = [s[2] for s in scores]
    n = len(ranks)
    ppl = math.exp(-statistics.mean(lps))
    vals = [p for _, p in sent_ppl]
    # Fast-DetectGPT analytic criterion: (log-likelihood - expected) / std. Higher = machine.
    total_mu = sum(s[3] for s in scores)
    total_var = sum(s[4] for s in scores)
    fastdetect = (sum(lps) - total_mu) / math.sqrt(total_var)
    return {
        "perplexity": round(ppl, 2),
        "log_perplexity": round(math.log(ppl), 4),
        "fastdetect": round(fastdetect, 4),
        "mean_entropy": round(-total_mu / n, 4),
        "top1_fraction": round(sum(r == 0 for r in ranks) / n, 4),
        "top10_fraction": round(sum(r < 10 for r in ranks) / n, 4),
        "rare_fraction": round(sum(r >= 1000 for r in ranks) / n, 4),
        "sentence_ppl_mean": round(statistics.mean(vals), 2) if vals else None,
        "sentence_ppl_cv": round(statistics.pstdev(vals) / statistics.mean(vals), 4) if len(vals) >= 3 else None,
        "sentence_ppl": [(s[:120], round(p, 1)) for s, p in sent_ppl],
        "tokens_scored": n,
        "model": name,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--json")
    args = ap.parse_args()
    if not available():
        sys.exit("Deep mode needs: pip install torch transformers")
    from extract_text import extract
    f = features(extract(args.input), args.model)
    if args.json:
        Path(args.json).write_text(json.dumps(f, indent=2), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps({k: v for k, v in f.items() if k != "sentence_ppl"}, indent=2))


if __name__ == "__main__":
    main()
