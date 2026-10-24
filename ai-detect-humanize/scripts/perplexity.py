#!/usr/bin/env python3
"""Deep detection features from a small local language model (GPT-2).

Usage: python perplexity.py INPUT [--model gpt2] [--json OUT.json]

AI text is unusually *predictable* to a language model: low perplexity, many
tokens that were the model's top guess, and little variation in perplexity
from sentence to sentence. This script measures that (the same idea behind
GPTZero's perplexity/burstiness and the GLTR tool).

Needs:  pip install torch transformers   (CPU is fine)
First run downloads the model (~500 MB for gpt2) into the Hugging Face cache.

ai_score.py calls features() when run with --deep; you can also run this
directly to see per-sentence perplexity.
"""
import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

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
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name)
        model.eval()
        torch.set_grad_enabled(False)
        _MODEL[name] = (tok, model)
    return _MODEL[name]


def token_stats(text, name="gpt2", max_tokens=1024, stride=512):
    """Per-token log-prob and rank of the true token, over a sliding window."""
    import torch
    tok, model = load(name)
    ids = tok(text, return_tensors="pt").input_ids[0]
    logps, ranks, offsets = [], [], []
    start = 0
    while start < len(ids) - 1:
        end = min(start + max_tokens, len(ids))
        chunk = ids[start:end].unsqueeze(0)
        logits = model(chunk).logits[0, :-1]
        target = chunk[0, 1:]
        lp = torch.log_softmax(logits.float(), dim=-1)
        true_lp = lp.gather(1, target.unsqueeze(1)).squeeze(1)
        rank = (lp > true_lp.unsqueeze(1)).sum(dim=1)
        # with a sliding window, only score tokens not already scored
        skip = 0 if start == 0 else max_tokens - stride
        logps += true_lp[skip:].tolist()
        ranks += rank[skip:].tolist()
        offsets += list(range(start + 1 + skip, end))
        if end == len(ids):
            break
        start += stride
    return ids, logps, ranks, offsets


def features(text, name="gpt2"):
    """Document-level deep features plus per-sentence perplexities."""
    from ai_score import prose_blocks, sentences
    prose = "\n\n".join(prose_blocks(text))
    sents = [s for b in prose_blocks(text) for s in sentences(b)]
    if not prose.strip():
        return None
    ids, logps, ranks, _ = token_stats(prose, name)
    if len(logps) < 20:
        return None
    ppl = math.exp(-statistics.mean(logps))
    sent_ppl = []
    for s in sents:
        if len(s.split()) < 4:
            continue
        _, lp, _, _ = token_stats(s, name)
        if lp:
            sent_ppl.append(math.exp(-statistics.mean(lp)))
    n = len(ranks)
    return {
        "perplexity": round(ppl, 2),
        "log_perplexity": round(math.log(ppl), 4),
        "top1_fraction": round(sum(r == 0 for r in ranks) / n, 4),
        "top10_fraction": round(sum(r < 10 for r in ranks) / n, 4),
        "rare_fraction": round(sum(r >= 1000 for r in ranks) / n, 4),
        "sentence_ppl_mean": round(statistics.mean(sent_ppl), 2) if sent_ppl else None,
        "sentence_ppl_cv": round(statistics.pstdev(sent_ppl) / statistics.mean(sent_ppl), 4) if len(sent_ppl) >= 3 else None,
        "sentence_ppl": [round(p, 1) for p in sent_ppl],
        "tokens": n,
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
