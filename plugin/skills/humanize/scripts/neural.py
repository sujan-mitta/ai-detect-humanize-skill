#!/usr/bin/env python3
"""Neural detector features (the "max" tier): pretrained AI-text classifiers and
Binoculars, all CPU-friendly and loaded from safetensors only (no remote code).

Usage: python neural.py INPUT [--json OUT.json]

Features
  fakespot   P(AI) from fakespot-ai/roberta-base-ai-text-detection-v1 (Apache-2.0)
  desklib    P(AI) from desklib/ai-text-detector-v1.01, DeBERTa-v3-large (MIT)
  binoculars Binoculars score (Hans et al., 2024) with SmolLM2-360M base as
             observer and SmolLM2-360M-Instruct as performer (Apache-2.0).
             Weak on its own at this size, but in the ensemble it lifts AI caught
             in unseen genres from 79% to 86%.

Long texts are scored on up to MAX_CHUNKS evenly spaced ~400-word chunks and
averaged, so a book costs about the same as a few pages.

Needs: pip install torch transformers   (first run downloads ~3.7 GB of models)
"""
import argparse
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

CHUNK_WORDS = 400
MAX_CHUNKS = 6
_CACHE = {}

FAKESPOT = "fakespot-ai/roberta-base-ai-text-detection-v1"
DESKLIB = "desklib/ai-text-detector-v1.01"
OBSERVER = "HuggingFaceTB/SmolLM2-360M"
PERFORMER = "HuggingFaceTB/SmolLM2-360M-Instruct"


def available():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _quiet():
    import torch
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    torch.set_num_threads(max(1, os.cpu_count() or 1))


def chunks(text, words=CHUNK_WORDS, max_chunks=MAX_CHUNKS):
    """Evenly spaced chunks of ~`words` words, cut on paragraph boundaries."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, cur, n = [], [], 0
    for p in paras:
        cur.append(p)
        n += len(p.split())
        if n >= words:
            out.append("\n\n".join(cur))
            cur, n = [], 0
    if cur and (n >= words // 3 or not out):
        out.append("\n\n".join(cur))
    if len(out) > max_chunks:
        step = (len(out) - 1) / (max_chunks - 1)
        out = [out[round(i * step)] for i in range(max_chunks)]
    return out


def _fakespot():
    if "fakespot" not in _CACHE:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        _quiet()
        tok = AutoTokenizer.from_pretrained(FAKESPOT)
        model = AutoModelForSequenceClassification.from_pretrained(FAKESPOT, use_safetensors=True).eval()
        labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
        ai_idx = next((i for name, i in labels.items() if "ai" in name or "machine" in name or "fake" in name), 1)
        _CACHE["fakespot"] = (tok, model, ai_idx)
    return _CACHE["fakespot"]


def fakespot_prob(text):
    import torch
    tok, model, ai_idx = _fakespot()
    with torch.inference_mode():
        enc = tok(text, truncation=True, max_length=512, return_tensors="pt")
        return float(torch.softmax(model(**enc).logits[0].float(), -1)[ai_idx])


def _desklib():
    if "desklib" not in _CACHE:
        import torch
        from torch import nn
        from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel
        _quiet()

        class DesklibAIDetectionModel(PreTrainedModel):
            # architecture as published on the model card: encoder + mean pooling + linear head
            config_class = AutoConfig

            def __init__(self, config):
                super().__init__(config)
                self.model = AutoModel.from_config(config)
                self.classifier = nn.Linear(config.hidden_size, 1)
                self.post_init()  # newer transformers versions require this before loading weights

            def forward(self, input_ids, attention_mask=None):
                hidden = self.model(input_ids, attention_mask=attention_mask)[0]
                mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                return self.classifier(pooled)

        tok = AutoTokenizer.from_pretrained(DESKLIB)
        model = DesklibAIDetectionModel.from_pretrained(DESKLIB, use_safetensors=True).eval()
        _CACHE["desklib"] = (tok, model)
    return _CACHE["desklib"]


def desklib_prob(text):
    import torch
    tok, model = _desklib()
    with torch.inference_mode():
        enc = tok(text, truncation=True, max_length=768, return_tensors="pt")
        return float(torch.sigmoid(model(enc["input_ids"], enc["attention_mask"])[0, 0].float()))


def _binoculars():
    if "bino" not in _CACHE:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _quiet()
        tok = AutoTokenizer.from_pretrained(OBSERVER)
        obs = AutoModelForCausalLM.from_pretrained(OBSERVER, use_safetensors=True, torch_dtype=torch.float32).eval()
        perf = AutoModelForCausalLM.from_pretrained(PERFORMER, use_safetensors=True, torch_dtype=torch.float32).eval()
        _CACHE["bino"] = (tok, obs, perf)
    return _CACHE["bino"]


def binoculars_score(text):
    """Binoculars (Hans et al., 2024), following the reference implementation:
    log-perplexity of the text under the PERFORMER (instruct) model, divided by the
    cross-perplexity -sum softmax(OBSERVER) . log_softmax(PERFORMER). Low = machine.

    (An earlier version here used the observer for the perplexity and the performer's
    probabilities in the cross term - the mirror image - and scored AUROC 0.48 on our
    corpus; this orientation scores 0.79. See dev/binoculars_check.py.)"""
    import torch
    tok, obs, perf = _binoculars()
    with torch.inference_mode():
        ids = tok(text, truncation=True, max_length=512, return_tensors="pt").input_ids
        if ids.shape[1] < 8:
            return None
        lo = obs(ids).logits[0, :-1].float()
        lp = perf(ids).logits[0, :-1].float()
        target = ids[0, 1:]
        log_pp = torch.log_softmax(lp, -1)
        ppl = -log_pp.gather(1, target.unsqueeze(1)).mean()
        x_ppl = -(torch.softmax(lo, -1) * log_pp).sum(-1).mean()
        return float(ppl / x_ppl)


def probe_prob(text):
    """Linear probe on the fakespot RoBERTa's mean-pooled last-layer embedding, trained on a diverse labelled corpus
    (weights in model.json under "probe"; dev/calibrate2.py --probe). None when the model has no probe."""
    import math
    import torch
    from ai_score import load_models, prose_blocks
    w = load_models().get("probe")
    if not w:
        return None
    tok, model, _ = _fakespot()
    # the same canonical prose the probe was trained on (dev/embed_rows.py): no headings/tables/code, hard wraps joined,
    # look-alike letters, zero-width characters and typography normalised
    text = " ".join(prose_blocks(text)) or text
    with torch.inference_mode():
        enc = tok(text, truncation=True, max_length=512, return_tensors="pt")
        h = model.roberta(**enc).last_hidden_state[0].float().mean(0).tolist()
    z = w["intercept"] + sum(c * (x - m) / (s or 1) for x, m, s, c in zip(h, w["mean"], w["scale"], w["coef"]))
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


def features(text, which=("fakespot", "desklib", "binoculars")):
    """Average of each detector over evenly spaced chunks of the prose."""
    from ai_score import prose_blocks
    prose = "\n\n".join(prose_blocks(text))
    if len(prose.split()) < 20:
        return None
    parts = chunks(prose)
    fns = {"fakespot": fakespot_prob, "desklib": desklib_prob, "binoculars": binoculars_score}
    out = {}
    for name in which:
        vals = [v for v in (fns[name](p) for p in parts) if v is not None]
        if vals:
            out[name] = round(statistics.mean(vals), 5)
            if len(vals) > 1:
                out[f"{name}_max"] = round(max(vals) if name != "binoculars" else min(vals), 5)
    out["chunks_scored"] = len(parts)
    # the probe was trained on the first 512 tokens of each whole text, so it is applied the same way (not chunk-averaged)
    pr = probe_prob(text)
    if pr is not None:
        out["probe"] = round(pr, 5)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--json")
    args = ap.parse_args()
    if not available():
        sys.exit("Needs: pip install torch transformers")
    from extract_text import extract
    f = features(extract(args.input))
    if args.json:
        Path(args.json).write_text(json.dumps(f, indent=2), encoding="utf-8")
    print(json.dumps(f, indent=2))


if __name__ == "__main__":
    main()
