#!/usr/bin/env python3
"""Mean-pooled last-layer embeddings of the canonicalised prose (ai_score.prose_blocks joined by a space: no headings/tables/code,
hard wraps joined, look-alike letters and typography normalised, exactly what the live scorer feeds the probe);
the first version embedded the raw text and learned formatting artefacts (dev/runs/emb_fakespot_raw.npz).
Old docstring: mean-pooled last-layer embeddings of the fakespot RoBERTa detector for every training/test text (cached).
Rows: calibrate2.load_rows(external, modern) + external benchmark + Gemini rounds 1-2 + API texts + human web.
Output: runs/emb_fakespot.npz (keys md5-12 -> 768-d float16) written incrementally; rerun resumes."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "humanize" / "scripts"))
import external_eval as ee  # noqa: E402
import ai_score  # noqa: E402
import calibrate2 as c2  # noqa: E402
NAME = "fakespot-ai/roberta-base-ai-text-detection-v1"
OUT = ROOT / "runs" / "emb_fakespot.npz"

def all_texts():
    texts = [r["text"] for r in c2.load_rows(True, True)]
    texts += [t["text"] for t in ee.load_texts()]
    for f in ("gemini_rows.json", "gemini2_rows.json", "api_rows.json", "humanweb_rows.json"):
        texts += [r["text"] for r in json.load(open(ROOT / "runs" / f, encoding="utf-8"))]
    seen, out = set(), []
    for t in texts:
        k = ee.key(t)
        if k not in seen:
            seen.add(k); out.append((k, t))
    return out

def main():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    items = all_texts()
    print(len(items), "texts", flush=True)
    done = {}   # always recompute (the raw-text version lives in emb_fakespot_raw.npz)
    tok = AutoTokenizer.from_pretrained(NAME)
    m = AutoModelForSequenceClassification.from_pretrained(NAME, use_safetensors=True).eval()
    torch.set_num_threads(12)
    todo = [(k, t) for k, t in items if k not in done]
    for i in range(0, len(todo), 8):
        batch = [(k, " ".join(ai_score.prose_blocks(t)) or t) for k, t in todo[i:i + 8]]
        enc = tok([t for _, t in batch], truncation=True, max_length=512, padding=True, return_tensors="pt")
        with torch.inference_mode():
            h = m.roberta(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            e = ((h * mask).sum(1) / mask.sum(1)).numpy().astype(np.float16)
        for (k, _), v in zip(batch, e):
            done[k] = v
        if (i // 8) % 25 == 0:
            np.savez(OUT, **done); print(i, "/", len(todo), flush=True)
    np.savez(OUT, **done); print("done", flush=True)

if __name__ == "__main__":
    main()
