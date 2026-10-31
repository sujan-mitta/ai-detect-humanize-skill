#!/usr/bin/env python3
"""Standalone test of extra pretrained AI-text detectors on a labelled pool (cached per model).
Usage: python dev/extra_detectors.py MODEL_ID [MODEL_ID ...]
Pool: external benchmark (316), human web reports/technical (80), ext2 sample (AI+human), API texts sample.
Output per model: runs/extra_<name>.json {key: prob_AI}; prints AUC per source and recall at 1% / 5% human FP.
For each model the AI label is the one whose AUC > 0.5 on the pool (printed), since label order differs between models."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "plugin" / "skills" / "humanize" / "scripts"))
import external_eval as ee  # noqa: E402

def pool():
    rows = []
    for t in ee.load_texts():
        rows.append((t["text"], t["label"], "ext"))
    for r in json.load(open(ROOT / "runs" / "humanweb_rows.json", encoding="utf-8")):
        rows.append((r["text"], 0, "humanweb"))
    for r in json.load(open(ROOT / "runs" / "ext2_rows.json", encoding="utf-8")):
        rows.append((r["text"], r["label"], "ext2"))
    api = json.load(open(ROOT / "runs" / "api_rows.json", encoding="utf-8"))
    for i, r in enumerate(api):
        if i % 2 == 0:
            rows.append((r["text"], 1, "api_sci" if r["source"] == "api_sci" else "api"))
    for r in json.load(open(ROOT / "runs" / "gemini2_rows.json", encoding="utf-8")):
        rows.append((r["text"], 1, "gemini2"))
    return rows

def load(name):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForSequenceClassification.from_pretrained(name).eval()
    return tok, m

def probs(tok, m, text, max_len=512, chunks=3):
    import torch
    ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"]
    step = max_len - 2
    parts = [ids[i:i + step] for i in range(0, max(1, len(ids)), step)][:chunks]
    out = []
    for p in parts:
        if len(p) < 20 and out:
            continue
        x = torch.tensor([[tok.cls_token_id] + p + [tok.sep_token_id]])
        with torch.inference_mode():
            out.append(torch.softmax(m(input_ids=x).logits[0].float(), -1).numpy())
    return np.mean(out, axis=0)

def main():
    rows = pool()
    print(len(rows), "pool texts", flush=True)
    for name in sys.argv[1:]:
        out = ROOT / "runs" / ("extra_" + name.replace("/", "_") + ".json")
        cache = json.loads(out.read_text()) if out.exists() else {}
        tok = m = None
        for i, (t, y, src) in enumerate(rows):
            k = ee.key(t)
            if k in cache:
                continue
            if m is None:
                tok, m = load(name)
            cache[k] = probs(tok, m, t).tolist()
            if i % 50 == 0:
                out.write_text(json.dumps(cache)); print(name, i, flush=True)
        out.write_text(json.dumps(cache))
        P = np.array([cache[ee.key(t)] for t, _, _ in rows]); y = np.array([r[1] for r in rows]); src = np.array([r[2] for r in rows])
        best = max(range(P.shape[1]), key=lambda j: roc_auc_score(y, P[:, j]))
        s = P[:, best]
        print(f"\n## {name}: AI label index {best}")
        h = s[y == 0]
        t1, t5 = np.quantile(h, .99), np.quantile(h, .95)
        print(f"  overall AUC {roc_auc_score(y, s):.3f} | recall at 1% human FP {np.mean(s[y==1] > t1):.0%} | at 5% {np.mean(s[y==1] > t5):.0%}")
        for g in sorted(set(src)):
            sel = src == g
            a = f"AUC {roc_auc_score(y[sel], s[sel]):.3f}" if len(set(y[sel])) > 1 else f"AI recall@5%FP {np.mean(s[sel] > t5):.0%}"
            print(f"  {g:10s} n={sel.sum():4d} {a}  mean {s[sel].mean():.2f}")

if __name__ == "__main__":
    main()
