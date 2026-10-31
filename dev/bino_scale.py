#!/usr/bin/env python3
"""Does a bigger Binoculars pair separate modern AI text from human text better than SmolLM2-360M?
Usage: python dev/bino_scale.py OBSERVER PERFORMER [OBSERVER PERFORMER ...]
Pool: sample of the labelled pool (ext benchmark, ext2, API, Gemini round 2, human web). Caches scores in runs/bino_<obs>.json."""
import json, random, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "plugin" / "skills" / "humanize" / "scripts"))
import external_eval as ee  # noqa: E402

def sample():
    random.seed(5)
    ext = ee.load_texts(); eh = [t for t in ext if t["label"] == 0]; ea = [t for t in ext if t["label"] == 1]
    ex2 = json.load(open(ROOT / "runs" / "ext2_rows.json", encoding="utf-8"))
    api = json.load(open(ROOT / "runs" / "api_rows.json", encoding="utf-8"))
    g2 = json.load(open(ROOT / "runs" / "gemini2_rows.json", encoding="utf-8"))
    hw = json.load(open(ROOT / "runs" / "humanweb_rows.json", encoding="utf-8"))
    pick = lambda xs, n: random.sample(xs, min(n, len(xs)))
    rows = [(t["text"], 0, "ext_h") for t in pick(eh, 60)] + [(t["text"], 1, "ext_ai") for t in pick(ea, 60)]
    rows += [(r["text"], 0, "ext2_h") for r in pick([r for r in ex2 if r["label"] == 0], 50)]
    rows += [(r["text"], 1, "ext2_ai") for r in pick([r for r in ex2 if r["label"] == 1], 60)]
    rows += [(r["text"], 1, "api") for r in pick(api, 60)] + [(r["text"], 1, "gemini2") for r in pick(g2, 60)]
    rows += [(r["text"], 0, "humanweb") for r in hw]
    return rows

def run(obs_name, perf_name, rows):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = ROOT / "runs" / ("bino_" + obs_name.replace("/", "_") + ".json")
    cache = json.loads(out.read_text()) if out.exists() else {}
    tok = obs = perf = None
    for i, (t, y, s) in enumerate(rows):
        k = ee.key(t)
        if k in cache:
            continue
        if obs is None:
            tok = AutoTokenizer.from_pretrained(obs_name)
            obs = AutoModelForCausalLM.from_pretrained(obs_name, torch_dtype=torch.float32).eval()
            perf = AutoModelForCausalLM.from_pretrained(perf_name, torch_dtype=torch.float32).eval()
        with torch.inference_mode():
            ids = tok(t, truncation=True, max_length=512, return_tensors="pt").input_ids
            lo, lp = obs(ids).logits[0, :-1].float(), perf(ids).logits[0, :-1].float()
            lpp = torch.log_softmax(lp, -1)
            ppl = -lpp.gather(1, ids[0, 1:].unsqueeze(1)).mean()
            xppl = -(torch.softmax(lo, -1) * lpp).sum(-1).mean()
            cache[k] = float(ppl / xppl)
        if i % 40 == 0:
            out.write_text(json.dumps(cache)); print(obs_name, i, flush=True)
    out.write_text(json.dumps(cache))
    return np.array([cache[ee.key(t)] for t, _, _ in rows])

def report(name, rows, score):
    y = np.array([r[1] for r in rows]); src = np.array([r[2] for r in rows]); s = -score  # low Binoculars = AI
    h = s[y == 0]; t1, t5 = np.quantile(h, .99), np.quantile(h, .95)
    print(f"\n## {name}: overall AUC {roc_auc_score(y, s):.3f}")
    for g in sorted(set(src[y == 1])):
        m = (src == g) | (y == 0)
        print(f"  {g:8s} AUC vs all humans {roc_auc_score(y[m], s[m]):.3f} | recall at 5% FP {np.mean(s[src == g] > t5):.0%} | at 1% FP {np.mean(s[src == g] > t1):.0%}")

if __name__ == "__main__":
    rows = sample(); print(len(rows), "texts", flush=True)
    a = sys.argv[1:]
    for o, p in zip(a[::2], a[1::2]):
        report(o, rows, run(o, p, rows))
