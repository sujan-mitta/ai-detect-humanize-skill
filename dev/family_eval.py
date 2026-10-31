#!/usr/bin/env python3
"""Recall on modern non-Claude model families (GPT-4o/4/3.5, Llama-3, Gemma, Mixtral, Cohere).

Texts come from COLING-2025 train rows that neither the shipped model nor its legacy head saw
(excluded: the legacy subset and the external eval set). Humans from the same pool give the FP rate.
Usage: python dev/family_eval.py [--per 15]"""
import argparse, collections, json, random, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "plugin" / "skills" / "humanize" / "scripts"))
import external_eval as ee  # noqa: E402
import ai_score  # noqa: E402
import legacy_features as lf  # noqa: E402

FAMILIES = {"gpt-4o/4": ("gpt4o", "gpt4"), "gpt-3.5": ("gpt-3.5-turbo", "gpt-35"), "llama-3": ("llama3-70b", "llama3-8b"),
            "gemma": ("gemma2-9b-it", "gemma-7b-it"), "mixtral": ("mixtral-8x7b",), "cohere": ("cohere",)}

ap = argparse.ArgumentParser()
ap.add_argument("--per", type=int, default=15)
a = ap.parse_args()
random.seed(99)
rows = [json.loads(l) for l in (ee.EXT / "coling_train.jsonl").read_text(encoding="utf-8").splitlines()]
rows = [r for r in rows if 120 <= len(r["text"].split()) <= 700]
used = {ee.key(r["text"]) for r in lf.subset()} | {ee.key(t["text"]) for t in ee.load_texts()}
rows = [r for r in rows if ee.key(r["text"]) not in used]
test = []
for fam, gens in FAMILIES.items():
    pool = [r for r in rows if r["label"] == 1 and r["generator"] in gens]
    for r in random.sample(pool, min(a.per * len(gens), len(pool)))[:a.per * 2 // len(gens) * len(gens) if False else a.per * len(gens)]:
        test.append(dict(r, family=fam))
hum = random.sample([r for r in rows if r["label"] == 0], 90)
test += [dict(r, family="human") for r in hum]
print(len(test), "texts", flush=True)
cache = ee.compute_features(test, True, True)
res = collections.defaultdict(list)
for kind in ("max",):
    models = ai_score.load_models(); thr = models[kind]["thresholds"]["ai"] / 100
    for t in test:
        rec = cache[ee.key(t["text"])]
        p = ee.score(t["text"], rec, kind)
        res[t["family"]].append(p)
    for fam, v in res.items():
        v = np.array([x for x in v if x is not None])
        print(f"{fam:10s} n={len(v):3d}  flagged at verdict {np.mean(v >= thr):.0%}  at 'likely AI' (0.60) {np.mean(v >= .6):.0%}  mean {v.mean():.2f}")
json.dump({k: [x for x in v] for k, v in res.items()}, open(ROOT / "runs" / "family_eval.json", "w"))
