#!/usr/bin/env python3
"""Frozen-benchmark floor: fail if a model is worse than the shipped numbers (max mode).

Uses cached features (runs/external_features.jsonl, runs/gemini_rows.json), so it runs in seconds.
Usage: python dev/regression_floor.py [MODEL.json]     (default: the shipped model.json)
Floors are the shipped v8 values (2026-10-29) with a small tolerance for noise."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ai-detect-humanize" / "scripts"))
import ai_score  # noqa: E402
import external_eval as ee  # noqa: E402
import legacy_features as lf  # noqa: E402

SHIPPED = HERE.parent / "ai-detect-humanize" / "scripts" / "model.json"
FAM = {"gpt4": ("gpt4o", "gpt4"), "gpt3.5": ("gpt-3.5-turbo", "gpt-35"), "llama3": ("llama3-70b", "llama3-8b"),
       "gemma": ("gemma2-9b-it", "gemma-7b-it"), "mixtral+cohere": ("mixtral-8x7b", "cohere")}

# (metric, floor, direction): "min" = must be >= floor, "max" = must be <= ceiling
FLOORS = {
    "ext_ai_verdict": (0.78, "min"), "ext_ai_60": (0.84, "min"),
    "ext_human_verdict": (0.025, "max"), "ext_human_60": (0.075, "max"),
    "gemini_verdict": (0.78, "min"), "gemini_60": (0.89, "min"),
    "fresh_human_60": (0.045, "max"), "fresh_human_verdict": (0.01, "max"),
    "llama3_verdict": (0.66, "min"), "gemma_verdict": (0.57, "min"),
    "gpt4_verdict": (0.80, "min"), "gpt3.5_verdict": (0.89, "min"),
}


def load_data():
    cache = ee.load_cache()
    ext = ee.load_texts()
    allg = json.load(open(HERE / "runs" / "gemini_rows.json", encoding="utf-8"))
    gem = [r for r in allg if r["split"] == "test"]
    rows = [json.loads(l) for l in (ee.EXT / "coling_train.jsonl").read_text(encoding="utf-8").splitlines()]
    used = ({ee.key(r["text"]) for r in lf.subset()}
            | {ee.key(r["text"]) for r in json.load(open(HERE / "runs" / "legacy_subset_extra.json", encoding="utf-8"))}
            | {ee.key(t["text"]) for t in ext} | {ee.key(r["text"]) for r in allg})

    def ok(r):
        k = ee.key(r["text"])
        return k in cache and "neural" in cache[k] and k not in used
    fam = {k: [r for r in rows if r["label"] == 1 and r["generator"] in g and ok(r)] for k, g in FAM.items()}
    hum = [r for r in rows if r["label"] == 0 and ok(r)]
    return cache, ext, gem, fam, hum


def metrics(model_path=SHIPPED, data=None):
    cache, ext, gem, fam, hum = data or load_data()
    ai_score.MODEL_PATH = Path(model_path)
    ai_score._MODELS = None
    thr = ai_score.load_models()["max"]["thresholds"]["ai"] / 100

    def sc(rs):
        return np.array([ee.score(t["text"], cache[ee.key(t["text"])], "max") or 0 for t in rs])
    h, a, g, fh = (sc([t for t in ext if t["label"] == 0]), sc([t for t in ext if t["label"] == 1]), sc(gem), sc(hum))
    m = {"threshold": thr,
         "ext_ai_verdict": np.mean(a >= thr), "ext_ai_60": np.mean(a >= .6),
         "ext_human_verdict": np.mean(h >= thr), "ext_human_60": np.mean(h >= .6),
         "gemini_verdict": np.mean(g >= thr), "gemini_60": np.mean(g >= .6),
         "fresh_human_60": np.mean(fh >= .6), "fresh_human_verdict": np.mean(fh >= thr)}
    for k, v in fam.items():
        if v:
            s = sc(v)
            m[f"{k}_verdict"], m[f"{k}_60"] = np.mean(s >= thr), np.mean(s >= .6)
    return m


def check(m):
    bad = []
    for k, (lim, kind) in FLOORS.items():
        v = m.get(k)
        if v is None or (v < lim if kind == "min" else v > lim):
            bad.append(f"{k} {v if v is None else round(v, 3)} (limit {'>=' if kind == 'min' else '<='} {lim})")
    return bad


def main():
    m = metrics(sys.argv[1] if len(sys.argv) > 1 else SHIPPED)
    for k, v in m.items():
        print(f"  {k:22s} {v:.3f}")
    bad = check(m)
    print("\nFAIL:\n  " + "\n  ".join(bad) if bad else "\nall floors met")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
