#!/usr/bin/env python3
"""Predictability-only "legacy" head for older / weaker generators (GPT-2, OPT, LLaMA-1, BLOOM...).

Why a second head: style cues are generator-specific (measured: mixing benchmark data into the
style model dropped its AUC 0.96 -> 0.78), but predictability signals (GPT-2 features,
Binoculars, the two neural classifiers) are not tied to one writing style. So the style
model stays trained on modern chat-LLM text, and a separate head that sees ONLY
predictability features is trained on ours + a generator-diverse external subset.
The two are combined as max(modern, legacy) after the legacy score is rescaled so that its
own verdict threshold (<=1% of held-out human texts above it) lands on the modern one.

Usage: python dev/legacy_head.py [--write]
Reports: leave-one-genre-out combined vs modern on our corpus (the false-positive cost),
and the external benchmark eval (COLING-dev + RAID; texts disjoint from the training subset).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "ai-detect-humanize" / "scripts"))
import calibrate as c  # noqa: E402
import external_eval as ee  # noqa: E402
import ai_score  # noqa: E402

NAMES = c.DEEP_FEATURES + c.NEURAL_FEATURES
FP_TARGET = 0.01
HOLD_GEMINI = False


def load_ext():
    sub = json.loads((ROOT / "runs" / "legacy_subset.json").read_text(encoding="utf-8"))
    extra = ROOT / "runs" / "legacy_subset_extra.json"  # Llama-3 / Gemma rows (legacy_extra.py)
    if extra.exists():
        sub += json.loads(extra.read_text(encoding="utf-8"))
    gem = ROOT / "runs" / "gemini_rows.json"  # Gemini texts, files 01-03 only (04-05 stay held out)
    if gem.exists() and not HOLD_GEMINI:
        sub += [r for r in json.loads(gem.read_text(encoding="utf-8")) if r["split"] == "train"]
    cache = ee.load_cache()
    rows = []
    for r in sub:
        rec = cache.get(ee.key(r["text"]))
        if not rec or not rec.get("deep") or not rec.get("neural"):
            continue
        rows.append({"text": r["text"], "label": r["label"], "genre": "x:" + r["genre"].split(":")[0],
                     "style": "external", "deep": rec["deep"], "neural": rec["neural"],
                     "group": "ext:" + ee.key(r["text"])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    ours = [r for r in c.extract_features(c.load_corpus(), True) if r["genre"] != "extra" and r.get("deep") and r.get("neural")]
    ext = load_ext()
    print(f"{len(ours)} own rows, {len(ext)} external rows ({sum(r['label'] for r in ext)} AI)", flush=True)

    # --- modern (max tier) out-of-fold probabilities on our rows, leave-one-genre-out
    c.GROUPS = np.array([r["group"] for r in ours])
    c.GENRES = np.array([r["genre"] for r in ours])
    y_o = np.array([r["label"] for r in ours])
    names_max = c.STYLE_FEATURES + c.DEEP_FEATURES + c.NEURAL_FEATURES
    # templates feature must exist on rows: reuse calibrate2's out-of-fold computation
    import calibrate2
    calibrate2.template_features(ours, [])
    p_mod = c.cv_predict_gb(c.matrix(ours, names_max), y_o, names_max)

    # --- legacy head, leave-one-genre-out over ours + external sources
    allr = ours + ext
    c.GROUPS = np.array([r["group"] for r in allr])
    c.GENRES = np.array([r["genre"] for r in allr])
    y = np.array([r["label"] for r in allr])
    X = c.matrix(allr, NAMES)
    p_leg_all = c.cv_predict_gb(X, y, NAMES)
    p_leg = p_leg_all[:len(ours)]
    t_leg = float(np.quantile(p_leg[y_o == 0], 1 - FP_TARGET))
    th_mod = c.verdict_thresholds(p_mod, y_o)["ai"] / 100
    scale = th_mod / max(t_leg, 1e-6)
    comb = np.maximum(p_mod, np.minimum(1.0, p_leg * scale))
    print(f"\nlegacy head threshold (1% of held-out own humans): {t_leg:.3f}; modern threshold {th_mod:.3f}; scale {scale:.2f}")
    print(f"legacy head alone, LOGO over own + external sources: AUC {roc_auc_score(y, p_leg_all):.3f}; "
          f"on external rows only {roc_auc_score(y[len(ours):], p_leg_all[len(ours):]):.3f}")
    for name, p in (("modern max", p_mod), ("combined", comb)):
        thr = c.verdict_thresholds(p, y_o)["ai"] / 100
        h = {g: np.mean(p[(y_o == 0) & (c.GENRES[:len(ours)] == g)] >= thr) for g in sorted(set(c.GENRES[:len(ours)]))}
        a = {g: np.mean(p[(y_o == 1) & (c.GENRES[:len(ours)] == g)] >= thr) for g in sorted(set(c.GENRES[:len(ours)]))}
        print(f"\n## {name} on own corpus (LOGO), threshold re-derived at 2% human FP = {thr:.3f}: AUC {roc_auc_score(y_o, p):.3f}, "
              f"AI caught {np.mean(p[y_o == 1] >= thr):.0%}, human flagged {np.mean(p[y_o == 0] >= thr):.1%}")
        print("  human flagged: " + " | ".join(f"{g} {v:.0%}" for g, v in h.items()))
        print("  AI caught    : " + " | ".join(f"{g} {v:.0%}" for g, v in a.items()))

    # --- final legacy model + external benchmark
    legacy = c.fit_gb(X, y, NAMES)
    legacy["threshold"] = round(t_leg * 100, 2)
    legacy["gate"] = 80
    legacy["scale"] = round(scale, 4)
    texts = ee.load_texts()
    cache = ee.load_cache()
    ye = np.array([t["label"] for t in texts])
    cur = ai_score.load_models()
    pm, pl = [], []
    for t in texts:
        rec = cache[ee.key(t["text"])]
        pm.append(ee.score(t["text"], rec, "max") or 0.0)
        vals = {**(rec.get("deep") or {}), **(rec.get("neural") or {})}
        v = [vals.get(n) for n in NAMES]
        pl.append(ai_score.gb_score(legacy, v, "legacy")["score"] / 100)
    pm, pl = np.array(pm), np.array(pl)
    th_cur = cur["max"]["thresholds"]["ai"] / 100
    pc = np.maximum(pm, np.minimum(1.0, pl * scale))
    print("\n## EXTERNAL benchmark (316 texts, COLING-dev + RAID), shipped modern threshold", th_cur)
    for name, p in (("modern max", pm), ("legacy head alone (rescaled)", np.minimum(1.0, pl * scale)), ("combined", pc)):
        print(f"  {name:30s} AUC {roc_auc_score(ye, p):.3f} | human flagged {np.mean(p[ye == 0] >= th_cur):.1%} | AI caught {np.mean(p[ye == 1] >= th_cur):.1%}")
    if args.write:
        path = ai_score.MODEL_PATH
        m = json.loads(path.read_text(encoding="utf-8"))
        m["legacy"] = legacy
        path.write_text(json.dumps(m, separators=(",", ":")), encoding="utf-8")
        print("wrote legacy head into", path)


if __name__ == "__main__":
    main()
