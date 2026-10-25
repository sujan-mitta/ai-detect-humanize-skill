#!/usr/bin/env python3
"""Fit and evaluate the AI-likeness model on the calibration corpus.

Usage: python dev/calibrate.py [--no-deep] [--write]

1. Loads dev/corpus/human.jsonl and dev/corpus/ai/<model>/<genre>__<style>__NN.txt
2. Extracts style features (ai_score.analyze) and, unless --no-deep, GPT-2
   features (perplexity.features). Features are cached in dev/runs/.
3. Reports 5-fold cross-validated accuracy / AUC for:
     - the hand-tuned heuristic score (baseline)
     - a logistic regression on style features
     - a logistic regression on style + deep features
   with per-genre and per-style breakdowns.
4. With --write, fits on all training data and saves the models to
   ai-detect-humanize/scripts/model.json, which ai_score.py then uses.

The "extra" genre (emails, cover letters, LinkedIn...) has no human
counterpart in the corpus, so it is held out of training and only reported
as a detection rate, to avoid the model learning genre instead of authorship.
"""
import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT.parent / "ai-detect-humanize" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import ai_score  # noqa: E402

STYLE_FEATURES = [
    "mean_sentence_length", "sentence_length_cv", "paragraph_length_cv", "wrapup_paragraph_share",
    "stock_hits_per_1k", "hedges_per_1k", "participial_tails_per_1k", "contrast_frames_per_1k",
    "transition_start_ratio", "template_opener_ratio", "top_opener_share", "triplets_per_1k",
    "fragment_ratio", "performed_voice_tics", "contractions_per_1k", "em_dashes_per_1k",
    "chatbot_artifacts", "repeated_4gram_ratio", "mattr", "avg_word_length", "commas_per_sentence",
    "semicolons_per_1k", "parens_per_1k", "questions_per_1k", "exclaims_per_1k", "first_person_per_1k",
    "digits_per_1k", "proper_nouns_per_1k", "bold_leads_per_1k", "fiction_cliches_per_1k", "dialogue_ratio",
]
# Features that are AI tells in the wild (ai-patterns.md). Their weights are
# constrained to be >= 0 so a quirk of the corpus (e.g. the Claude samples
# using few em dashes) can't teach the model that the tell means "human".
AI_DIRECTION = {
    "stock_hits_per_1k", "hedges_per_1k", "participial_tails_per_1k", "contrast_frames_per_1k",
    "transition_start_ratio", "template_opener_ratio", "triplets_per_1k", "performed_voice_tics",
    "em_dashes_per_1k", "chatbot_artifacts", "wrapup_paragraph_share", "bold_leads_per_1k",
    "fiction_cliches_per_1k",
}
# Heavy-tailed counts are log-scaled (log1p) so one habit - e.g. a couple of
# parentheses in a short text - can't swing the whole score.
LOG_FEATURES = {f for f in STYLE_FEATURES if f.endswith("_per_1k")} | {"chatbot_artifacts", "performed_voice_tics"}
DEEP_FEATURES = ["log_perplexity", "top1_fraction", "top10_fraction", "rare_fraction", "sentence_ppl_cv"]
# "max" tier: pretrained AI-text classifiers + Binoculars (scripts/neural.py)
# Binoculars (CPU-sized SmolLM2 pair) is weak alone (AUC 0.42) but in the ensemble it
# lifts held-out genres from 79% to 86% caught, so the max tier keeps all three.
NEURAL_FEATURES = ["fakespot", "desklib", "binoculars"]
# Binoculars should be lower for machine text, but with the small CPU-sized
# SmolLM2 pair it measured AUC 0.42 stand-alone (weak and inverted), so it is
# left unconstrained and the trees decide how much to trust it.
AI_NEGATIVE = set()


def load_corpus():
    rows = [json.loads(l) for l in (ROOT / "corpus" / "human.jsonl").read_text(encoding="utf-8").splitlines()]
    for r in rows:
        r["style"] = "human"
        r["id"] = r["source"] + ":" + hashlib.md5(r["text"].encode()).hexdigest()[:10]
    for f in sorted((ROOT / "corpus" / "ai").glob("*/*.txt")):
        genre, style, _ = f.stem.split("__")
        rows.append({"text": f.read_text(encoding="utf-8"), "label": 1, "genre": genre, "style": style,
                     "source": f"ai:{f.parent.name}", "id": f"{f.parent.name}/{f.stem}"})
    return rows


def extract_features(rows, deep):
    cache_path = ROOT / "runs" / "features.jsonl"
    cache = {}
    if cache_path.exists():
        for l in cache_path.read_text(encoding="utf-8").splitlines():
            c = json.loads(l)
            cache[c["id"]] = c
    if deep:
        import perplexity
        if not perplexity.available():
            sys.exit("deep features need torch + transformers (or pass --no-deep)")
    changed = False
    for i, r in enumerate(rows):
        c = cache.get(r["id"], {"id": r["id"]})
        if "feat" not in c:
            a = ai_score.analyze(r["text"])
            c["feat"] = {k: a["raw"][k] for k in STYLE_FEATURES}
            c["heuristic"] = a["heuristic_score"]
            changed = True
        if deep and "deep" not in c:
            import perplexity
            c["deep"] = perplexity.features(r["text"])
            changed = True
            if i % 25 == 0:
                print(f"  deep features {i}/{len(rows)}", flush=True)
        cache[r["id"]] = c
        r.update({k: c.get(k) for k in ("feat", "heuristic", "deep")})
    if changed:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("\n".join(json.dumps(c) for c in cache.values()), encoding="utf-8")
    neural_path = ROOT / "runs" / "neural.jsonl"
    if neural_path.exists():
        neural = {}
        for line in neural_path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            neural[d["id"]] = d["neural"]
        for r in rows:
            r["neural"] = neural.get(r["id"])
    return rows


def matrix(rows, names, deep=None):
    """Feature matrix for `names`, taking each from the style, GPT-2 or neural features."""
    X = []
    for r in rows:
        v = []
        for k in names:
            if k in STYLE_FEATURES:
                x = float(r["feat"][k])
                v.append(math.log1p(max(0.0, x)) if k in LOG_FEATURES else x)
            else:
                src = (r.get("deep") if k in DEEP_FEATURES else r.get("neural")) or {}
                v.append(float(src[k]) if src.get(k) is not None else np.nan)
        X.append(v)
    X = np.array(X)
    # impute missing deep values with column medians
    med = np.nanmedian(X, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(med, idx[1])
    return X


def fit_lr(X, y, names, C=0.5):
    """Class-balanced L2 logistic regression; AI_DIRECTION weights bounded >= 0."""
    w = np.where(y == 1, len(y) / (2 * y.sum()), len(y) / (2 * (len(y) - y.sum())))

    def loss(theta):
        b, coef = theta[0], theta[1:]
        z = X @ coef + b
        p = 1 / (1 + np.exp(-z))
        nll = np.sum(w * (np.logaddexp(0, z) - y * z))
        grad_z = w * (p - y)
        return nll + 0.5 / C * coef @ coef, np.concatenate([[grad_z.sum()], X.T @ grad_z + coef / C])

    bounds = [(None, None)] + [(0, None) if n in AI_DIRECTION else (None, None) for n in names]
    res = minimize(loss, np.zeros(X.shape[1] + 1), jac=True, method="L-BFGS-B", bounds=bounds)
    return res.x[1:], res.x[0]


def cv_predict(X, y, names, C=0.5, seed=0):
    probs = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        coef, b = fit_lr(sc.transform(X[tr]), y[tr], names, C)
        probs[te] = 1 / (1 + np.exp(-(sc.transform(X[te]) @ coef + b)))
    return probs


def make_gb(names):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, min_samples_leaf=20, class_weight="balanced", random_state=0,
        monotonic_cst=[1 if n in AI_DIRECTION or n in ("fakespot", "desklib") else -1 if n in AI_NEGATIVE else 0
                       for n in names])


def cv_predict_gb(X, y, names, seed=0):
    """Out-of-fold probabilities, Platt-calibrated within each training fold."""
    from sklearn.linear_model import LogisticRegression
    probs = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        inner = np.zeros(len(tr))
        for itr, ite in StratifiedKFold(4, shuffle=True, random_state=seed).split(X[tr], y[tr]):
            inner[ite] = make_gb(names).fit(X[tr][itr], y[tr][itr]).decision_function(X[tr][ite])
        platt = LogisticRegression().fit(inner[:, None], y[tr])
        z = make_gb(names).fit(X[tr], y[tr]).decision_function(X[te])
        probs[te] = platt.predict_proba(z[:, None])[:, 1]
    return probs


def fit_gb(X, y, names):
    """Fit boosted trees and export them as plain JSON, so ai_score.py can run
    them without scikit-learn. Medians are kept for explaining scores."""
    # Platt scaling on out-of-fold margins: boosted trees are overconfident, and
    # a "100/100" on a few hundred training texts would overstate the evidence
    from sklearn.linear_model import LogisticRegression
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        oof[te] = make_gb(names).fit(X[tr], y[tr]).decision_function(X[te])
    platt = LogisticRegression().fit(oof[:, None], y)
    m = make_gb(names).fit(X, y)
    trees = []
    for (pred,) in m._predictors:
        trees.append([[int(n["feature_idx"]), round(float(n["num_threshold"]), 6), int(n["left"]), int(n["right"]),
                       int(n["is_leaf"]), round(float(n["value"]), 6), int(n["missing_go_to_left"])]
                      for n in pred.nodes])
    return {"type": "gb", "features": names, "log_features": sorted(LOG_FEATURES & set(names)),
            "baseline": float(np.ravel(m._baseline_prediction)[0]), "trees": trees,
            "platt": [round(float(platt.coef_[0][0]), 6), round(float(platt.intercept_[0]), 6)],
            "medians": np.median(X, axis=0).round(6).tolist()}


def verdict_thresholds(p, y, max_fp=0.05, max_fn=0.10):
    s = np.round(100 * p)
    t_ai = next(t for t in range(1, 101) if (s[y == 0] >= t).mean() <= max_fp)
    t_h = max([t for t in range(1, t_ai + 1) if (s[y == 1] < t).mean() <= max_fn] or [1])
    return {"ai": t_ai, "human": t_h,
            "human_said_human": round(float((s[y == 0] < t_h).mean()), 4), "human_said_ai": round(float((s[y == 0] >= t_ai).mean()), 4),
            "ai_said_ai": round(float((s[y == 1] >= t_ai).mean()), 4), "ai_said_human": round(float((s[y == 1] < t_h).mean()), 4)}


def first_words(text, n):
    """The first ~n words, keeping paragraph breaks and ending on a sentence."""
    m = list(re.finditer(r"\S+", text))
    if len(m) <= n:
        return text
    cut = text[:m[n - 1].end()]
    end = max(cut.rfind(". "), cut.rfind(".\n"), cut.rfind("? "), cut.rfind("! "))
    return cut[:end + 1] if end > len(cut) * 0.6 else cut


def length_curve(train, y, lengths=(50, 100, 150, 200, 300)):
    """Accuracy when only the first N words are available (style model, out-of-fold)."""
    print("\n## Accuracy vs text length (style model; trained on full texts, tested on the first N words)")
    X = matrix(train, STYLE_FEATURES, False)
    curve = {}
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
    models_ = [make_gb(STYLE_FEATURES).fit(X[tr], y[tr]) for tr, _ in folds]
    for n in lengths:
        cut = [dict(r, feat={k: ai_score.analyze(first_words(r["text"], n))["raw"][k] for k in STYLE_FEATURES})
               for r in train]
        Xc = matrix(cut, STYLE_FEATURES, False)
        p = np.zeros(len(y))
        for (tr, te), m in zip(folds, models_):
            p[te] = m.predict_proba(Xc[te])[:, 1]
        acc, auc = float(((p >= 0.5) == y).mean()), float(roc_auc_score(y, p))
        curve[n] = {"accuracy": round(acc, 4), "auc": round(auc, 4)}
        print(f"  first {n:4d} words: accuracy {acc:.1%} | AUC {auc:.3f}")
    return curve


def report(name, rows, probs, threshold=0.5):
    y = np.array([r["label"] for r in rows])
    pred = probs >= threshold
    acc = (pred == y).mean()
    auc = roc_auc_score(y, probs)
    fpr = pred[y == 0].mean()
    tpr = pred[y == 1].mean()
    print(f"\n## {name}\naccuracy {acc:.1%} | AUC {auc:.3f} | AI caught {tpr:.1%} | humans wrongly flagged {fpr:.1%}")
    groups = defaultdict(list)
    for r, p in zip(rows, probs):
        groups[(r["label"], r["genre"], r["style"])].append(p >= threshold)
    for (label, genre, style), v in sorted(groups.items()):
        what = "flagged as AI" if label == 0 else "caught"
        print(f"  {'AI   ' if label else 'human'} {genre:13s} {style:7s} n={len(v):3d}  {what}: {np.mean(v):.0%}")
    return {"accuracy": round(acc, 4), "auc": round(auc, 4), "tpr": round(tpr, 4), "fpr": round(fpr, 4)}


def fit_full(X, y, names, C=0.5):
    sc = StandardScaler().fit(X)
    coef, b = fit_lr(sc.transform(X), y, names, C)
    return {"features": names, "log_features": sorted(LOG_FEATURES & set(names)),
            "mean": sc.mean_.round(6).tolist(), "scale": sc.scale_.round(6).tolist(),
            "coef": coef.round(6).tolist(), "intercept": round(float(b), 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-deep", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--length-curve", action="store_true")
    args = ap.parse_args()
    deep = not args.no_deep

    rows = extract_features(load_corpus(), deep)
    # genres with no human counterpart are held out so the model can't learn genre
    held_out = {"extra", "report"}
    train = [r for r in rows if r["genre"] not in held_out]
    extra = [r for r in rows if r["genre"] in held_out]
    y = np.array([r["label"] for r in train])
    print(f"training rows: {len(train)} ({int(y.sum())} AI, {int((1 - y).sum())} human); held-out extra: {len(extra)}")

    results = {"n_train": len(train), "n_ai": int(y.sum()), "n_human": int((1 - y).sum())}
    results["heuristic"] = report("Hand-tuned heuristic (score >= 50)", train,
                                  np.array([r["heuristic"] / 100 for r in train]))
    Xs = matrix(train, STYLE_FEATURES, False)
    results["style_lr_cv"] = report("Logistic regression: style features (5-fold CV)", train, cv_predict(Xs, y, STYLE_FEATURES))
    oof = {"style": cv_predict_gb(Xs, y, STYLE_FEATURES)}
    results["style_cv"] = report("Boosted trees: style features (5-fold CV) - deployed", train, oof["style"])
    lr_style = fit_full(Xs, y, STYLE_FEATURES)
    models = {"style": fit_gb(Xs, y, STYLE_FEATURES)}
    sets = [("style", STYLE_FEATURES, False)]
    if deep:
        names = STYLE_FEATURES + DEEP_FEATURES
        Xd = matrix(train, names, True)
        results["deep_lr_cv"] = report("Logistic regression: style + GPT-2 features (5-fold CV)", train, cv_predict(Xd, y, names))
        oof["deep"] = cv_predict_gb(Xd, y, names)
        results["deep_cv"] = report("Boosted trees: style + GPT-2 features (5-fold CV) - deployed", train, oof["deep"])
        models["deep"] = fit_gb(Xd, y, names)
        sets.append(("deep", names, True))
        if sum(1 for r in train if r.get("neural")) >= 0.95 * len(train):
            names3 = STYLE_FEATURES + DEEP_FEATURES + NEURAL_FEATURES
            Xm = matrix(train, names3)
            oof["max"] = cv_predict_gb(Xm, y, names3)
            results["max_cv"] = report("Boosted trees: style + GPT-2 + neural detectors (5-fold CV) - deployed", train, oof["max"])
            models["max"] = fit_gb(Xm, y, names3)
            sets.append(("max", names3, True))

    # Verdict thresholds from out-of-fold scores: "likely AI" needs a score that at
    # most 5% of human texts reach; "likely human" a score below which at most 10%
    # of AI texts fall. In between is "mixed / uncertain".
    for key, p in oof.items():
        th = verdict_thresholds(p, y)
        models[key]["thresholds"] = th
        results[f"{key}_thresholds"] = th
        print(f"\n## {key} verdict bands: likely AI >= {th['ai']:.0f}, likely human < {th['human']:.0f}")
        print(f"  humans: {th['human_said_human']:.0%} 'likely human', {th['human_said_ai']:.0%} 'likely AI' | "
              f"AI: {th['ai_said_ai']:.0%} 'likely AI', {th['ai_said_human']:.0%} 'likely human'")

    if args.length_curve:
        results["length_curve"] = length_curve(train, y)

    # held-out genres, scored by models fit on all training data
    if extra:
        print("\n## Held-out genres (emails, cover letters, LinkedIn, blogs, business reports) - all AI")
        line = []
        for key, names, d in sets:
            gb = make_gb(names).fit(matrix(train, names, d), y)
            caught = float((gb.predict_proba(matrix(extra, names, d))[:, 1] >= 0.5).mean())
            results[f"extra_{key}_caught"] = round(caught, 4)
            line.append(f"{key} model caught {caught:.0%}")
        print("  " + " | ".join(line))

    print("\n## Logistic-regression weights, for reading (standardized; + means pushes toward AI)")
    for n, c in sorted(zip(STYLE_FEATURES, lr_style["coef"]), key=lambda t: -abs(t[1])):
        print(f"  {n:28s} {c:+.2f}")

    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs" / "calibration_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.write:
        out = SCRIPTS / "model.json"
        models["trained_on"] = {k: results[k] for k in ("n_train", "n_ai", "n_human")}
        models["cv"] = {k: results[k] for k in results if k.endswith("_cv")}
        out.write_text(json.dumps(models, separators=(",", ":")), encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
