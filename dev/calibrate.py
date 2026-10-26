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
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT.parent / "ai-detect-humanize" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import ai_score  # noqa: E402

STYLE_FEATURES = [
    "mean_sentence_length", "sentence_length_cv", "wrapup_sentence_rate",
    "stock_hits_per_1k", "hedges_per_1k", "participial_tails_per_1k", "contrast_frames_per_1k",
    "transition_start_ratio", "template_opener_ratio", "top_opener_share", "triplets_per_1k",
    "fragment_ratio", "performed_voice_tics", "contractions_per_1k", "em_dashes_per_1k",
    "chatbot_artifacts", "repeated_4gram_ratio", "mattr", "avg_word_length", "commas_per_sentence",
    "semicolons_per_1k", "parens_per_1k", "questions_per_1k", "exclaims_per_1k", "first_person_per_1k",
    "digits_per_1k", "proper_nouns_per_1k", "bold_leads_per_1k", "fiction_cliches_per_1k", "dialogue_ratio",
    "fillers_per_1k", "dropped_apostrophes_per_1k", "ellipses_per_1k", "personal_hedges_per_1k",
    "second_person_per_1k", "conj_start_ratio", "length_alternation", "template_prob",
]
# Features that are AI tells in the wild (ai-patterns.md). Their weights are
# constrained to be >= 0 so a quirk of the corpus (e.g. the Claude samples
# using few em dashes) can't teach the model that the tell means "human".
AI_DIRECTION = {
    "stock_hits_per_1k", "hedges_per_1k", "participial_tails_per_1k", "contrast_frames_per_1k",
    "transition_start_ratio", "template_opener_ratio", "triplets_per_1k", "performed_voice_tics",
    "em_dashes_per_1k", "chatbot_artifacts", "wrapup_sentence_rate", "bold_leads_per_1k",
    "fiction_cliches_per_1k", "template_prob", "fastdetect",
}
# Heavy-tailed counts are log-scaled (log1p) so one habit - e.g. a couple of
# parentheses in a short text - can't swing the whole score.
LOG_FEATURES = {f for f in STYLE_FEATURES if f.endswith("_per_1k")} | {"chatbot_artifacts", "performed_voice_tics"}
DEEP_FEATURES = ["log_perplexity", "top1_fraction", "top10_fraction", "rare_fraction", "sentence_ppl_cv",
                 "fastdetect", "mean_entropy"]
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
        r["label"] = 0
        r["style"] = "human"
        r["id"] = r["source"] + ":" + hashlib.md5(r["text"].encode()).hexdigest()[:10]
    for f in sorted((ROOT / "corpus" / "ai").glob("*/*.txt")):
        genre, style, _ = f.stem.split("__")
        rows.append({"text": f.read_text(encoding="utf-8"), "label": 1, "genre": genre, "style": style,
                     "source": f"ai:{f.parent.name}", "id": f"{f.parent.name}/{f.stem}"})
    # AI text rewritten by an assistant ("humanized", "paraphrased", "casual") is still AI.
    # group = the original's id, so a rewrite and its source never straddle a CV split.
    for f in sorted((ROOT / "corpus" / "humanized_wild").glob("*.txt")) + sorted((ROOT / "corpus" / "humanized").glob("*.txt")):
        parts = f.stem.split("__")
        wild = f.parent.name == "humanized_wild"
        folder, src_stem = (parts[1], "__".join(parts[2:-1])) if wild else (parts[0], "__".join(parts[1:]))
        rows.append({"text": f.read_text(encoding="utf-8"), "label": 1, "genre": src_stem.split("__")[0],
                     "style": "humanized", "source": f"humanized:{parts[0]}", "id": f"{f.parent.name}/{f.stem}",
                     "group": f"{folder}/{src_stem}"})
    # Human texts are grouped by SOURCE (book, site, forum, arXiv category): several
    # passages come from each source, and letting the same book sit on both sides of
    # a CV split let the template model memorise individual authors (measured: 1.7% CV
    # false positives vs 20% on unseen books). AI texts are grouped by original text.
    for r in rows:
        if r["label"] == 0:
            kind = r["source"].split(":")[0]
            if kind.startswith("gutenberg"):
                r["group"] = "author:" + r["source"]          # one book = one author
            elif kind.startswith("paulgraham"):
                r["group"] = "author:paulgraham"              # all essays share one author
        r.setdefault("group", r["id"])                        # forums, Q&A, arXiv...: different authors
    return rows


GROUPS = None  # set in main(): one group per original text
GENRES = None  # set in main(): genre of each training row


def outer_splits(X, y, seed=0):
    """The evaluation protocol: LEAVE ONE GENRE OUT. Each genre in turn is held out whole
    (its human texts, AI texts and the AI rewrites of its AI texts), the model trains on
    all the others, and scores the held-out genre. That is the real situation when
    someone submits a kind of document the model hasn't seen. Ordinary grouped k-fold
    mixed every genre into every fold, which (measured) hid a 15-23 point drop in AI
    recall on new genres behind a rising AUC."""
    if GENRES is None:
        return folds(X, y, 5, seed)
    out = []
    for g in sorted(set(GENRES)):
        te = np.where(GENRES == g)[0]
        tr = np.where(GENRES != g)[0]
        if len(te):
            out.append((tr, te))
    return out


def folds(X, y, k=5, seed=0, idx=None):
    """Stratified folds that keep each original and its rewrites in the same fold."""
    if GROUPS is None:
        return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y))
    g = GROUPS if idx is None else GROUPS[idx]
    return list(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(X, y, g))


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
        r["stale"] = ai_score.typography_changed(r["text"])
        if "feat" not in c or c.get("feat_v") != FEAT_VERSION:
            a = ai_score.analyze(r["text"])
            c["feat"] = {k: a["raw"][k] for k in STYLE_FEATURES if k != "template_prob"}
            c["heuristic"] = a["heuristic_score"]
            c["feat_v"] = FEAT_VERSION
            changed = True
        if deep and r.get("want_deep", True) and ("deep" not in c or c.get("deep_v") != DEEP_VERSION):
            import perplexity
            c["deep"] = perplexity.features(r["text"])
            c["deep_v"] = DEEP_VERSION
            changed = True
            if i % 25 == 0:
                print(f"  deep features {i}/{len(rows)}", flush=True)
                cache_path.write_text("\n".join(json.dumps(x) for x in cache.values()), encoding="utf-8")
        cache[r["id"]] = c
        r.update({k: c.get(k) for k in ("feat", "heuristic", "deep")})
    if changed:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("\n".join(json.dumps(c) for c in cache.values()), encoding="utf-8")
    neural_path = ROOT / "runs" / "neural.jsonl"
    if neural_path.exists():
        neural = load_neural(neural_path)
        for r in rows:
            n, idx = neural.get(r["id"], (None, -1))
            r["neural"] = n if neural_is_valid(r["text"], n, idx) else None
    return rows


V1_LINES = 1328   # neural.jsonl lines written before the canonicalisation job began


def load_neural(path):
    """id -> (record, index of its LAST line in the file); later lines override earlier ones."""
    out = {}
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        d = json.loads(line)
        out[d["id"]] = (d["neural"], i)
    return out


def neural_is_valid(text, rec, idx):
    """A cached neural-feature record is reusable only if it was computed (a) with the fixed
    Binoculars and (b) on the same canonicalised text the scorer sees now. Records carry the
    canonicalisation version they were computed under ('cv'); older ones are judged by where
    they sit: lines before V1_LINES predate any canonicalisation, later lines used v1."""
    if not rec or not rec.get("binoculars_v2"):
        return False
    if rec.get("cv") == ai_score.CANON_VERSION:
        return True
    if idx >= V1_LINES:
        return not ai_score.detok_changed(text)
    return not ai_score.typography_changed(text)


# Bump when feature definitions change so cached rows are recomputed.
FEAT_VERSION = 5   # 5: detokenisation in canon_typography; layout/length-independent features
DEEP_VERSION = 3   # 3: canonicalised text; Fast-DetectGPT fields


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
    for tr, te in outer_splits(X, y, seed):
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
    for tr, te in outer_splits(X, y, seed):
        inner = np.zeros(len(tr))
        for itr, ite in folds(X[tr], y[tr], 4, seed, idx=tr):
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
    for tr, te in outer_splits(X, y, 0):
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


def fit_templates(texts, y, min_df=5, max_feats=30000, C=2.0):
    """Skeleton n-gram model (scripts/templates.py): tf-idf + logistic regression."""
    from scipy.sparse import csr_matrix
    from sklearn.linear_model import LogisticRegression
    import templates
    counts = [templates.ngram_counts(t) for t in texts]
    df = Counter(g for c in counts for g in c)
    vocab = [g for g, d in df.most_common(max_feats) if d >= min_df]
    index = {g: i for i, g in enumerate(vocab)}
    n = len(texts)
    idf = [math.log((1 + n) / (1 + df[g])) + 1 for g in vocab]
    rows_, cols, vals = [], [], []
    for i, c in enumerate(counts):
        for j, x in templates.vectorize(c, index, idf).items():
            rows_.append(i); cols.append(j); vals.append(x)
    X = csr_matrix((vals, (rows_, cols)), shape=(n, len(vocab)))
    lr = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(X, y)
    return {"vocab": vocab, "idf": [round(v, 5) for v in idf],
            "coef": [round(float(v), 5) for v in lr.coef_[0]], "intercept": round(float(lr.intercept_[0]), 5)}


def add_template_feature(train, extra, y):
    """Out-of-fold template probabilities for training rows; full-model scores for the rest."""
    import templates
    texts = [r["text"] for r in train]
    oof = np.zeros(len(train))
    dummy = np.zeros((len(train), 1))
    for tr, te in outer_splits(dummy, y, 0):
        m = fit_templates([texts[i] for i in tr], y[tr])
        for i in te:
            oof[i] = templates.score(texts[i], m)
    full = fit_templates(texts, y)
    for r, p in zip(train, oof):
        r["feat"]["template_prob"] = float(p)
    for r in extra:
        r["feat"]["template_prob"] = templates.score(r["text"], full)
    auc = roc_auc_score(y, oof)
    print(f"\n## Learned template model alone (skeleton n-grams, out-of-fold): AUC {auc:.3f} | "
          f"accuracy {((oof >= 0.5) == y).mean():.1%} | {len(full['vocab'])} templates")
    return full, auc


def verdict_thresholds(p, y, max_fp=0.02, max_fn=0.10):
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
    fl = outer_splits(X, y, 0)
    models_ = [make_gb(STYLE_FEATURES).fit(X[tr], y[tr]) for tr, _ in fl]
    for n in lengths:
        cut = [dict(r, feat={k: ai_score.analyze(first_words(r["text"], n))["raw"][k] for k in STYLE_FEATURES})
               for r in train]
        Xc = matrix(cut, STYLE_FEATURES, False)
        p = np.zeros(len(y))
        for (tr, te), m in zip(fl, models_):
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
    # Only the AI-only 'extra' genre (cover letters, LinkedIn posts, personal statements,
    # thank-you notes...) stays out of training: there is still no human data for it, so it
    # is a pure recall test, reported separately. email/report/review/post now have human
    # texts (Enron, GAO/CRS, Yelp, Reddit) and AI texts written on the same topics.
    held_out = {"extra"}
    train = [r for r in rows if r["genre"] not in held_out]
    extra = [r for r in rows if r["genre"] in held_out]
    y = np.array([r["label"] for r in train])
    global GROUPS, GENRES
    GROUPS = np.array([r["group"] for r in train])
    GENRES = np.array([r["genre"] for r in train])
    template_model, template_auc = add_template_feature(train, extra, y)
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
        # Detector benchmarks (RAID, MAGE) compare methods by TPR at a FIXED low false-positive
        # rate, because accuracy at a 50% cut-off hides how a tool behaves where it matters.
        tpr = {}
        for fp in (0.05, 0.01, 0.005):
            cut = np.quantile(p[y == 0], 1 - fp)
            tpr[str(fp)] = round(float((p[y == 1] > cut).mean()), 4)
        results[f"{key}_tpr_at_fpr"] = tpr
        print(f"  AI caught at fixed human false-positive rate: 5% FPR -> {tpr['0.05']:.0%} | "
              f"1% FPR -> {tpr['0.01']:.0%} | 0.5% FPR -> {tpr['0.005']:.0%}")
        # per-domain false positives at the 'likely AI' threshold (RAID found global thresholds
        # give 13-52% per-domain FPR for some detectors)
        thr = th["ai"] / 100
        by_genre = {}
        for r, pi in zip(train, p):
            if r["label"] == 0:
                by_genre.setdefault(r["genre"], []).append(pi >= thr)
        worst = sorted(((float(np.mean(v)), g, len(v)) for g, v in by_genre.items()), reverse=True)[:3]
        results[f"{key}_worst_domains"] = [(g, round(fp_, 3), n) for fp_, g, n in worst]
        print("  worst human domains at that threshold: " + ", ".join(f"{g} {fp_:.0%} (n={n})" for fp_, g, n in worst))
        # Recall by AI type and by genre at the verdict threshold, every one measured on a
        # genre the model did not train on (leave-one-genre-out)
        by_style, by_gen = {}, {}
        for r, pi in zip(train, p):
            if r["label"] == 1:
                by_style.setdefault(r["style"], []).append(pi >= thr)
                by_gen.setdefault(r["genre"], []).append(pi >= thr)
        results[f"{key}_recall_by_style"] = {s: round(float(np.mean(v)), 3) for s, v in by_style.items()}
        results[f"{key}_recall_by_genre"] = {g: round(float(np.mean(v)), 3) for g, v in by_gen.items()}
        print("  AI caught by type : " + " | ".join(f"{s} {np.mean(v):.0%} (n={len(v)})" for s, v in sorted(by_style.items())))
        print("  AI caught by genre: " + " | ".join(f"{g} {np.mean(v):.0%}" for g, v in sorted(by_gen.items())))
        fpg = {}
        for r, pi in zip(train, p):
            if r["label"] == 0:
                fpg.setdefault(r["genre"], []).append(pi >= thr)
        results[f"{key}_human_fp_by_genre"] = {g: round(float(np.mean(v)), 3) for g, v in fpg.items()}
        print("  human flagged by genre: " + " | ".join(f"{g} {np.mean(v):.0%}" for g, v in sorted(fpg.items())))

    if args.length_curve:
        results["length_curve"] = length_curve(train, y)

    # held-out genres, scored by models fit on all training data
    if extra:
        print("\n## Genre with NO human training data (cover letters, LinkedIn, personal statements, thank-you "
              "notes, resignation letters...) - all AI, so recall only")
        line = []
        for key, names, d in sets:
            gb = make_gb(names).fit(matrix(train, names, d), y)
            a, b = models[key]["platt"]
            z = gb.decision_function(matrix(extra, names, d))
            pe = 1 / (1 + np.exp(-(a * z + b)))
            thr_ai = models[key]["thresholds"]["ai"] / 100
            results[f"extra_{key}_caught"] = round(float((pe >= thr_ai).mean()), 4)
            results[f"extra_{key}_caught_at_50"] = round(float((pe >= 0.5).mean()), 4)
            line.append(f"{key}: {np.mean(pe >= thr_ai):.0%} at its verdict threshold ({np.mean(pe >= 0.5):.0%} at 50%)")
        print("  " + " | ".join(line))

    print("\n## Logistic-regression weights, for reading (standardized; + means pushes toward AI)")
    for n, c in sorted(zip(STYLE_FEATURES, lr_style["coef"]), key=lambda t: -abs(t[1])):
        print(f"  {n:28s} {c:+.2f}")

    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs" / "calibration_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.write:
        out = SCRIPTS / "model.json"
        models["templates"] = template_model
        models["trained_on"] = {k: results[k] for k in ("n_train", "n_ai", "n_human")}
        models["cv"] = {k: results[k] for k in results if k.endswith("_cv")}
        out.write_text(json.dumps(models, separators=(",", ":")), encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
