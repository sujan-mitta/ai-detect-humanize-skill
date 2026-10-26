#!/usr/bin/env python3
"""Tier-by-tier calibration with leave-one-genre-out evaluation and diverse external data.

Usage: python dev/calibrate2.py [--external] [--write] [--skip-max] [--tag NAME]

  style  all training rows: our corpus (+ with --external: 3,000 COLING-train texts from 39
         generators / 3 benchmark sources)
  deep   rows that have GPT-2 features (our corpus)
  max    rows that also have the neural-detector features

Evaluation = LEAVE ONE GENRE OUT. Genres include the external benchmark sources (x:hc3,
x:mage, x:m4gt), so a whole benchmark - dozens of generators and human sources - is held out
while the model trains on the rest. Out-of-fold scores give the verdict thresholds (at most
2% of held-out human texts reach "likely AI"), fixed-FPR recall, and recall by AI type.
"""
import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import calibrate as c  # noqa: E402
import templates  # noqa: E402

SCRIPTS = c.SCRIPTS
EXT = ROOT / "corpus" / "external" / "coling_train.jsonl"


def load_rows(external):
    rows = c.load_corpus()
    if external and EXT.exists():
        for l in EXT.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            src = r["genre"].split(":")[0]
            rows.append({"text": r["text"], "label": r["label"], "genre": f"x:{src}", "style": "external",
                         "source": f"ext:{src}:{r['generator']}", "generator": r["generator"],
                         "id": "ext/" + hashlib.md5(r["text"].encode()).hexdigest()[:12],
                         "group": r["group"], "want_deep": False})
    return rows


def fit_from_counts(counts, y, min_df=5, max_feats=30000, C=2.0):
    from scipy.sparse import csr_matrix
    from sklearn.linear_model import LogisticRegression
    df = Counter(g for cn in counts for g in cn)
    vocab = [g for g, d in df.most_common(max_feats) if d >= min_df]
    index = {g: i for i, g in enumerate(vocab)}
    n = len(counts)
    idf = [math.log((1 + n) / (1 + df[g])) + 1 for g in vocab]
    ri, ci, vv = [], [], []
    for i, cn in enumerate(counts):
        for j, x in templates.vectorize(cn, index, idf).items():
            ri.append(i); ci.append(j); vv.append(x)
    X = csr_matrix((vv, (ri, ci)), shape=(n, len(vocab)))
    lr = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(X, y)
    return {"vocab": vocab, "idf": [round(v, 5) for v in idf], "coef": [round(float(v), 5) for v in lr.coef_[0]],
            "intercept": round(float(lr.intercept_[0]), 5)}


def template_features(train, extra):
    """Out-of-fold (leave-one-genre-out) template probability for every training row; the
    full model scores the held-out 'extra' genre. N-gram counts are computed once."""
    y = np.array([r["label"] for r in train])
    counts = [templates.ngram_counts(r["text"]) for r in train]
    oof = np.zeros(len(train))
    for tr, te in c.outer_splits(np.zeros((len(train), 1)), y, 0):
        m = fit_from_counts([counts[i] for i in tr], y[tr])
        m["_vocab_index"] = {g: i for i, g in enumerate(m["vocab"])}
        for i in te:
            oof[i] = templates.score_counts(counts[i], m)
    full = fit_from_counts(counts, y)
    for r, p in zip(train, oof):
        r["feat"]["template_prob"] = float(p)
    full["_vocab_index"] = {g: i for i, g in enumerate(full["vocab"])}
    for r in extra:
        r["feat"]["template_prob"] = templates.score_counts(templates.ngram_counts(r["text"]), full)
    del full["_vocab_index"]
    auc = roc_auc_score(y, oof)
    print(f"\n## Learned template model alone, leave-one-genre-out: AUC {auc:.3f} | "
          f"accuracy {((oof >= 0.5) == y).mean():.1%} | {len(full['vocab'])} templates", flush=True)
    return full, auc


def tier_rows(train, kind):
    if kind == "style":
        return train
    need_neural = kind == "max"
    return [r for r in train if r.get("deep") and (not need_neural or r.get("neural"))]


def report_tier(key, sub, y, p, results, models):
    th = c.verdict_thresholds(p, y)
    models[key]["thresholds"] = th
    auc = float(roc_auc_score(y, p))
    acc = float(((p >= 0.5) == y).mean())
    results[f"{key}_cv"] = {"accuracy": round(acc, 4), "auc": round(auc, 4), "n": len(y), "n_ai": int(y.sum())}
    thr = th["ai"] / 100
    print(f"\n## {key} tier ({len(y)} texts, {int(y.sum())} AI) - leave-one-genre-out")
    print(f"  AUC {auc:.3f} | accuracy at 50% {acc:.1%}")
    print(f"  verdict bands: likely AI >= {th['ai']:.0f} (at most 2% of held-out humans), likely human < {th['human']:.0f}")
    print(f"  humans: {th['human_said_human']:.0%} 'likely human', {th['human_said_ai']:.0%} 'likely AI' | "
          f"AI: {th['ai_said_ai']:.0%} 'likely AI', {th['ai_said_human']:.0%} 'likely human'")
    tpr = {}
    for fp in (0.05, 0.01, 0.005):
        cut = np.quantile(p[y == 0], 1 - fp)
        tpr[str(fp)] = round(float((p[y == 1] > cut).mean()), 4)
    results[f"{key}_tpr_at_fpr"] = tpr
    print(f"  AI caught at fixed human FPR: 5% -> {tpr['0.05']:.0%} | 1% -> {tpr['0.01']:.0%} | 0.5% -> {tpr['0.005']:.0%}")
    by = {"human_fp": {}, "ai_recall": {}, "ai_type": {}}
    for r, pi in zip(sub, p):
        if r["label"] == 0:
            by["human_fp"].setdefault(r["genre"], []).append(pi >= thr)
        else:
            by["ai_recall"].setdefault(r["genre"], []).append(pi >= thr)
            by["ai_type"].setdefault(r["style"], []).append(pi >= thr)
    for k, v in by.items():
        results[f"{key}_{k}"] = {g: round(float(np.mean(x)), 3) for g, x in v.items()}
    print("  human flagged by genre: " + " | ".join(f"{g} {np.mean(v):.0%}" for g, v in sorted(by["human_fp"].items())))
    print("  AI caught by genre    : " + " | ".join(f"{g} {np.mean(v):.0%}" for g, v in sorted(by["ai_recall"].items())))
    print("  AI caught by type     : " + " | ".join(f"{s} {np.mean(v):.0%}(n={len(v)})" for s, v in sorted(by["ai_type"].items())))
    return th


PROTOCOL = "leave-one-genre-out"


def main():
    global PROTOCOL
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--skip-max", action="store_true")
    ap.add_argument("--style-only", action="store_true", help="style tier only; no GPT-2/neural (cheap, safe to run beside other jobs)")
    ap.add_argument("--tag", default="calibration2")
    ap.add_argument("--protocol", choices=["logo", "grouped"], default="logo",
                    help="logo: hold out whole genres (pessimistic: a genre the model never saw). "
                         "grouped: hold out whole sources/authors but keep every genre in training "
                         "(what a deployed model faces on a genre it has seen).")
    args = ap.parse_args()
    PROTOCOL = "leave-one-genre-out" if args.protocol == "logo" else "grouped by source/author (genres seen)"

    rows = c.extract_features(load_rows(args.external), not args.style_only)
    train = [r for r in rows if r["genre"] != "extra"]
    extra = [r for r in rows if r["genre"] == "extra"]
    c.GROUPS = np.array([r["group"] for r in train])
    c.GENRES = np.array([r["genre"] for r in train]) if args.protocol == "logo" else None
    y_all = np.array([r["label"] for r in train])
    print(f"{len(train)} training rows ({int(y_all.sum())} AI) over {len(set(r['genre'] for r in train))} genres; {len(extra)} AI-only 'extra' rows held out")
    template_model, _ = template_features(train, extra)

    results = {"n_train": len(train), "n_ai": int(y_all.sum()), "n_human": int((1 - y_all).sum()), "external": args.external}
    models, tiers = {}, [("style", c.STYLE_FEATURES)]
    if not args.style_only:
        tiers.append(("deep", c.STYLE_FEATURES + c.DEEP_FEATURES))
    if not args.style_only and not args.skip_max:
        tiers.append(("max", c.STYLE_FEATURES + c.DEEP_FEATURES + c.NEURAL_FEATURES))
    fitted = {}
    for key, names in tiers:
        sub = tier_rows(train, key)
        if len(sub) < 0.5 * len(train) and key != "style" and len(sub) < 400:
            print(f"\n{key}: only {len(sub)} rows have the features; skipped")
            continue
        c.GROUPS = np.array([r["group"] for r in sub])
        c.GENRES = np.array([r["genre"] for r in sub]) if args.protocol == "logo" else None
        y = np.array([r["label"] for r in sub])
        X = c.matrix(sub, names)
        p = c.cv_predict_gb(X, y, names)
        models[key] = c.fit_gb(X, y, names)
        report_tier(key, sub, y, p, results, models)
        fitted[key] = (names, sub, y)

    if extra and fitted:
        print("\n## AI-only 'extra' genre (cover letters, LinkedIn, personal statements, thank-you notes...): recall only")
        for key, (names, sub, y) in fitted.items():
            gb = c.make_gb(names).fit(c.matrix(sub, names), y)
            a, b = models[key]["platt"]
            ex = [r for r in extra if (key == "style" or r.get("deep")) and (key != "max" or r.get("neural"))]
            if not ex:
                continue
            pe = 1 / (1 + np.exp(-(a * gb.decision_function(c.matrix(ex, names)) + b)))
            thr = models[key]["thresholds"]["ai"] / 100
            results[f"extra_{key}_caught"] = round(float((pe >= thr).mean()), 4)
            print(f"  {key}: {np.mean(pe >= thr):.0%} caught at the verdict threshold (n={len(ex)})")

    (ROOT / "runs" / f"{args.tag}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.write:
        models["templates"] = template_model
        models["trained_on"] = {k: results[k] for k in ("n_train", "n_ai", "n_human")}
        models["cv"] = {k: v for k, v in results.items() if k.endswith("_cv")}
        out = SCRIPTS / "model.json"
        out.write_text(json.dumps(models, separators=(",", ":")), encoding="utf-8")
        print(f"\nwrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
