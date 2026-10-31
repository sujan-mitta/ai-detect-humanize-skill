#!/usr/bin/env python3
"""Score the deployed model on EXTERNAL benchmark texts (COLING-2025 MGT, RAID).

Usage: python dev/external_eval.py [--model PATH] [--tiers style,deep,max] [--out NAME]

Texts come from generators, domains and authors I neither wrote nor selected, so this is
the generalisation test that matters. Features (GPT-2, neural detectors) are expensive and
model-independent, so they are cached in dev/runs/external_features.jsonl and reused for
every later model.

Reports, per tier:
  AUC                    threshold-free separability on unseen generators/domains
  at verdict threshold   human false-positive rate and AI recall using the model's own
                         shipped cut-off (the realistic deployment number)
  oracle TPR@FPR 5/1%    best case if the threshold were re-tuned on this data (shows
                         how much of any gap is miscalibration vs lost signal)
  by generator / domain  where it fails
"""
import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "humanize" / "scripts"))
import ai_score  # noqa: E402

EXT = ROOT / "corpus" / "external"
CACHE = ROOT / "runs" / "external_features.jsonl"


def key(text):
    return hashlib.md5(text.encode()).hexdigest()[:12]


def load_texts():
    out = []
    for name in ("coling_eval", "raid_eval"):
        f = EXT / f"{name}.jsonl"
        if f.exists():
            out += [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    return out


def load_cache():
    cache = {}
    if CACHE.exists():
        for l in CACHE.read_text(encoding="utf-8").splitlines():
            d = json.loads(l)
            cache[d["k"]] = d
    return cache


def compute_features(texts, need_deep, need_neural):
    import neural
    import perplexity
    cache = load_cache()
    todo = [t for t in texts if key(t["text"]) not in cache
            or (need_deep and "deep" not in cache[key(t["text"])])
            or (need_neural and "neural" not in cache[key(t["text"])])]
    print(f"{len(todo)} of {len(texts)} external texts need features", flush=True)
    t0 = time.time()
    with CACHE.open("a", encoding="utf-8") as f:
        for i, t in enumerate(todo, 1):
            k = key(t["text"])
            rec = cache.get(k, {"k": k})
            if need_deep and "deep" not in rec:
                d = perplexity.features(t["text"]) or {}
                d.pop("sentence_ppl", None)
                rec["deep"] = d
            if need_neural and "neural" not in rec:
                rec["neural"] = neural.features(t["text"]) or {}
            cache[k] = rec
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 25 == 0:
                rate = (time.time() - t0) / i
                print(f"  {i}/{len(todo)} {rate:.1f}s/text ~{rate * (len(todo) - i) / 60:.0f} min left", flush=True)
    return cache


def score(text, rec, kind):
    a = ai_score.analyze(text)
    extra = None
    if kind != "style":
        extra = dict(rec.get("deep") or {})
        if kind == "max":
            extra.update(rec.get("neural") or {})
        if not extra:
            return None
    cal = ai_score.model_score(a["raw"], extra, kind)
    if not cal:
        return None
    p, _ = ai_score.apply_tell_net(cal["score"], a["raw"]["tells_per_100w"])
    return p / 100.0


def tpr_at_fpr(p, y, fpr):
    cut = np.quantile(p[y == 0], 1 - fpr)
    return float((p[y == 1] > cut).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT.parent / "humanize" / "scripts" / "model.json"))
    ap.add_argument("--tiers", default="style,deep,max")
    ap.add_argument("--out", default="external_results")
    args = ap.parse_args()
    ai_score.MODEL_PATH = Path(args.model)
    ai_score._MODELS = None
    models = ai_score.load_models()
    tiers = [t for t in args.tiers.split(",") if t in models]
    texts = load_texts()
    cache = compute_features(texts, any(t != "style" for t in tiers), "max" in tiers)
    y = np.array([t["label"] for t in texts])
    results = {}
    print(f"\n{len(texts)} external texts: {int(y.sum())} AI / {int((1 - y).sum())} human | model: {Path(args.model).name}")
    for kind in tiers:
        ps = [score(t["text"], cache.get(key(t["text"]), {}), kind) for t in texts]
        ok = np.array([p is not None for p in ps])
        p = np.array([x if x is not None else 0.0 for x in ps])
        yy, pp = y[ok], p[ok]
        thr = models[kind]["thresholds"]["ai"] / 100.0
        fp = float((pp[yy == 0] >= thr).mean())
        tp = float((pp[yy == 1] >= thr).mean())
        res = {"n": int(ok.sum()), "auc": round(float(roc_auc_score(yy, pp)), 4),
               "fpr_at_verdict": round(fp, 4), "tpr_at_verdict": round(tp, 4),
               "oracle_tpr_at_5": round(tpr_at_fpr(pp, yy, 0.05), 4), "oracle_tpr_at_1": round(tpr_at_fpr(pp, yy, 0.01), 4)}
        results[kind] = res
        print(f"\n## {kind}: AUC {res['auc']:.3f} | at the shipped verdict threshold: human flagged {fp:.0%}, AI caught {tp:.0%} "
              f"| oracle TPR @5% FPR {res['oracle_tpr_at_5']:.0%}, @1% FPR {res['oracle_tpr_at_1']:.0%}")
        for field in ("source", "genre"):
            by = collections.defaultdict(lambda: [[], []])
            for t, pi, o in zip(texts, p, ok):
                if o:
                    by[t[field]][t["label"]].append(pi >= thr)
            line = []
            for g, (h, a) in sorted(by.items(), key=lambda kv: -len(kv[1][0]) - len(kv[1][1]))[:8]:
                line.append(f"{g}: FP {np.mean(h):.0%}(n={len(h)}) TP {np.mean(a):.0%}(n={len(a)})" if h and a else
                            f"{g}: " + (f"FP {np.mean(h):.0%}(n={len(h)})" if h else f"TP {np.mean(a):.0%}(n={len(a)})"))
            print(f"   by {field}: " + " | ".join(line))
        gen = collections.defaultdict(list)
        for t, pi, o in zip(texts, p, ok):
            if o and t["label"] == 1:
                gen[t["generator"]].append(pi >= thr)
        worst = sorted(((np.mean(v), g, len(v)) for g, v in gen.items() if len(v) >= 3))[:6]
        print("   hardest generators (recall): " + ", ".join(f"{g} {r:.0%}(n={n})" for r, g, n in worst))
        res["recall_by_generator"] = {g: round(float(np.mean(v)), 3) for g, v in gen.items()}
    (ROOT / "runs" / f"{args.out}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
