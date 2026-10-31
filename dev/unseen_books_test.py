#!/usr/bin/env python3
"""False-positive check on human writing from authors the model never saw.

Usage: python dev/unseen_books_test.py [--model PATH] [--compare PATH]

Cross-validation can flatter a model when several passages from one author sit
in the training data (it learns the author, not "human"). This test scores 40
passages from four public-domain books that are NOT in the corpus (Twain,
Carroll, Austen, Jerome's Three Men in a Boat) and reports how many each mode
wrongly calls "likely AI". Passages and their neural features are cached in
dev/runs/unseen_books.json, so re-runs take about a minute.
"""
import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "skills" / "humanize" / "scripts"))
import ai_score  # noqa: E402

CACHE = ROOT / "runs" / "unseen_books.json"
BOOKS = {74: "Tom Sawyer", 11: "Alice in Wonderland", 158: "Emma", 308: "Three Men in a Boat"}


def build_cache():
    import neural
    import perplexity
    random.seed(5)
    items = []
    for bid, name in BOOKS.items():
        raw = subprocess.run(["curl", "-sSL", "--retry", "3", f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt"],
                             capture_output=True).stdout.decode("utf-8", "replace").replace("\r\n", "\n")
        body = raw.split("*** START", 1)[-1].split("*** END", 1)[0]
        body = re.sub(r"(?<!\n)\n(?!\n)", " ", body)
        paras = [p.strip() for p in re.split(r"\n\s*\n", body) if len(p.split()) > 30]
        for s in random.sample(range(20, len(paras) - 10), 10):
            chunk, n = [], 0
            for p in paras[s:]:
                chunk.append(p)
                n += len(p.split())
                if n >= 250:
                    break
            text = "\n\n".join(chunk)
            extra = perplexity.features(text) or {}
            extra.pop("sentence_ppl", None)
            extra.update(neural.features(text) or {})
            items.append({"book": name, "text": text, "extra": extra})
    CACHE.write_text(json.dumps(items), encoding="utf-8")
    return items


def evaluate(items, model_path):
    ai_score.MODEL_PATH = Path(model_path)
    ai_score._MODELS = None
    out = {}
    for kind in ("style", "deep", "max"):
        if kind not in ai_score.load_models():
            continue
        flagged = []
        for it in items:
            a = ai_score.analyze(it["text"])
            cal = ai_score.model_score(a["raw"], it["extra"] if kind != "style" else None, kind)
            p, _ = ai_score.apply_tell_net(cal["score"], a["raw"]["tells_per_100w"])
            flagged.append(ai_score.display(p, kind) >= ai_score.SCORE_AI)
        out[kind] = float(np.mean(flagged))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT.parent / "skills" / "humanize" / "scripts" / "model.json"))
    ap.add_argument("--compare")
    args = ap.parse_args()
    items = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else build_cache()
    for label, path in (("compare", args.compare), ("model", args.model)):
        if path:
            res = evaluate(items, path)
            print(f"{label:8s} " + " | ".join(f"{k}: {v:.0%} wrongly flagged" for k, v in res.items())
                  + f"  (n={len(items)} unseen human passages)")


if __name__ == "__main__":
    main()
