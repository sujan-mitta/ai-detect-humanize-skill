#!/usr/bin/env python3
"""Compute GPT-2 + neural features for a balanced, generator-diverse subset of the
COLING-2025 *train* split (disjoint from the eval texts used to test), cached in
dev/runs/external_features.jsonl. Used only by the predictability-only "legacy" head."""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "humanize" / "scripts"))
import external_eval as ee  # noqa: E402

N_AI, N_HUMAN = 150, 150
random.seed(5)


def subset():
    rows = [json.loads(l) for l in (ee.EXT / "coling_train.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if 120 <= len(r["text"].split()) <= 700]
    ev = {ee.key(t["text"]) for t in ee.load_texts()}
    rows = [r for r in rows if ee.key(r["text"]) not in ev]
    by = defaultdict(list)
    for r in rows:
        by[(r["label"], r["generator"] if r["label"] else r["genre"])].append(r)
    out = []
    for label, n in ((1, N_AI), (0, N_HUMAN)):
        keys = [k for k in by if k[0] == label]
        pools = [random.sample(by[k], len(by[k])) for k in keys]
        pick = []
        while len(pick) < n and any(pools):
            for p in pools:
                if p and len(pick) < n:
                    pick.append(p.pop())
        out += pick
    return out


if __name__ == "__main__":
    s = subset()
    json.dump(s, open(ROOT / "runs" / "legacy_subset.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(len(s), sum(r["label"] for r in s), "AI", flush=True)
    ee.compute_features(s, True, True)
