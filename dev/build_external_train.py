#!/usr/bin/env python3
"""Diverse training texts from the COLING-2025 MGT-en TRAIN split (disjoint from the
dev-split external eval), balanced across generators and sources.

Usage: python dev/build_external_train.py [--n 3000]

Why: our own AI texts come from a handful of Claude-family prompts. A detector trained only
on those learns "Claude-ness". These rows add dozens of other generators and many human
sources. Used for the cheap tiers (style + learned templates); GPT-2/neural features are
computed only for a subset. Writes dev/corpus/external/coling_train.jsonl.
"""
import argparse
import collections
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "corpus" / "external" / "coling_train.jsonl"
random.seed(52)
API = "https://datasets-server.huggingface.co/rows?dataset=Jinyan1/COLING_2025_MGT_en&config=default&split=train"


def fetch(offset):
    for attempt in range(5):
        p = subprocess.run(["curl", "-sS", "--max-time", "90", f"{API}&offset={offset}&length=100"], capture_output=True)
        try:
            return [r["row"] for r in json.loads(p.stdout)["rows"]]
        except Exception:
            time.sleep(2 + 3 * attempt)
    return []


def words(t):
    return len(t.split())


def clip(text, lo=110, hi=450):
    text = re.sub(r"[ \t]+", " ", text.strip())
    if words(text) < lo:
        return None
    if words(text) > hi:
        cut = " ".join(text.split()[:hi])
        end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
        text = cut[: end + 1] if end > len(cut) * 0.6 else cut
    return text if words(text) >= lo else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()
    # find the size of the split
    lo, hi = 0, 1 << 22
    while hi - lo > 100:
        mid = (lo + hi) // 2
        if fetch(mid):
            lo = mid
        else:
            hi = mid
    total = lo
    print("train rows ~", total, flush=True)
    offsets = random.sample(range(0, max(1, total - 100), 100), min(total // 100 - 1, 90))
    rows = []
    for k, off in enumerate(offsets):
        rows += fetch(off)
        if k % 20 == 0:
            print(f"  {k}/{len(offsets)} requests, {len(rows)} rows", flush=True)
    seen, cand = set(), []
    for r in rows:
        t = clip(r["text"])
        if not t:
            continue
        h = hashlib.md5(t.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        cand.append({"text": t, "label": int(r["label"]), "genre": f"{r['source']}:{r.get('sub_source') or ''}",
                     "generator": r["model"], "source": "coling_train", "group": "coling_train:" + str(r["id"])})
    print(f"usable after clipping/dedup: {len(cand)} (AI {sum(c['label'] for c in cand)})")
    # balance: equal human and AI, AI spread over generators, humans over sources
    hum = [c for c in cand if c["label"] == 0]
    ai = [c for c in cand if c["label"] == 1]
    k = min(args.n // 2, len(hum), len(ai))
    by_gen = collections.defaultdict(list)
    for c in ai:
        by_gen[c["generator"]].append(c)
    for v in by_gen.values():
        random.shuffle(v)
    pick_ai = []
    while len(pick_ai) < k and any(by_gen.values()):
        for g in sorted(by_gen):
            if by_gen[g] and len(pick_ai) < k:
                pick_ai.append(by_gen[g].pop())
    random.shuffle(hum)
    out = hum[:len(pick_ai)] + pick_ai
    OUT.write_text("\n".join(json.dumps(r) for r in out) + "\n", encoding="utf-8")
    print(f"wrote {len(out)} ({len(pick_ai)} AI) | generators: {len({c['generator'] for c in pick_ai})} | "
          f"sources: {dict(collections.Counter(c['genre'].split(':')[0] for c in out))}")


if __name__ == "__main__":
    main()
