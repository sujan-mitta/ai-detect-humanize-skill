#!/usr/bin/env python3
"""External benchmark texts that I did not write or curate - the generalisation test.

Usage: python dev/build_external.py

RAID (Dugan et al. 2024, liamdugan/raid): human text + generations from 11 models
  (gpt2, gpt3, chatgpt, gpt4, mistral, mistral-chat, mpt, mpt-chat, llama-chat,
  cohere, cohere-chat) across 8 domains (abstracts, books, news, poetry, recipes,
  reddit, reviews, wiki). Only the un-attacked ('attack' == 'none') texts are used here.
COLING-2025 MGT-en (Jinyan1/COLING_2025_MGT_en): M4GT-Bench and other sources, many
  generators.

Writes dev/corpus/external/{raid_eval,raid_pool,coling_eval}.jsonl with
  text, label (1 = AI), genre (domain), generator, source, group
Eval and pool sets share no RAID `source_id`, so the same human text (or generations
prompted from it) never appears on both sides.
"""
import collections
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "corpus" / "external"
random.seed(31)
API = "https://datasets-server.huggingface.co/rows"


def fetch(dataset, config, split, offset, length=100):
    url = f"{API}?dataset={dataset}&config={config}&split={split}&offset={offset}&length={length}"
    for attempt in range(4):
        p = subprocess.run(["curl", "-sS", "--max-time", "90", url], capture_output=True)
        try:
            return [r["row"] for r in json.loads(p.stdout)["rows"]]
        except Exception:
            time.sleep(2 + attempt * 3)
    return []


def n_words(t):
    return len(t.split())


def clip(text, lo=110, hi=450):
    """First <= hi words ending on a sentence boundary; None if shorter than lo."""
    text = re.sub(r"[ \t]+", " ", text.strip())
    if n_words(text) < lo:
        return None
    if n_words(text) > hi:
        cut = " ".join(text.split()[:hi]) if "\n" not in text else text[: len(" ".join(text.split()[:hi])) + 40]
        cut = " ".join(text.split()[:hi])
        end = max(cut.rfind(". "), cut.rfind(".\n"), cut.rfind("? "), cut.rfind("! "))
        text = cut[: end + 1] if end > len(cut) * 0.6 else cut
    return text if n_words(text) >= lo else None


def raid(n_requests=260):
    total = 5_627_439  # RAID train rows (attack variants included); we only keep attack == none
    offsets = random.sample(range(0, total - 100, 100), n_requests)
    rows = []
    for k, off in enumerate(offsets):
        rows += fetch("liamdugan/raid", "raid", "train", off)
        if k % 40 == 0:
            print(f"  raid {k}/{n_requests} requests, {len(rows)} rows", flush=True)
    rows = [r for r in rows if r.get("attack") == "none" and r.get("generation")]
    print(f"  raid rows without attack: {len(rows)}")
    return rows


def build_raid(rows):
    by_source = collections.defaultdict(list)
    for r in rows:
        t = clip(r["generation"])
        if t:
            by_source[r["source_id"]].append({"text": t, "label": 0 if r["model"] == "human" else 1,
                                              "genre": r["domain"], "generator": r["model"],
                                              "source": "raid", "group": "raid:" + r["source_id"],
                                              "decoding": r.get("decoding"), "rep_pen": r.get("repetition_penalty")})
    ids = sorted(by_source)
    random.shuffle(ids)
    half = len(ids) // 2
    evals, pool = [], []
    for part, dest in ((ids[:half], evals), (ids[half:], pool)):
        for sid in part:
            dest += by_source[sid]
    # balanced evaluation set: per domain, equal human and AI, AI spread over generators
    final_eval = []
    for dom in sorted({r["genre"] for r in evals}):
        hum = [r for r in evals if r["genre"] == dom and r["label"] == 0]
        ai = [r for r in evals if r["genre"] == dom and r["label"] == 1]
        random.shuffle(hum), random.shuffle(ai)
        # round-robin over generators so one model doesn't dominate
        buckets = collections.defaultdict(list)
        for r in ai:
            buckets[r["generator"]].append(r)
        picked = []
        while len(picked) < 22 and any(buckets.values()):
            for g in sorted(buckets):
                if buckets[g] and len(picked) < 22:
                    picked.append(buckets[g].pop())
        final_eval += hum[:22] + picked
    return final_eval, pool


def coling(n_requests=60):
    rows = []
    for k in range(n_requests):
        rows += fetch("Jinyan1/COLING_2025_MGT_en", "default", "dev", k * 100)
    out = []
    for r in rows:
        t = clip(r["text"])
        if t:
            out.append({"text": t, "label": int(r["label"]), "genre": r.get("sub_source") or r["source"],
                        "generator": r["model"], "source": "coling", "group": "coling:" + str(r["id"])})
    by = collections.defaultdict(list)
    for r in out:
        by[(r["label"], r["generator"])].append(r)
    picked = []
    for key in sorted(by):
        random.shuffle(by[key])
    # about 110 human and 110 AI, AI spread over generators
    hum = [r for k, v in by.items() if k[0] == 0 for r in v]
    random.shuffle(hum)
    picked += hum[:110]
    ai_keys = [k for k in by if k[0] == 1]
    while sum(1 for r in picked if r["label"] == 1) < 110 and any(by[k] for k in ai_keys):
        for k in ai_keys:
            if by[k] and sum(1 for r in picked if r["label"] == 1) < 110:
                picked.append(by[k].pop())
    return picked


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("RAID...")
    ev, pool = build_raid(raid())
    (OUT / "raid_eval.jsonl").write_text("\n".join(json.dumps(r) for r in ev) + "\n", encoding="utf-8")
    (OUT / "raid_pool.jsonl").write_text("\n".join(json.dumps(r) for r in pool) + "\n", encoding="utf-8")
    print(f"raid_eval {len(ev)} (AI {sum(r['label'] for r in ev)}), raid_pool {len(pool)} (AI {sum(r['label'] for r in pool)})")
    print("  eval generators:", dict(collections.Counter(r["generator"] for r in ev if r["label"])))
    print("COLING...")
    co = coling()
    (OUT / "coling_eval.jsonl").write_text("\n".join(json.dumps(r) for r in co) + "\n", encoding="utf-8")
    print(f"coling_eval {len(co)} (AI {sum(r['label'] for r in co)})")
    print("  generators:", dict(collections.Counter(r["generator"] for r in co if r["label"])))


if __name__ == "__main__":
    main()
