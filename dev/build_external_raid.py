#!/usr/bin/env python3
"""RAID external test/pool sets, sampled from the right blocks of the file.

RAID's train split is ordered: for each domain, [human texts][un-attacked AI texts][attacked
variants...]. Random offsets therefore almost always hit attacked AI rows (the first attempt
produced zero human texts). This script finds the domain and block boundaries by binary
search, then samples inside the human and un-attacked-AI blocks of every domain.

Writes dev/corpus/external/raid_eval.jsonl and raid_pool.jsonl (label 1 = AI). The split is
by a hash of `source_id`, so a human text and generations prompted from it never straddle
eval and pool.
"""
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
OUT = ROOT / "corpus" / "external"
random.seed(41)
API = "https://datasets-server.huggingface.co/rows?dataset=liamdugan/raid&config=raid&split=train"
_cache = {}


def rows(off, n=100):
    key = (off, n)
    if key in _cache:
        return _cache[key]
    for attempt in range(5):
        p = subprocess.run(["curl", "-sS", "--max-time", "90", f"{API}&offset={off}&length={n}"], capture_output=True)
        try:
            rs = [r["row"] for r in json.loads(p.stdout)["rows"]]
            _cache[key] = rs
            return rs
        except Exception:
            time.sleep(2 + 3 * attempt)
    _cache[key] = []
    return []


def first(off):
    rs = rows(off, 1)
    return rs[0] if rs else None


def total_rows():
    lo, hi = 0, 1 << 23   # find first empty offset
    while hi - lo > 100:
        mid = (lo + hi) // 2
        if first(mid):
            lo = mid
        else:
            hi = mid
    return lo


def boundary(lo, hi, pred):
    """Smallest offset in (lo, hi] where pred(row) turns True, assuming pred(lo) is False
    and pred is monotone over the range."""
    while hi - lo > 100:
        mid = (lo + hi) // 2
        r = first(mid)
        if r is not None and pred(r):
            hi = mid
        else:
            lo = mid
    return hi


def words(t):
    return len(t.split())


def clip(text, lo=110, hi=450):
    text = re.sub(r"[ \t]+", " ", text.strip())
    if words(text) < lo:
        return None
    if words(text) > hi:
        cut = " ".join(text.split()[:hi])
        end = max(cut.rfind(". "), cut.rfind(".\n"), cut.rfind("? "), cut.rfind("! "))
        text = cut[: end + 1] if end > len(cut) * 0.6 else cut
    return text if words(text) >= lo else None


def main():
    n = total_rows()
    print("total rows ~", n, flush=True)
    # domain starts: scan coarsely, then binary-search each change
    coarse = list(range(0, n, 100_000)) + [n - 1]
    domains, starts = [], []
    prev = None
    for off in coarse:
        r = first(off)
        if r is None:
            continue
        if r["domain"] != prev:
            if prev is None:
                starts.append(0)
            else:
                lo = coarse[coarse.index(off) - 1]
                starts.append(boundary(lo, off, lambda x, d=r["domain"]: x["domain"] == d))
            domains.append(r["domain"])
            prev = r["domain"]
    print("domain starts:", dict(zip(domains, starts)), flush=True)
    ends = starts[1:] + [n]
    plan = {}
    for dom, s, e in zip(domains, starts, ends):
        h_end = boundary(s, e, lambda x: x["model"] != "human") if first(s)["model"] == "human" else s
        a_end = boundary(h_end, e, lambda x: x["attack"] != "none")
        plan[dom] = {"human": (s, h_end), "ai": (h_end, a_end)}
        print(f"  {dom:10s} human {s}-{h_end} ({h_end - s}) | AI un-attacked {h_end}-{a_end} ({a_end - h_end})", flush=True)

    texts = []
    for dom, p in plan.items():
        for kind, (lo, hi) in p.items():
            span = max(1, hi - lo - 100)
            want = 14 if kind == "human" else 18   # requests of 100 rows each
            for off in random.sample(range(lo, lo + span, 100), min(want, max(1, span // 100))):
                for r in rows(off):
                    if r.get("attack") != "none" or not r.get("generation"):
                        continue
                    t = clip(r["generation"])
                    if t:
                        texts.append({"text": t, "label": 0 if r["model"] == "human" else 1, "genre": r["domain"],
                                      "generator": r["model"], "source": "raid", "group": "raid:" + r["source_id"],
                                      "decoding": r.get("decoding"), "rep_pen": r.get("repetition_penalty")})
        print(f"  fetched {dom}: {len(texts)} total so far", flush=True)

    def part(r):
        return int(hashlib.md5(r["group"].encode()).hexdigest(), 16) % 2

    ev_raw = [r for r in texts if part(r) == 0]
    pool = [r for r in texts if part(r) == 1]
    ev = []
    for dom in domains:
        hum = [r for r in ev_raw if r["genre"] == dom and r["label"] == 0]
        ai = [r for r in ev_raw if r["genre"] == dom and r["label"] == 1]
        random.shuffle(hum)
        buckets = collections.defaultdict(list)
        for r in ai:
            buckets[r["generator"]].append(r)
        for b in buckets.values():
            random.shuffle(b)
        picked = []
        while len(picked) < 24 and any(buckets.values()):
            for g in sorted(buckets):
                if buckets[g] and len(picked) < 24:
                    picked.append(buckets[g].pop())
        k = min(24, len(hum), len(picked)) if hum and picked else 0
        ev += hum[:k] + picked[:k]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raid_eval.jsonl").write_text("\n".join(json.dumps(r) for r in ev) + "\n", encoding="utf-8")
    (OUT / "raid_pool.jsonl").write_text("\n".join(json.dumps(r) for r in pool) + "\n", encoding="utf-8")
    print(f"raid_eval {len(ev)} (AI {sum(r['label'] for r in ev)}) | raid_pool {len(pool)} (AI {sum(r['label'] for r in pool)})")
    print("eval generators:", dict(collections.Counter(r["generator"] for r in ev if r["label"])))
    print("eval domains:", dict(collections.Counter(r["genre"] for r in ev)))


if __name__ == "__main__":
    main()
