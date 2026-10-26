#!/usr/bin/env python3
"""Recompute ONLY the Binoculars feature for every corpus text and update
dev/runs/neural.jsonl in place (the other neural features are unchanged).

Usage: python dev/binoculars_refresh.py
Resumable: rows already refreshed are marked and skipped.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "ai-detect-humanize" / "scripts"))
import calibrate as c  # noqa: E402
import neural  # noqa: E402

CACHE = ROOT / "runs" / "neural.jsonl"
STAMP = "binoculars_v2"


def main():
    rows = c.load_corpus()
    text_by_id = {r["id"]: r["text"] for r in rows}
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        cache[d["id"]] = d["neural"]
    from ai_score import prose_blocks
    todo = [i for i in text_by_id if cache.get(i) and cache[i].get(STAMP) is None]
    print(f"{len(todo)} of {len(text_by_id)} texts to refresh", flush=True)
    t0 = time.time()
    for n, i in enumerate(todo, 1):
        prose = "\n\n".join(prose_blocks(text_by_id[i]))
        parts = neural.chunks(prose) if len(prose.split()) >= 20 else []
        vals = [v for v in (neural.binoculars_score(p) for p in parts) if v is not None]
        if vals:
            cache[i]["binoculars"] = round(sum(vals) / len(vals), 5)
            if len(vals) > 1:
                cache[i]["binoculars_max"] = round(min(vals), 5)
        cache[i][STAMP] = True
        if n % 50 == 0:
            rate = (time.time() - t0) / n
            print(f"  {n}/{len(todo)}  {rate:.1f}s per text, ~{rate * (len(todo) - n) / 60:.0f} min left", flush=True)
            CACHE.write_text("\n".join(json.dumps({"id": k, "neural": v}) for k, v in cache.items()) + "\n", encoding="utf-8")
    CACHE.write_text("\n".join(json.dumps({"id": k, "neural": v}) for k, v in cache.items()) + "\n", encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
