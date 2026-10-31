#!/usr/bin/env python3
"""Extra legacy-head training rows aimed at the weakest modern families (Llama-3, Gemma).
Only COLING-train rows with no cached features yet, so the family_eval test texts stay held out."""
import json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "humanize" / "scripts"))
import external_eval as ee  # noqa: E402
random.seed(17)
cache = ee.load_cache()
rows = [json.loads(l) for l in (ee.EXT / "coling_train.jsonl").read_text(encoding="utf-8").splitlines()]
rows = [r for r in rows if 120 <= len(r["text"].split()) <= 700 and ee.key(r["text"]) not in cache]
pick = []
for gens, n in ((("llama3-70b", "llama3-8b"), 60), (("gemma2-9b-it", "gemma-7b-it"), 60)):
    pick += random.sample([r for r in rows if r["label"] == 1 and r["generator"] in gens], n)
pick += random.sample([r for r in rows if r["label"] == 0], 60)
json.dump(pick, open(ROOT / "runs" / "legacy_subset_extra.json", "w", encoding="utf-8"), ensure_ascii=False)
print(len(pick), flush=True)
ee.compute_features(pick, True, True)
