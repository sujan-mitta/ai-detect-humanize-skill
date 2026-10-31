#!/usr/bin/env python3
"""GPT-2 + neural features for the round-2 Gemini texts (dev/corpus/gemini2), cached like round 1.
split = the prompt condition (casual / persona) so the legacy head can be trained on one condition
and tested on the other."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "humanize" / "scripts"))
import external_eval as ee  # noqa: E402
rows = []
for f in sorted((ROOT / "corpus" / "gemini2").glob("*/*.txt")):
    t = f.read_text(encoding="utf-8")
    if len(t.split()) < 100:
        continue
    model, cond = f.parent.name.split("__")
    rows.append({"text": t, "label": 1, "genre": "x:" + f.stem.split("__")[0], "generator": f.parent.name,
                 "split": cond, "source": "gemini2"})
json.dump(rows, open(ROOT / "runs" / "gemini2_rows.json", "w", encoding="utf-8"), ensure_ascii=False)
print(len(rows), flush=True)
ee.compute_features(rows, True, True)
