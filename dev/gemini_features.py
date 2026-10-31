#!/usr/bin/env python3
"""GPT-2 + neural features for the Gemini texts (cached in runs/external_features.jsonl)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "skills" / "humanize" / "scripts"))
import external_eval as ee  # noqa: E402
rows = []
for f in sorted((ROOT / "corpus" / "gemini").glob("*/*.txt")):
    n = int(f.stem.split("__")[1])
    rows.append({"text": f.read_text(encoding="utf-8"), "label": 1, "genre": "x:" + f.stem.split("__")[0],
                 "generator": f.parent.name, "split": "train" if n <= 3 else "test", "source": "gemini"})
json.dump(rows, open(ROOT / "runs" / "gemini_rows.json", "w", encoding="utf-8"), ensure_ascii=False)
print(len(rows), flush=True)
ee.compute_features(rows, True, True)
