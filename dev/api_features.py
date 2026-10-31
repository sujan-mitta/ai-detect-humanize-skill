#!/usr/bin/env python3
"""GPT-2 + neural features for the API-generated texts (dev/corpus/api/<model>/<style>__<task>.txt).
Rows: generator = model, split = prompt style, genre = task. Re-run to pick up new files (cached)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "humanize" / "scripts"))
import external_eval as ee  # noqa: E402
rows = []
for f in sorted((ROOT / "corpus").glob("api*/*/*.txt")):
    t = f.read_text(encoding="utf-8")
    if len(t.split()) < 100:
        continue
    style, task = f.stem.split("__")
    rows.append({"text": t, "label": 1, "genre": "x:" + task, "generator": f.parent.name, "split": style, "source": f.parent.parent.name})
(ROOT / "runs" / "api_rows.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
print(len(rows), flush=True)
ee.compute_features(rows, True, True)
print("done", flush=True)
