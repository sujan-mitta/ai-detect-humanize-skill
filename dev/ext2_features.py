#!/usr/bin/env python3
"""Cache GPT-2 + neural features for a sample of the ext2 public-dataset texts (dev/corpus/ext2).
30 texts per AI generator and per human source (arxiv dropped: lowercased/tokenised artefact).
split = 'a' (first half) / 'b' (second half): train on one, test on the other."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "plugin" / "skills" / "humanize" / "scripts"))
import external_eval as ee  # noqa: E402
N = 30
rows = []
for kind, label in (("ai", 1), ("human", 0)):
    for d in sorted((ROOT / "corpus" / "ext2" / kind).iterdir()):
        if kind == "human" and d.name.startswith("arxiv"):
            continue
        fs = [f for f in sorted(d.glob("*.txt")) if len(f.read_text(encoding="utf-8").split()) >= 100][:N]
        for i, f in enumerate(fs):
            rows.append({"text": f.read_text(encoding="utf-8"), "label": label, "genre": "x:ext2", "generator": d.name,
                         "split": "a" if i < len(fs) // 2 else "b", "source": "ext2"})
json.dump(rows, open(ROOT / "runs" / "ext2_rows.json", "w", encoding="utf-8"), ensure_ascii=False)
print(len(rows), sum(r["label"] for r in rows), flush=True)
ee.compute_features(rows, True, True)
print("done", flush=True)
