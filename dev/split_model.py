#!/usr/bin/env python3
"""Split a full single-file model into the shipped layout: a small manifest (model.json) plus component files in model/
(each under 1 MiB, so the plugin directory can read them as text). ai_score.load_models() resolves the "@model/x.json" references.
Usage: python dev/split_model.py FULL_MODEL.json [OUT_DIR]   (default OUT_DIR = plugin/skills/humanize/scripts)"""
import json, sys
from pathlib import Path
full = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / "plugin" / "skills" / "humanize" / "scripts"
(out / "model").mkdir(parents=True, exist_ok=True)
manifest = {}
for k, v in full.items():
    size = len(json.dumps(v, separators=(",", ":")))
    if size > 50_000:
        assert size < 1_000_000, f"{k} is {size} bytes; split it further"
        (out / "model" / f"{k}.json").write_text(json.dumps(v, separators=(",", ":")), encoding="utf-8")
        manifest[k] = f"@model/{k}.json"
    else:
        manifest[k] = v
(out / "model.json").write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
print("manifest", (out / "model.json").stat().st_size, "bytes;", ", ".join(f"{p.name} {p.stat().st_size/1048576:.2f} MiB" for p in sorted((out / "model").glob("*.json"))))
