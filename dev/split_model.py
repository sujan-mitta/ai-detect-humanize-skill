#!/usr/bin/env python3
"""Split a full single-file model into the shipped layout: a small manifest (model.json) plus component files in model/.
Anthropic's plugin directory reads only text files under 256 KiB, so every file written here stays under ~150 KB: a component
with long lists (trees, template vocabulary...) becomes a small file plus shard files with slices of those lists.
ai_score.load_models() reassembles them. Usage: python dev/split_model.py FULL_MODEL.json [OUT_DIR]
(default OUT_DIR = plugin/skills/humanize/scripts)"""
import json, shutil, sys
from pathlib import Path
LIMIT = 150_000
dump = lambda v: json.dumps(v, separators=(",", ":"))
full = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / "plugin" / "skills" / "humanize" / "scripts"
shutil.rmtree(out / "model", ignore_errors=True)
(out / "model").mkdir(parents=True)

def shards(name, field, items):
    files, cur, size = [], [], 2
    def flush():
        nonlocal cur, size
        if cur:
            rel = f"model/{name}.{field}.{len(files)}.json"
            (out / rel).write_text(dump(cur), encoding="utf-8"); files.append(rel); cur, size = [], 2
    for it in items:
        n = len(dump(it)) + 1
        assert n < LIMIT, f"one item of {name}.{field} is {n} bytes"
        if size + n > LIMIT:
            flush()
        cur.append(it); size += n
    flush()
    return files

manifest = {}
for k, v in full.items():
    if len(dump(v)) <= 50_000:
        manifest[k] = v
        continue
    comp, sh = {}, {}
    for f, x in v.items():
        if isinstance(x, list) and len(dump(x)) > 30_000:
            sh[f] = shards(k, f, x)
        else:
            comp[f] = x
    if sh:
        comp["__shards__"] = sh
    assert len(dump(comp)) < LIMIT, f"{k} still has {len(dump(comp))} bytes of non-list data"
    (out / "model" / f"{k}.json").write_text(dump(comp), encoding="utf-8")
    manifest[k] = f"@model/{k}.json"
(out / "model.json").write_text(dump(manifest), encoding="utf-8")
sizes = [p.stat().st_size for p in (out / "model").glob("*.json")]
print(f"manifest {(out / 'model.json').stat().st_size} bytes; {len(sizes)} component/shard files, largest {max(sizes)/1024:.0f} KiB")
