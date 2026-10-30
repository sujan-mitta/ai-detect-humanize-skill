#!/usr/bin/env python3
"""Why do humanized rewrites still score AI? Compare feature means: originals vs rewrites vs real humans.
Caches features for corpus/heldout/ai (36) and humanized_tool/{opus,sonnet} (12)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "ai-detect-humanize" / "scripts"))
import external_eval as ee, calibrate as c  # noqa: E402
H = ROOT / "corpus" / "heldout"
rows = []
for f in sorted((H / "ai").glob("*.txt")):
    rows.append({"text": f.read_text(encoding="utf-8"), "label": 1, "genre": "x:h", "generator": "orig", "split": "orig", "source": f.stem})
for who in ("opus", "sonnet"):
    for f in sorted((H / "humanized_tool" / who).glob("*.txt")):
        rows.append({"text": f.read_text(encoding="utf-8"), "label": 1, "genre": "x:h", "generator": who, "split": "hum", "source": f.stem})
ee.compute_features(rows, True, True)
cache = ee.load_cache()
names = c.DEEP_FEATURES + c.NEURAL_FEATURES
def vec(t):
    r = cache[ee.key(t)]; v = {**(r.get("deep") or {}), **(r.get("neural") or {})}
    return [v.get(n) for n in names]
groups = {"orig": [], "hum": []}
for r in rows: groups[r["split"]].append(vec(r["text"]))
hum_ref = [vec(r["text"]) for r in json.load(open(ROOT / "runs" / "humanweb_rows.json", encoding="utf-8")) if ee.key(r["text"]) in cache]
groups["human(web)"] = hum_ref
print(f"{'feature':34s}" + "".join(f"{k:>12s}" for k in groups))
for i, n in enumerate(names):
    print(f"{n:34s}" + "".join(f"{np.nanmean([x[i] for x in v if x[i] is not None]):12.3f}" for v in groups.values()))
