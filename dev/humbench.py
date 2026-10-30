#!/usr/bin/env python3
"""Humanizer benchmark: score a variant's rewrites against the originals.
Usage: python dev/humbench.py SPLIT VARIANT      (SPLIT dev|test; VARIANT = folder name under corpus/humbench/SPLIT, 'orig' for the unedited AI texts)
Reports per text: max-mode score (0-100), facts check (verify_rewrite), length ratio; summary: mean score,
share under the 60 and 85 lines (85 = verdict). Scoring uses the shipped model, one text at a time (cached features)."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "ai-detect-humanize" / "scripts"))
import external_eval as ee  # noqa: E402
B = ROOT / "corpus" / "humbench"
split, variant = sys.argv[1], sys.argv[2]
files = sorted((B / split / variant).glob("*.txt"))
rows = [{"text": f.read_text(encoding="utf-8"), "label": 1, "genre": "x:hb", "generator": variant, "split": split, "source": f.name} for f in files]
ee.compute_features(rows, True, True)
cache = ee.load_cache()
sc, ok, ratios = [], 0, []
for f, r in zip(files, rows):
    s = (ee.score(r["text"], cache[ee.key(r["text"])], "max") or 0) * 100
    sc.append(s)
    line = f"{f.name} {s:5.1f}"
    if variant != "orig":
        o = B / split / "orig" / f.name
        p = subprocess.run([sys.executable, str(ROOT.parent / "ai-detect-humanize" / "scripts" / "verify_rewrite.py"), str(o), str(f)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        ok += p.returncode == 0
        ratio = len(r["text"].split()) / max(1, len(o.read_text(encoding="utf-8").split()))
        ratios.append(ratio)
        line += f"  facts {'ok' if p.returncode == 0 else 'FAIL'}  len x{ratio:.2f}"
    print(line)
sc = np.array(sc)
print(f"\n{split}/{variant}: n={len(sc)} mean {sc.mean():.1f} median {np.median(sc):.1f} | under 60: {np.mean(sc < 60):.0%} | under 85 (not flagged): {np.mean(sc < 85):.0%}"
      + (f" | facts ok {ok}/{len(sc)} | length x{np.mean(ratios):.2f}" if ratios else ""))
