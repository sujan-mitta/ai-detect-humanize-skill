#!/usr/bin/env python3
"""False-positive test on real pre-2021 human report/technical texts (dev/corpus/human_web).
Usage: python dev/humanweb_eval.py [MODEL.json]"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "skills" / "humanize" / "scripts"))
import external_eval as ee  # noqa: E402
import ai_score  # noqa: E402
rows = []
HELD = "--held-odd" in sys.argv
sys.argv = [a for a in sys.argv if a != "--held-odd"]
files = []
for g in ("report", "technical"):
    fs = sorted((ROOT / "corpus" / "human_web" / g).glob("*.txt"))
    files += [f for i, f in enumerate(fs) if not HELD or i % 2 == 1]
for f in files:
    t = f.read_text(encoding="utf-8")
    if len(t.split()) >= 100:
        rows.append({"text": t, "label": 0, "genre": "x:" + f.parent.name, "generator": "human_web",
                     "split": f.parent.name, "source": "human_web"})
json.dump(rows, open(ROOT / "runs" / "humanweb_rows.json", "w", encoding="utf-8"), ensure_ascii=False)
ee.compute_features(rows, True, True)
cache = ee.load_cache()
if len(sys.argv) > 1:
    ai_score.MODEL_PATH = Path(sys.argv[1]); ai_score._MODELS = None
thr = ai_score.load_models()["max"]["thresholds"]["ai"] / 100
for g in ("report", "technical"):
    s = np.array([ee.score(r["text"], cache[ee.key(r["text"])], "max") or 0 for r in rows if r["split"] == g])
    print(f"{g}: n={len(s)} flagged>=60 {np.mean(s >= .6):.1%}  verdict {np.mean(s >= thr):.1%}  median {np.median(s):.2f}")
