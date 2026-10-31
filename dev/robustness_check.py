#!/usr/bin/env python3
"""Live max-mode robustness: original vs homoglyph + zero-width attack vs typography edits vs hard-wrapped lines (4 API texts)."""
import json, random, sys, textwrap
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "ai-detect-humanize" / "scripts"))
import ai_score
rows = json.load(open(__import__("pathlib").Path(__file__).resolve().parent / "runs" / "api_rows.json", encoding="utf-8"))
random.seed(2)
homo = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х"}
attack = lambda t: "".join((homo[c] if c in homo and random.random() < .25 else c) + ("\u200b" if c == " " and random.random() < .1 else "") for c in t)
fmt = lambda t: t.replace(". ", ".  ").replace('"', "“").replace("'", "’")
wrap = lambda t: "\n".join("\n".join(textwrap.wrap(p, 70)) for p in t.split("\n"))
print("displayed max score: original | homoglyph+zero-width | typography | hard-wrapped")
for r in (rows[3], rows[150], rows[260], rows[40]):
    t = r["text"]
    print("  ", [ai_score.build(x, deep=True, max_mode=True)["score"] for x in (t, attack(t), fmt(t), wrap(t))])
