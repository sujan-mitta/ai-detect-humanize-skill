#!/usr/bin/env python3
"""Human texts from genres the detector has NEVER seen, as a pure false-positive test.

  legal     CRS bill summaries (human-written, BillSum 2016-era)       60
  statute   US bill text, prose sections only (BillSum)               40
  medical   medical textbook passages (MedRAG textbooks: Gray's Anatomy, Harrison's, ...)  60
Written to dev/corpus/newgenre_human.jsonl (NOT part of training: used to measure false
positives on unseen genres; may be folded into training afterwards).
"""
import json
import random
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "corpus" / "raw"
random.seed(61)
words = lambda t: len(t.split())


def legal(n=60):
    out = []
    for s in pq.read_table(RAW / "billsum_test.parquet", columns=["summary"]).column("summary").to_pylist():
        s = re.sub(r"\s+", " ", s).strip()
        if 150 <= words(s) <= 450:
            out.append({"text": s, "source": f"billsum:{len(out)}", "genre": "legal", "label": 0})
    random.shuffle(out)
    return out[:n]


def statute(n=40):
    out = []
    for t in pq.read_table(RAW / "billsum_test.parquet", columns=["text"]).column("text").to_pylist():
        paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", t)]
        paras = [p.replace("``", '"').replace("''", '"') for p in paras
                 if words(p) >= 40 and not re.match(r"^(SEC\.|\(\w+\)|[A-Z ]{12,})", p)]
        if len(paras) >= 3:
            k = random.randint(0, len(paras) - 3)
            s = " ".join(paras[k:k + 3])
            if 150 <= words(s) <= 450:
                out.append({"text": s, "source": f"billtext:{len(out)}", "genre": "statute", "label": 0})
    random.shuffle(out)
    return out[:n]


def medical(n=60):
    t = pq.read_table(RAW / "medtext.parquet", columns=["id", "title", "content"]).to_pandas()
    out = []
    for title, g in t.groupby("title"):
        g = g.reset_index(drop=True)
        taken = 0
        for _ in range(30):
            if len(g) < 5 or taken >= 4:
                break
            i = random.randint(0, len(g) - 4)
            s = " ".join(g.loc[i:i + 3, "content"].tolist())
            s = re.sub(r"\s+", " ", s)
            if 170 <= words(s) <= 480 and s.count(". ") >= 4:
                out.append({"text": s, "source": f"medtext:{title}", "genre": "medical", "label": 0})
                taken += 1
    random.shuffle(out)
    return out[:n]


if __name__ == "__main__":
    rows = legal() + statute() + medical()
    (ROOT / "corpus" / "newgenre_human.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    from collections import Counter
    print(Counter(r["genre"] for r in rows))
    for g in ("legal", "statute", "medical"):
        print(g, next(r["text"][:240] for r in rows if r["genre"] == g))
