#!/usr/bin/env python3
"""More human text for two weak genres, plus topic seeds for matched AI texts.

  report     +100 US government report passages (GAO/CRS), distinct from the 49 already used.
             Formal bureaucratic prose is what the detector flags most often as AI (16% of
             human reports in leave-one-genre-out), so it needs more human examples.
  technical  +60 IETF RFC passages (1981-2018, human-written protocol specifications and
             explanations). Technical writing is the weakest AI genre (31% recall).

Appends to dev/corpus/human.jsonl; writes seeds under "report2" / "technical2" in
dev/corpus/business_topics.json (existing keys untouched).
"""
import json
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "ai-detect-humanize" / "scripts"))
from build_business_corpus import download, words  # noqa: E402
from extract_text import reflow  # noqa: E402

OUT = ROOT / "corpus" / "human.jsonl"
TOPICS = ROOT / "corpus" / "business_topics.json"
random.seed(77)

RFCS = [768, 791, 793, 826, 854, 959, 1034, 1035, 1122, 1123, 1321, 1738, 1918, 1945, 2045, 2046, 2104, 2119, 2131,
        2246, 2326, 2328, 2460, 2616, 2617, 2818, 2821, 2822, 3022, 3261, 3315, 3550, 3986, 4001, 4034, 4120, 4193,
        4251, 4271, 4291, 4346, 4366, 4510, 4566, 4627, 4861, 5246, 5321, 5322, 5389, 5681, 5735, 5880, 5905, 6265,
        6455, 6749, 6762, 6797, 6847, 6891, 7231, 7233, 7234, 7235, 7519, 7540, 8200, 8259, 8446, 8484, 3339, 3629,
        4122, 4648, 3164, 2865, 3748, 4291, 5424, 6376]


def window(paras, lo, hi):
    out, n = [], 0
    for p in paras:
        out.append(p)
        n += words(p)
        if n >= lo:
            break
    s = "\n\n".join(out)
    if words(s) > hi:
        cut = " ".join(s.split()[:hi])
        s = cut[: cut.rfind(".") + 1] or cut
    return s if lo * 0.85 <= words(s) <= hi + 20 else None


def reports(n=100):
    import pandas as pd
    import pyarrow.parquet as pq
    df = pq.read_table(download("ccdv/govreport-summarization", "test")).to_pandas().sample(frac=1, random_state=11)
    seen = {json.loads(l)["text"][:80] for l in OUT.read_text(encoding="utf-8").splitlines() if '"report"' in l}
    out, topics = [], []
    for _, r in df.iterrows():
        sents = [x for x in re.split(r"(?<=[.!?])\s+(?=[A-Z])", r["report"]) if 6 <= words(x) <= 60]
        if len(sents) < 60:
            continue
        i = random.randint(len(sents) // 4, len(sents) * 3 // 4)
        chunk, nw = [], 0
        for x in sents[i:]:
            chunk.append(x)
            nw += words(x)
            if nw >= 260:
                break
        paras = [" ".join(chunk[k:k + random.choice((4, 5))]) for k in range(0, len(chunk), 4)]
        text = "\n\n".join(paras)
        if 180 <= words(text) <= 450 and text[:80] not in seen:
            seen.add(text[:80])
            out.append({"text": text, "source": "govreport2", "genre": "report", "label": 0})
            topics.append(str(r["summary"])[:160].split(". ")[0])
        if len(out) >= n:
            break
    return out, topics


def rfc(n=60):
    out, topics = [], []
    for num in RFCS:
        if len(out) >= n:
            break
        raw = subprocess.run(["curl", "-sSL", "--max-time", "60", f"https://www.rfc-editor.org/rfc/rfc{num}.txt"],
                             capture_output=True).stdout.decode("utf-8", "replace")
        if len(raw) < 5000 or "<html" in raw[:300].lower():
            continue
        title = None
        head = raw[:2500].split("\n")
        # the title is the first long, centred, non-header line after the author block
        for line in head:
            s = line.strip()
            if 12 < len(s) < 75 and not re.search(r"Request for Comments|Network Working|Category|ISSN|^\w+ \d{4}$|Obsoletes|Updates|STANDARDS|Status of", s, re.I) \
                    and line.startswith("   ") and not re.search(r"\d{4}\s*$", s) and "@" not in s:
                title = s
                break
        body = re.sub(r"\f.*?\n", "\n", raw)
        body = re.sub(r"\n(RFC \d+.*|[A-Z][^\n]{0,60}\[Page \d+\])\n", "\n", body)
        # explanatory prose: unindented-after-3-spaces paragraphs of real sentences (skip diagrams, tables, code)
        paras = []
        for blk in re.split(r"\n\s*\n", body):
            lines = [l for l in blk.split("\n") if l.strip()]
            if len(lines) < 3:
                continue
            flat = " ".join(l.strip() for l in lines)
            letters = sum(ch.isalpha() for ch in flat) / max(1, len(flat))
            if words(flat) >= 45 and letters > 0.8 and flat.count(".") >= 2 and not re.search(r"^\s*(\d+\.|[A-Z]\.)\s", blk) and "   " not in flat:
                paras.append(flat)
        if len(paras) < 8:
            continue
        mid = paras[len(paras) // 5: len(paras) * 4 // 5]
        k = random.randint(0, max(0, len(mid) - 6))
        w = window(mid[k:k + 8], 190, 450)
        if w:
            out.append({"text": w, "source": f"rfc{num}", "genre": "technical", "label": 0})
            topics.append(f"{title or 'an internet protocol specification'} (RFC {num})")
    return out, topics


def main():
    rows = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines()]
    topics = json.loads(TOPICS.read_text(encoding="utf-8"))
    for name, fn, key in (("govreport2", reports, "report2"), ("rfc", rfc, "technical2")):
        got, tp = fn()
        print(f"{name}: {len(got)}")
        rows = [r for r in rows if not (r["source"] == name or (name == "rfc" and r["source"].startswith("rfc")))]
        rows += got
        topics[key] = tp[:len(got)]
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    TOPICS.write_text(json.dumps(topics, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", len(rows), "human rows;", {k: len(v) for k, v in topics.items()})
    print("sample technical topics:", topics.get("technical2", [])[:4])
    t = next(r for r in rows if r["source"].startswith("rfc"))
    print("\nsample RFC text:\n", t["text"][:500])


if __name__ == "__main__":
    main()
