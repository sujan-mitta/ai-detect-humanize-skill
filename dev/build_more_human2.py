#!/usr/bin/env python3
"""Second round of human text aimed at the remaining weak spots.

  encyclopedic  +100 WikiText-103 articles (Wikipedia snapshot, 2016)    -> 5-10% false positives
  academic      +80 more arXiv abstracts (2015-2020, 8 new categories)
  review        +100 Yelp reviews (<= 2015)
  report        +80 more GAO/CRS passages
  literary      +100 Reddit WritingPrompts stories (amateur fiction, 2018) - casual human fiction, so
                that casual voice stops being evidence of AI
Appends to dev/corpus/human.jsonl. Topic seeds for matched AI texts go to
dev/corpus/ai_new/topics2.json (fiction seeds are the writing prompts the humans answered).
"""
import json
import random
import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "ai-detect-humanize" / "scripts"))
import pyarrow.parquet as pq  # noqa: E402
from build_business_corpus import words  # noqa: E402
from build_human_corpus import get  # noqa: E402
from ai_score import canon_typography  # noqa: E402

RAW = ROOT / "corpus" / "raw"
OUT = ROOT / "corpus" / "human.jsonl"
random.seed(99)
clean = lambda s: re.sub(r"[^\x20-\x7e]", "-", s).strip()


def wikitext(n=100):
    arts, cur = [], None
    for f in ("wt_test.parquet", "wt_val.parquet"):
        for t in pq.read_table(RAW / f).column("text").to_pylist():
            if re.match(r"^ = [^=].* = \n$", t):
                cur = {"title": t.strip(" =\n"), "paras": []}
                arts.append(cur)
            elif cur is not None and t.strip() and not t.startswith(" ="):
                cur["paras"].append(t.strip())
    out, topics = [], []
    for a in arts:
        paras = [re.sub(r" @(.)@ ", r"\1", p) for p in a["paras"] if words(p) >= 30]
        paras = [canon_typography(p) for p in paras]
        if len(paras) < 3:
            continue
        i = random.randint(0, max(0, len(paras) - 3))
        chunk, k = [], 0
        for p in paras[i:]:
            chunk.append(p)
            k += words(p)
            if k >= 200:
                break
        text = "\n\n".join(chunk)
        if 170 <= words(text) <= 480:
            out.append({"text": text, "source": f"wikitext:{a['title']}", "genre": "encyclopedic"})
            topics.append(f"the Wikipedia article '{clean(a['title'])}' (one section of it)")
        if len(out) >= n:
            break
    return out, topics


def arxiv(n=80):
    out, topics = [], []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for cat in ["cs.LG", "physics.gen-ph", "math.ST", "astro-ph.GA", "cond-mat.stat-mech", "econ.EM", "cs.SE", "q-bio.PE"]:
        q = urllib.parse.quote(f"cat:{cat} AND submittedDate:[201501010000 TO 202012312359]")
        import time
        root = None
        for attempt in range(4):
            try:
                xml = get(f"http://export.arxiv.org/api/query?search_query={q}&start=20&max_results=14")
                root = ET.fromstring(xml)
                break
            except Exception:
                time.sleep(8)
        if root is None:
            print("arxiv skip", cat, flush=True)
            continue
        time.sleep(4)
        for e in root.findall("a:entry", ns):
            t = re.sub(r"\s+", " ", e.find("a:summary", ns).text).strip()
            title = re.sub(r"\s+", " ", e.find("a:title", ns).text).strip()
            if words(t) >= 120:
                out.append({"text": t, "source": f"arxiv2:{cat}", "genre": "academic"})
                topics.append(f"a research paper abstract titled '{clean(title)}'")
    z = list(zip(out, topics))
    random.shuffle(z)
    z = z[:n]
    return [a for a, _ in z], [b for _, b in z]


def yelp(n=100):
    df = pq.read_table(RAW / "Yelp__yelp_review_full__test__0.parquet").to_pandas().sample(frac=1, random_state=9)
    seen = {json.loads(l)["text"][:80] for l in OUT.read_text(encoding="utf-8").splitlines() if '"review"' in l}
    out, topics = [], []
    for _, r in df.iterrows():
        t = r["text"].replace("\n", "\n").strip()
        if 150 <= words(t) <= 450 and t[:80] not in seen:
            out.append({"text": t, "source": "yelp2", "genre": "review"})
            topics.append(f"a {int(r['label']) + 1}-star review of a local business")
        if len(out) >= n:
            break
    return out, topics


def reports(n=80):
    df = pq.read_table(RAW / "ccdv__govreport-summarization__test__0.parquet").to_pandas().sample(frac=1, random_state=12)
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
            if nw >= 250:
                break
        text = "\n\n".join(" ".join(chunk[k:k + 4]) for k in range(0, len(chunk), 4))
        if 180 <= words(text) <= 450:
            out.append({"text": text, "source": "govreport3", "genre": "report"})
            topics.append(clean(str(r["summary"])[:160].split(". ")[0]))
        if len(out) >= n:
            break
    return out, topics


BAD = re.compile(r"kill|murder|rape|sex|suicid|dead|death|die\b|dies\b|war\b|blood|torture|abuse|nazi|terror|weapon|gun|shoot|drug|slave|hell\b|demon|zombie|apocalyp", re.I)


def fiction(n=100):
    df = pq.read_table(RAW / "wp_test.parquet").to_pandas().sample(frac=1, random_state=5)
    out, topics = [], []
    for _, r in df.iterrows():
        prompt = re.sub(r"^\[\s*\w+\s*\]\s*", "", r["prompt"]).strip()
        story = r["story"].replace("<newline>", "\n").strip()
        if BAD.search(prompt) or BAD.search(story) or not 15 <= words(prompt) <= 60:
            continue
        story = canon_typography(story)
        parts = story.split()
        if len(parts) < 230:
            continue
        cut = " ".join(parts[:random.randint(230, 420)])
        cut = cut[: cut.rfind(".") + 1] or cut
        if words(cut) < 200:
            continue
        out.append({"text": cut, "source": f"writingprompts:{len(out)}", "genre": "literary"})
        topics.append(clean(prompt))
        if len(out) >= n:
            break
    return out, topics


def main():
    rows = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if not r["source"].startswith(("wikitext:", "arxiv2:", "yelp2", "govreport3", "writingprompts:"))]
    topics = {}
    for key, fn in (("encyclopedic", wikitext), ("academic", arxiv), ("review", yelp), ("report", reports), ("fiction", fiction)):
        got, tp = fn()
        print(key, len(got), flush=True)
        rows += got
        topics[key] = tp[:len(got)]
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (ROOT / "corpus" / "ai_new" / "topics2.json").write_text(json.dumps(topics, indent=1), encoding="utf-8")
    print("human rows:", len(rows))
    for k in ("encyclopedic", "fiction"):
        print(k, "->", topics[k][:2])


if __name__ == "__main__":
    main()
