#!/usr/bin/env python3
"""Human texts for genres the first corpus had no human data for.

Usage: python dev/build_business_corpus.py

Adds to dev/corpus/human.jsonl (keeps existing rows):
  email    Enron ham emails (2000-2002, SetFit/enron_spam)
  report   US government reports, GAO/CRS (ccdv/govreport-summarization)
  review   Yelp reviews (<= 2015, Yelp/yelp_review_full), long ones only
  post     Reddit posts (webis/tldr-17, 2006-2016): personal posts, blog-like

All predate LLM text. Also writes dev/corpus/business_topics.json - one topic
seed per text - so matching AI texts can be written about the SAME topics.
Large parquet files are downloaded to dev/corpus/raw/ (ignored by git).
"""
import json
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "corpus" / "raw"
OUT = ROOT / "corpus" / "human.jsonl"
TOPICS = ROOT / "corpus" / "business_topics.json"
random.seed(21)


def words(t):
    return len(t.split())


def parquet_url(dataset, split, shard=0):
    j = json.loads(subprocess.run(["curl", "-sSL", "--max-time", "60",
                                   f"https://datasets-server.huggingface.co/parquet?dataset={dataset}"],
                                  capture_output=True, check=True).stdout)
    files = [f for f in j["parquet_files"] if f["split"] == split]
    return files[shard]["url"]


def download(dataset, split, shard=0):
    RAW.mkdir(parents=True, exist_ok=True)
    name = f"{dataset.replace('/', '__')}__{split}__{shard}.parquet"
    path = RAW / name
    if not path.exists():
        print("downloading", dataset, split, flush=True)
        subprocess.run(["curl", "-sSL", "--retry", "5", "--max-time", "1800", "-o", str(path),
                        parquet_url(dataset, split, shard)], check=True)
    return path


def window(text, lo, hi):
    """First lo..hi words, ending on a sentence boundary."""
    paras = [p.strip() for p in re.split(r"\n\s*\n|\r\n\r\n", text) if p.strip()]
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
    return s if words(s) >= lo * 0.8 else None


def clean_email(t):
    t = re.sub(r"(?im)^(from|to|cc|bcc|sent|subject|date):.*$", "", t)
    t = re.sub(r"-{3,}.*?(original message|forwarded by).*", "", t, flags=re.S | re.I)
    t = re.sub(r"\n>.*", "", t)
    t = re.sub(r"\S+@\S+", "", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def enron(n=60):
    """AESLC: the Enron emails with their original casing and punctuation (the SetFit
    enron_spam copy is lowercased and tokenised - a shortcut the model would learn)."""
    import pyarrow.parquet as pq
    parts = [pq.read_table(download("Yale-LILY/aeslc", sp)).to_pandas() for sp in ("train", "validation", "test")]
    import pandas as pd
    df = pd.concat(parts).sample(frac=1, random_state=3)
    out, topics = [], []
    for _, r in df.iterrows():
        body = clean_email(str(r["email_body"]))
        if 150 <= words(body) <= 700 and not re.search(r"forwarded|original message|http", body, re.I):
            w = window(body, 150, 450)
            if w and w != w.lower() and " ," not in w:
                out.append({"text": w, "source": "enron", "genre": "email"})
                topics.append(str(r["subject_line"] or "business matter")[:120])
        if len(out) >= n:
            break
    return out, topics


def govreport(n=50):
    import pyarrow.parquet as pq
    df = pq.read_table(download("ccdv/govreport-summarization", "test")).to_pandas().sample(frac=1, random_state=3)
    out, topics = [], []
    for _, r in df.iterrows():
        sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", r["report"])
        sents = [x for x in sents if 6 <= words(x) <= 60]
        if len(sents) < 60:
            continue
        i = random.randint(len(sents) // 5, len(sents) // 2)
        chunk, n_w = [], 0
        for x in sents[i:]:
            chunk.append(x)
            n_w += words(x)
            if n_w >= 260:
                break
        # the source has no paragraph breaks; restore them (4-5 sentences each) so layout
        # alone can't tell the model "human"
        paras = [" ".join(chunk[k:k + random.choice((4, 5))]) for k in range(0, len(chunk), 4)]
        text = "\n\n".join(paras)
        if 180 <= words(text) <= 450:
            out.append({"text": text, "source": "govreport", "genre": "report"})
            topics.append(str(r["summary"])[:160].split(". ")[0])
        if len(out) >= n:
            break
    return out, topics


def yelp(n=50):
    import pyarrow.parquet as pq
    df = pq.read_table(download("Yelp/yelp_review_full", "test")).to_pandas().sample(frac=1, random_state=3)
    out, topics = [], []
    for _, r in df.iterrows():
        t = r["text"].replace("\\n", "\n").strip()
        if 150 <= words(t) <= 500:
            out.append({"text": t, "source": "yelp", "genre": "review"})
            topics.append(f"a {int(r['label']) + 1}-star review of a local business")
        if len(out) >= n:
            break
    return out, topics


def reddit(n=50):
    import pyarrow.parquet as pq
    path = download("webis/tldr-17", "train")
    df = pq.read_table(path, columns=["subreddit", "content"]).to_pandas()
    df = df[df["content"].str.len() > 800].sample(frac=1, random_state=3)
    out, topics = [], []
    for _, r in df.iterrows():
        t = str(r["content"]).strip()
        t = re.sub(r"http\S+", "", t)
        if 170 <= words(t) <= 500 and "tl;dr" not in t.lower()[:50]:
            out.append({"text": t, "source": f"reddit:{r['subreddit']}", "genre": "post"})
            topics.append(f"a personal post on r/{r['subreddit']}")
        if len(out) >= n:
            break
    return out, topics


def main():
    rows = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["genre"] not in {"email", "report", "review", "post"}]
    topics = {}
    for fn in (enron, govreport, yelp, reddit):
        try:
            got, tp = fn()
        except Exception as e:
            print(f"{fn.__name__} failed: {type(e).__name__}: {e}")
            continue
        print(f"{fn.__name__}: {len(got)}")
        rows += got
        if got:
            topics[got[0]["genre"]] = tp[:len(got)]
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    TOPICS.write_text(json.dumps(topics, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(rows)} human rows; topics for {list(topics)}")


if __name__ == "__main__":
    main()
