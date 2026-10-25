#!/usr/bin/env python3
"""Download human-written texts for calibrating the scorer.

Writes dev/corpus/human.jsonl with {"text", "source", "genre", "label": 0}.
Every source predates widespread LLM text (or is public-domain literature),
so the samples are human-written with high confidence.

Sources: 20 Newsgroups (1990s Usenet, casual), arXiv abstracts before 2021
(academic), Wikipedia article intros (encyclopedic), Project Gutenberg books
(literary / essay), Paul Graham essays (modern personal essay).
"""
import json
import random
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path(__file__).parent / "corpus" / "human.jsonl"
UA = {"User-Agent": "ai-detect-humanize-calibration/1.0 (research script)"}
random.seed(7)


def get(url):
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read().decode("utf-8", errors="replace")
    except urllib.error.URLError:
        # Some machines intercept TLS with a local root CA that Python doesn't
        # trust; curl uses the OS certificate store, so fall back to it.
        out = subprocess.run(["curl", "-sSL", "--max-time", "90", "-A", UA["User-Agent"], url],
                             capture_output=True, check=True)
        return out.stdout.decode("utf-8", errors="replace")


def words(t):
    return len(t.split())


def window(text, lo=180, hi=450):
    """Cut a passage of lo..hi words ending on a sentence boundary."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if words(p) > 20]
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


def newsgroups(n=90):
    from sklearn.datasets import fetch_20newsgroups
    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    idx = list(range(len(data.data)))
    random.shuffle(idx)
    out = []
    for i in idx:
        t = re.sub(r"[ \t]+", " ", data.data[i]).strip()
        if 150 <= words(t) <= 600 and t.count("|") < 5 and t.count(">") < 3 and not re.search(r"-----BEGIN|\w+@\w+\.\w+.*\n.*\w+@", t):
            out.append({"text": t, "source": "20newsgroups", "genre": "forum"})
        if len(out) >= n:
            break
    return out


def arxiv(n=60):
    out = []
    for cat in ["cs.CL", "q-bio.NC", "physics.soc-ph", "econ.GN", "cs.CY", "stat.ME"]:
        q = urllib.parse.quote(f"cat:{cat} AND submittedDate:[201501010000 TO 202012312359]")
        xml = get(f"http://export.arxiv.org/api/query?search_query={q}&start=0&max_results=15")
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for e in ET.fromstring(xml).findall("a:entry", ns):
            t = re.sub(r"\s+", " ", e.find("a:summary", ns).text).strip()
            if words(t) >= 120:
                out.append({"text": t, "source": f"arxiv:{cat}", "genre": "academic"})
    random.shuffle(out)
    return out[:n]


def wikipedia(n=50):
    titles = ["Photosynthesis", "Roman_Empire", "Jazz", "Volcano", "Printing_press", "Coral_reef",
              "French_Revolution", "Black_hole", "Tea", "Bicycle", "Antibiotic", "Silk_Road",
              "Honey_bee", "Plate_tectonics", "Chess", "Great_Depression", "Vaccine", "Lighthouse",
              "Monsoon", "Penicillin", "Origami", "Steam_engine", "Octopus", "Inflation",
              "Bronze_Age", "Telescope", "Glacier", "Opera", "Cotton", "Typhoid_fever",
              "Suez_Canal", "Ballet", "Coffee", "Hurricane", "Rice", "Mount_Everest",
              "Printing", "Desert", "Elephant", "Railway", "Chocolate", "Earthquake",
              "Renaissance", "Wolf", "Salt", "Library", "Paper", "Rainforest", "Tide", "Bread"]
    out = []
    for t in titles:
        try:
            url = ("https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1"
                   f"&exsectionformat=plain&format=json&titles={t}")
            pages = json.loads(get(url))["query"]["pages"]
            text = next(iter(pages.values())).get("extract", "")
            w = window(text, 200, 420)
            if w:
                out.append({"text": w, "source": f"wikipedia:{t}", "genre": "encyclopedic"})
        except Exception as e:  # network hiccup: skip this article
            print("wiki skip", t, e)
        if len(out) >= n:
            break
    return out


def gutenberg(n=50):
    books = {1342: "Pride and Prejudice", 2701: "Moby Dick", 1661: "Sherlock Holmes", 84: "Frankenstein",
             98: "A Tale of Two Cities", 205: "Walden", 1080: "A Modest Proposal", 2591: "Grimms",
             174: "Dorian Gray", 345: "Dracula", 16328: "Beowulf", 1232: "The Prince",
             4300: "Ulysses", 5200: "Metamorphosis", 76: "Huckleberry Finn", 1952: "Yellow Wallpaper",
             2148: "Poe works", 3207: "Leviathan", 1497: "Republic", 768: "Wuthering Heights"}
    out = []
    for bid, name in books.items():
        try:
            raw = get(f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt")
        except Exception as e:
            print("gutenberg skip", name, e)
            continue
        raw = raw.replace("\r\n", "\n")
        body = raw.split("*** START", 1)[-1].split("*** END", 1)[0]
        body = re.sub(r"(?<!\n)\n(?!\n)", " ", body)  # unwrap hard line breaks
        paras = [p for p in re.split(r"\n\s*\n", body) if words(p) > 40]
        if len(paras) < 40:
            continue
        for start in random.sample(range(10, len(paras) - 10), 3):
            w = window("\n\n".join(paras[start:start + 8]), 200, 420)
            if w:
                out.append({"text": w, "source": f"gutenberg:{name}", "genre": "literary"})
    random.shuffle(out)
    return out[:n]


def paulgraham(n=25):
    slugs = ["greatwork", "useful", "smart", "read", "woke", "goodwriting", "heresy", "alien",
             "think", "richnow", "fn", "wisdom", "early", "persistence", "reddits", "simply",
             "best", "superlinear", "getideas", "own", "hwh", "worked", "writes", "kids", "words",
             "selfindulgence", "real", "field", "weird", "conformism"]
    out = []
    for s in slugs:
        try:
            html = get(f"https://paulgraham.com/{s}.html")
        except Exception:
            continue
        text = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"&\w+;", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        m = re.search(r"((January|February|March|April|May|June|July|August|September|October|November|December) \d{4})", text)
        if m:
            text = text[m.end():]
        w = window(text, 200, 450)
        if w:
            out.append({"text": w, "source": f"paulgraham:{s}", "genre": "essay"})
        if len(out) >= n:
            break
    return out


def stackexchange(n=80):
    """Accepted/high-voted answers from before 2020 - human explanatory writing,
    the closest human match to an AI assistant's answer. Also saves the question
    titles so AI answers to the same questions can be generated."""
    import gzip
    import html as html_mod
    sites = ["cooking", "diy", "travel", "workplace", "writing", "gardening", "bicycles", "money",
             "parenting", "pets", "photo", "outdoors"]
    out, questions = [], []
    for site in sites:
        url = ("https://api.stackexchange.com/2.3/answers?order=desc&sort=votes&pagesize=12"
               f"&todate=1577836800&site={site}&filter=withbody")
        try:
            raw = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
        except Exception:
            raw = subprocess.run(["curl", "-sSL", "--compressed", "--max-time", "60", url],
                                 capture_output=True, check=True).stdout
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        data = json.loads(raw.decode("utf-8"))
        ids = ";".join(str(i["question_id"]) for i in data.get("items", []))
        qraw = subprocess.run(["curl", "-sSL", "--compressed", "--max-time", "60",
                               f"https://api.stackexchange.com/2.3/questions/{ids}?site={site}"],
                              capture_output=True, check=True).stdout
        qtitles = {q["question_id"]: q["title"] for q in json.loads(qraw.decode("utf-8")).get("items", [])}
        for item in data.get("items", []):
            item["title"] = qtitles.get(item["question_id"], "")
            body = item.get("body", "")
            body = re.sub(r"<pre.*?</pre>|<code>.*?</code>|<blockquote>.*?</blockquote>", " ", body, flags=re.S)
            body = re.sub(r"<li>", "\n- ", body)
            body = re.sub(r"</p>|<br\s*/?>|</h\d>", "\n\n", body)
            text = html_mod.unescape(re.sub(r"<[^>]+>", "", body))
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            title = html_mod.unescape(item.get("title", ""))
            if 150 <= words(text) <= 600 and title:
                out.append({"text": text, "source": f"stackexchange:{site}", "genre": "qa"})
                questions.append({"site": site, "title": title})
    (OUT.parent / "se_questions.json").write_text(json.dumps(questions, indent=1), encoding="utf-8")
    random.shuffle(out)
    return out[:n]


def wikinews(n=40):
    """Wikinews articles (human-written news, 2005-2019)."""
    out, titles = [], []
    url = ("https://en.wikinews.org/w/api.php?action=query&list=categorymembers&cmtitle=Category:Published"
           "&cmlimit=300&cmsort=timestamp&cmdir=asc&cmstart=2012-01-01T00:00:00Z&format=json")
    members = json.loads(get(url))["query"]["categorymembers"]
    random.shuffle(members)
    for m in members:
        t = urllib.parse.quote(m["title"])
        try:
            pages = json.loads(get("https://en.wikinews.org/w/api.php?action=query&prop=extracts&explaintext=1"
                                   f"&format=json&titles={t}"))["query"]["pages"]
        except Exception:
            continue
        text = next(iter(pages.values())).get("extract", "")
        text = re.split(r"\n(Sources|Related news|External links)\n", text)[0]
        text = re.sub(r"^\w+day, \w+ \d+, \d{4}\s*", "", text.strip())
        w = window(text, 150, 420)
        if w:
            out.append({"text": w, "source": f"wikinews:{m['title'][:60]}", "genre": "news"})
            titles.append(m["title"])
        if len(out) >= n:
            break
    (OUT.parent / "news_headlines.json").write_text(json.dumps(titles, indent=1), encoding="utf-8")
    return out


def peps(n=40):
    """Python Enhancement Proposals (public domain, human-written technical prose
    with code). Saves titles so AI write-ups on the same topics can be made."""
    out, titles = [], []
    for num in list(range(201, 400, 6)) + list(range(400, 600, 9)):
        try:
            rst = get(f"https://raw.githubusercontent.com/python/peps/main/peps/pep-{num:04d}.rst")
        except Exception:
            continue
        if "404: Not Found" in rst[:40]:
            continue
        m = re.search(r"^Title:\s*(.+)$", rst, re.M)
        body = rst.split("\n\n", 1)[-1]
        body = re.sub(r"^\.\. .*$", "", body, flags=re.M)              # rst directives
        body = re.sub(r"^(?: {3,}|\t).*$", "", body, flags=re.M)        # indented code / quotes
        body = re.sub(r"^[=\-~^\"`']{3,}\s*$", "", body, flags=re.M)    # heading underlines
        body = re.sub(r"``([^`]+)``", r"`\1`", body)                    # rst literals -> markdown code
        body = re.sub(r"`([^`<]+) <[^>]+>`_+", r"\1", body)             # rst links
        body = re.sub(r"\[\d+\]_", "", body)
        w = window(body, 200, 450)
        if w and m:
            out.append({"text": w, "source": f"pep:{num}", "genre": "technical"})
            titles.append(m.group(1).strip())
        if len(out) >= n:
            break
    (OUT.parent / "pep_titles.json").write_text(json.dumps(titles, indent=1), encoding="utf-8")
    return out


def gutenberg_modern(n=80):
    """Public-domain fiction and essays from 1900-1928, so human fiction isn't
    only 19th-century prose (the model mustn't learn 'old-fashioned = human')."""
    books = {805: ("This Side of Paradise", "literary"), 64317: ("The Great Gatsby", "literary"),
             543: ("Main Street", "literary"), 242: ("My Antonia", "literary"), 5670: ("Jacob's Room", "literary"),
             2814: ("Dubliners", "literary"), 416: ("Winesburg, Ohio", "literary"),
             4217: ("Portrait of the Artist", "literary"), 215: ("The Call of the Wild", "literary"),
             1429: ("The Garden Party", "literary"), 2776: ("The Four Million", "literary"),
             8092: ("Tremendous Trifles", "essay"), 386: ("Virginibus Puerisque", "essay"),
             70: ("What Is Man", "essay"), 64457: ("The Common Reader", "essay"),
             1786: ("And Even Now", "essay"), 4280: ("Heretics", "essay"), 3070: ("Essays of Elia", "essay")}
    out = []
    for bid, (name, genre) in books.items():
        try:
            raw = get(f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt").replace("\r\n", "\n")
        except Exception as e:
            print("skip", name, e)
            continue
        body = raw.split("*** START", 1)[-1].split("*** END", 1)[0]
        body = re.sub(r"(?<!\n)\n(?!\n)", " ", body)
        paras = [p for p in re.split(r"\n\s*\n", body) if words(p) > 30]
        if len(paras) < 30:
            continue
        for start in random.sample(range(5, len(paras) - 8), 5):
            w = window("\n\n".join(paras[start:start + 8]), 200, 420)
            if w:
                out.append({"text": w, "source": f"gutenberg_modern:{name}", "genre": genre})
    random.shuffle(out)
    return out[:n]


def paulgraham2(n=50):
    """More polished modern human essays - the genre most often wrongly flagged."""
    slugs = ["avg", "hp", "say", "gh", "nerds", "taste", "love", "ds", "startupideas", "growth", "wealth",
             "ideas", "marginal", "procrastination", "stuff", "jessica", "bias", "lies", "discover", "before",
             "boss", "fierce", "gap", "future", "noob", "schlep", "really", "relres", "copy", "mean",
             "credentials", "foundermode", "when", "newideas", "users", "pow", "cities", "identity",
             "disagree", "popular", "desres", "lesson", "top", "hiring", "determination", "ambitious",
             "genius", "13sentences", "speak", "heroes", "vb", "fh", "gba", "kate", "smart"]
    out = []
    for s in slugs:
        try:
            html = get(f"https://paulgraham.com/{s}.html")
        except Exception:
            continue
        text = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"&\w+;", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        m = re.search(r"((January|February|March|April|May|June|July|August|September|October|November|December) \d{4})", text)
        if m:
            text = text[m.end():]
        # take a passage from the middle too, not just openings
        paras = [p for p in re.split(r"\n\s*\n", text) if words(p) > 25]
        if len(paras) > 12:
            text = "\n\n".join(paras[len(paras) // 3:])
        w = window(text, 200, 450)
        if w:
            out.append({"text": w, "source": f"paulgraham2:{s}", "genre": "essay"})
        if len(out) >= n:
            break
    return out


def stackexchange2(n=90):
    import gzip
    import html as html_mod
    sites = ["english", "academia", "philosophy", "history", "scifi", "music", "movies", "interpersonal",
             "literature", "linguistics", "fitness", "rpg"]
    out = []
    for site in sites:
        url = (f"https://api.stackexchange.com/2.3/answers?order=desc&sort=votes&pagesize=12"
               f"&todate=1577836800&site={site}&filter=withbody")
        raw = subprocess.run(["curl", "-sSL", "--compressed", "--max-time", "60", url],
                             capture_output=True, check=True).stdout
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        for item in json.loads(raw.decode("utf-8")).get("items", []):
            body = item.get("body", "")
            body = re.sub(r"<pre.*?</pre>|<code>.*?</code>|<blockquote>.*?</blockquote>", " ", body, flags=re.S)
            body = re.sub(r"<li>", "\n- ", body)
            body = re.sub(r"</p>|<br\s*/?>|</h\d>", "\n\n", body)
            text = html_mod.unescape(re.sub(r"<[^>]+>", "", body))
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if 150 <= words(text) <= 600:
                out.append({"text": text, "source": f"stackexchange2:{site}", "genre": "qa"})
    random.shuffle(out)
    return out[:n]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sources = (newsgroups, arxiv, wikipedia, gutenberg, paulgraham, stackexchange, wikinews, peps,
               gutenberg_modern, paulgraham2, stackexchange2)
    # optional: only (re)fetch named sources and replace just those rows
    only = set(sys.argv[1:])
    rows = []
    if only and OUT.exists():
        prefix = {"newsgroups": "20newsgroups"}
        drop = {prefix.get(o, o) for o in only}
        rows = [r for r in map(json.loads, OUT.read_text(encoding="utf-8").splitlines())
                if r["source"].split(":")[0] not in drop]
        sources = [fn for fn in sources if fn.__name__ in only]
    for fn in sources:
        try:
            got = fn()
        except Exception as e:
            print(f"{fn.__name__} failed: {e}")
            got = []
        print(f"{fn.__name__}: {len(got)}")
        rows += got
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            r["label"] = 0
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} human samples -> {OUT}")


if __name__ == "__main__":
    main()
