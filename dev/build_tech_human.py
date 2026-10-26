#!/usr/bin/env python3
"""Human technical *explanatory* prose (tutorials, how-tos, guides) - the kind of
technical writing people actually submit. Sources, all written before 2021:
  CPython 3.9 docs (Doc/tutorial, Doc/howto), Linux 5.4 kernel Documentation (admin-guide,
  process, filesystems), Rust book 2018 edition, Pro Git 2 (2014/2018).
Appends to dev/corpus/human.jsonl (genre 'technical', one source per file so groups never
straddle a CV split) and writes topic seeds to dev/corpus/ai_new/topics_tech.json for
matched AI passages.
"""
import json
import random
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "corpus" / "human.jsonl"
random.seed(21)


def get(url):
    return subprocess.run(["curl", "-sSL", "--max-time", "60", url], capture_output=True).stdout.decode("utf-8", "replace")


def listing(repo, path, ref):
    d = json.loads(get(f"https://api.github.com/repos/{repo}/contents/{path}?ref={ref}") or "[]")
    return [x["name"] for x in d if isinstance(x, dict) and x["name"].endswith((".rst", ".md", ".asc"))]


SOURCES = [
    ("cpython", "python/cpython", "Doc/howto", "v3.9.0", 14),
    ("cpython-tut", "python/cpython", "Doc/tutorial", "v3.9.0", 12),
    ("kernel-admin", "torvalds/linux", "Documentation/admin-guide", "v5.4", 22),
    ("kernel-process", "torvalds/linux", "Documentation/process", "v5.4", 12),
    ("rustbook", "rust-lang/book", "src", "nostarch-2019-01-03", 22),
    ("progit", "progit/progit2", "book/01-introduction/sections", "main", 6),
    ("progit2", "progit/progit2", "book/02-git-basics/sections", "main", 8),
    ("progit3", "progit/progit2", "book/03-git-branching/sections", "main", 8),
]


def prose_blocks(raw):
    raw = re.sub(r"```.*?```", "", raw, flags=re.S)
    raw = re.sub(r"\.\. code-block::.*?(?=\n\S)", "", raw, flags=re.S)
    out = []
    for blk in re.split(r"\n\s*\n", raw):
        if blk.startswith((" ", "\t", ">", "|", "=", "-", "*", "#", ".. ", "::", "+", "[[", "image:", "----")) or re.match(r"^\s*(\d+\.|[-*]) ", blk):
            continue
        flat = " ".join(x.strip() for x in blk.split("\n"))
        flat = re.sub(r":\w+:`~?([^`<]*?)(?: <[^`]*>)?`", r"\1", flat)
        flat = re.sub(r"`([^`]*)`_?", r"\1", flat)
        flat = re.sub(r"\*\*?([^*]+)\*\*?", r"\1", flat)
        flat = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", flat)
        flat = re.sub(r"(?:link|xref):[^\[]*\[([^\]]*)\]", r"\1", flat).replace("``", "")
        letters = sum(c.isalpha() or c == " " for c in flat) / max(1, len(flat))
        if len(flat.split()) >= 35 and letters > 0.9 and flat.count(". ") >= 1 and "::" not in flat and "{" not in flat:
            out.append(flat)
    return out


def main():
    rows = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if not r["source"].startswith("techdoc:")]
    new, topics = [], []
    for tag, repo, path, ref, cap in SOURCES:
        names = listing(repo, path, ref)
        random.shuffle(names)
        got = 0
        for name in names:
            if got >= cap:
                break
            raw = get(f"https://raw.githubusercontent.com/{repo}/{ref}/{path}/{name}")
            paras = prose_blocks(raw)
            if len(paras) < 4:
                continue
            k = random.randint(0, max(0, len(paras) - 4))
            chunk, n = [], 0
            for p in paras[k:]:
                chunk.append(p)
                n += len(p.split())
                if n >= 200:
                    break
            text = "\n\n".join(chunk)
            if 170 <= len(text.split()) <= 480:
                new.append({"text": text, "source": f"techdoc:{tag}:{name}", "genre": "technical", "label": 0})
                topics.append(f"{repo.split('/')[-1]} documentation: {name.rsplit('.', 1)[0].replace('-', ' ').replace('_', ' ')} ({tag})")
                got += 1
        print(tag, got, flush=True)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows + new) + "\n", encoding="utf-8")
    (ROOT / "corpus" / "ai_new" / "topics_tech.json").write_text(json.dumps(topics, indent=1), encoding="utf-8")
    print(len(new), "added;", new[0]["text"][:400])


if __name__ == "__main__":
    main()
