#!/usr/bin/env python3
"""One-off layout repair for the human email/post/review/report rows (idempotent).

- email: Enron bodies are hard-wrapped at ~70 columns with single newlines, so the whole
  email reads as ONE paragraph. Reflow (join wrapped lines, restore blank-line paragraphs).
- post: Reddit uses single newlines between paragraphs -> make them paragraph breaks.
- drop rows that are mostly symbols/ASCII art (e.g. '= = = =' rules in emails).
- all: collapse runs of spaces/tabs (canon_typography handles this at scoring time too).
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "ai-detect-humanize" / "scripts"))
from extract_text import reflow  # noqa: E402

PATH = ROOT / "corpus" / "human.jsonl"


def letters_share(t):
    return sum(ch.isalpha() for ch in t) / max(1, len(t))


def main():
    rows = [json.loads(l) for l in PATH.read_text(encoding="utf-8").splitlines()]
    out, dropped, changed = [], 0, 0
    for r in rows:
        t = r["text"]
        if r["genre"] == "email":
            t = reflow(t)
        elif r["genre"] == "post":
            t = re.sub(r"\n+", "\n\n", t.strip())
        if r["genre"] in {"email", "post", "review", "report"}:
            t = re.sub(r"[ \t]{2,}", " ", t).strip()
            if letters_share(t) < 0.72 or len(t.split()) < 120:
                dropped += 1
                continue
        if t != r["text"]:
            changed += 1
        r["text"] = t
        out.append(r)
    PATH.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
    for g in ("email", "report", "review", "post"):
        sub = [r["text"] for r in out if r["genre"] == g]
        print(f"{g:7s} n={len(sub):3d}  multi-paragraph {sum(t.count(chr(10) * 2) >= 1 for t in sub):3d}  "
              f"median words {sorted(len(t.split()) for t in sub)[len(sub) // 2]}")
    print(f"changed {changed}, dropped {dropped}")


if __name__ == "__main__":
    main()
