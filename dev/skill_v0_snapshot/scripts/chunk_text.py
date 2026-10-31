#!/usr/bin/env python3
"""Split a long text into ~N-word chunks at paragraph boundaries.

Usage: python chunk_text.py INPUT --out DIR [--words 900]

Writes DIR/chunk_001.txt, chunk_002.txt, ... and DIR/manifest.json.
A heading line is never left as the last thing in a chunk; it moves to the
next chunk with the content it introduces. Fenced code blocks are kept whole.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extract_text import extract  # noqa: E402


def blocks_of(text):
    """Paragraph blocks, keeping fenced code blocks intact."""
    blocks, buf, in_code = [], [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        if not line.strip() and not in_code:
            if buf:
                blocks.append("\n".join(buf))
                buf = []
        else:
            buf.append(line)
    if buf:
        blocks.append("\n".join(buf))
    return blocks


def is_heading(block):
    return bool(re.match(r"^#{1,6}\s", block)) and "\n" not in block


def chunk(blocks, target):
    chunks, cur, count = [], [], 0
    for block in blocks:
        n = len(block.split())
        if cur and count + n > target * 1.25:
            carry = []
            while cur and is_heading(cur[-1]):
                carry.insert(0, cur.pop())
            if cur:
                chunks.append(cur)
            cur = carry
            count = sum(len(b.split()) for b in cur)
        cur.append(block)
        count += n
        if count >= target and not is_heading(block):
            chunks.append(cur)
            cur, count = [], 0
    if cur:
        chunks.append(cur)
    return chunks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("--out", required=True)
    ap.add_argument("--words", type=int, default=900)
    args = ap.parse_args()

    text = extract(args.input)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("chunk_*.txt"):
        old.unlink()

    manifest = {"source": str(args.input), "target_words": args.words, "chunks": []}
    for i, blocks in enumerate(chunk(blocks_of(text), args.words), 1):
        name = f"chunk_{i:03d}.txt"
        body = "\n\n".join(blocks)
        (out / name).write_text(body, encoding="utf-8")
        first = next((b for b in blocks if is_heading(b)), "")
        manifest["chunks"].append({
            "file": name,
            "humanized_file": name.replace(".txt", ".human.txt"),
            "words": len(body.split()),
            "first_heading": first.lstrip("# ").strip(),
        })
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total = sum(c["words"] for c in manifest["chunks"])
    print(f"{len(manifest['chunks'])} chunks, {total} words total -> {out}")
    for c in manifest["chunks"]:
        print(f"  {c['file']}  {c['words']:>5} words  {c['first_heading'][:50]}")


if __name__ == "__main__":
    main()
