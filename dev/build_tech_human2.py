#!/usr/bin/env python3
"""More human technical prose: CPython 3.9 library reference / FAQ / language reference
(API-documentation style, written 2020 and earlier). Reads a sparse checkout (path argument),
appends 'technical' rows to dev/corpus/human.jsonl, one source per file, and writes topic
seeds for matched AI passages to dev/corpus/ai_new/topics_tech2.json."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_tech_human import prose_blocks, OUT, ROOT  # noqa: E402

random.seed(33)


def main(base):
    rows = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if not r["source"].startswith("techdoc2:")]
    new, topics = [], []
    for sub, tag, cap in (("Doc/library", "cpython-lib", 100), ("Doc/faq", "cpython-faq", 14), ("Doc/reference", "cpython-ref", 14)):
        files = sorted((Path(base) / sub).glob("*.rst"))
        random.shuffle(files)
        got = 0
        for f in files:
            if got >= cap:
                break
            paras = prose_blocks(f.read_text(encoding="utf-8", errors="replace"))
            if len(paras) < 3:
                continue
            k = random.randint(0, max(0, len(paras) - 3))
            chunk, n = [], 0
            for p in paras[k:]:
                chunk.append(p)
                n += len(p.split())
                if n >= 190:
                    break
            text = "\n\n".join(chunk)
            if 160 <= len(text.split()) <= 480:
                new.append({"text": text, "source": f"techdoc2:{tag}:{f.name}", "genre": "technical", "label": 0})
                topics.append(f"Python {sub.split('/')[-1]} documentation: {f.stem.replace('-', ' ').replace('_', ' ')}")
                got += 1
        print(tag, got)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows + new) + "\n", encoding="utf-8")
    (ROOT / "corpus" / "ai_new" / "topics_tech2.json").write_text(json.dumps(topics, indent=1), encoding="utf-8")
    print(len(new), "added")


if __name__ == "__main__":
    main(sys.argv[1])
