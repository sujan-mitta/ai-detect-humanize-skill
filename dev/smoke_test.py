#!/usr/bin/env python3
"""Fast regression check for the shipped scripts (style tier, no model downloads).

Runs edge-case inputs through ai_score.py and the helper scripts and fails loudly if
any of them crashes or gives a wrong verdict class. Usage: python dev/smoke_test.py"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "ai-detect-humanize" / "scripts"
AI = "Overall, it is important to note that this comprehensive approach plays a crucial role. " * 12
FAILS = []


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPTS / args[0]), *args[1:]], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def score(tmp, name, data, **kw):
    p = tmp / name
    if isinstance(data, bytes):
        p.write_bytes(data)
    else:
        p.write_text(data, encoding="utf-8", **kw)
    out = tmp / (name + ".json")
    r = run("ai_score.py", str(p), "--json", str(out))
    return r, (json.loads(out.read_text(encoding="utf-8")) if out.exists() else None)


def main():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for name, data in [("empty.txt", ""), ("blank.txt", "  \n\n"), ("one.txt", "Hello world.")]:
            r, o = score(tmp, name, data)
            check(f"{name}: abstains", r.returncode == 0 and o and o.get("abstained"), r.stderr[-200:])
        r, o = score(tmp, "ai.txt", AI)
        check("stock AI text flagged", r.returncode == 0 and o and o["score"] >= 85, str(o and o["score"]))
        r, o = score(tmp, "bom.txt", b"\xef\xbb\xbf" + AI.encode("utf-8"))
        check("BOM file scores like plain", o and o["score"] >= 85)
        cp = ("Café “quotes” and naïve ideas, " * 30).encode("cp1252")
        r, o = score(tmp, "cp1252.txt", cp)
        check("Windows-1252 file reads", r.returncode == 0 and o is not None, r.stderr[-200:])
        r = run("ai_score.py", str(tmp / "nope.txt"))
        check("missing file: clean message", r.returncode != 0 and "Traceback" not in r.stderr and "not found" in r.stderr, r.stderr[-200:])
        (tmp / "x.doc").write_text("x")
        r = run("ai_score.py", str(tmp / "x.doc"))
        check(".doc: clean message", r.returncode != 0 and "Traceback" not in r.stderr, r.stderr[-200:])
        (tmp / "a.txt").write_text(AI, encoding="utf-8")
        (tmp / "b.txt").write_text(AI.replace("crucial", "central"), encoding="utf-8")
        r = run("verify_rewrite.py", str(tmp / "a.txt"), str(tmp / "b.txt"))
        check("verify_rewrite runs", r.returncode == 0, r.stderr[-200:])
        r = run("rewrite_targets.py", str(tmp / "a.txt"))
        check("rewrite_targets runs", r.returncode == 0 and r.stdout.strip(), r.stderr[-200:])
        r = run("chunk_text.py", str(tmp / "a.txt"), "--out", str(tmp / "chunks"))
        check("chunk_text runs", r.returncode == 0 and (tmp / "chunks" / "manifest.json").exists(), r.stderr[-200:])
        r = run("author_questions.py", str(tmp / "a.txt"))
        check("author_questions runs", r.returncode == 0, r.stderr[-200:])
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
