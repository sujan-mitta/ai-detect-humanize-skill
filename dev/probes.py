#!/usr/bin/env python3
"""Sensitivity study: which mechanical edits move the max-mode score of AI text at all?
Writes probe variants of the dev originals to corpus/humbench/dev/p_<name>/ (not fact-safe; analysis only)."""
import re, sys
from pathlib import Path
B = Path(__file__).resolve().parent / "corpus" / "humbench" / "dev"
STOCK = re.compile(r"\b(significant|crucial|critical|testament|landscape|pivotal|vital|delve|comprehensive|robust|seamless|holistic|"
                   r"foster|enhance|leverage|underscore|navigate|realm|tapestry|journey|ultimately|moreover|furthermore|additionally|"
                   r"inclusive|essential|invaluable|rich|vibrant|myriad|plethora)\w*", re.I)
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“])")

def strip_md(t):
    t = re.sub(r"^#{1,6}\s*", "", t, flags=re.M)
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t); t = re.sub(r"(?m)^\s*[-*•]\s+", "", t)
    return re.sub(r"(?m)^\s*\d+\.\s+", "", t)
def ascii_punct(t):
    return t.replace("—", ", ").replace("–", "-").replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
def drop_stock_sentences(t):
    out = []
    for para in t.split("\n\n"):
        ss = SENT.split(para)
        keep = [s for s in ss if not STOCK.search(s)] if len(ss) > 2 else ss
        out.append(" ".join(keep) if keep else para)
    return "\n\n".join(out)
def drop_first_last_para(t):
    ps = [p for p in t.split("\n\n") if p.strip()]
    return "\n\n".join(ps[1:-1]) if len(ps) > 4 else t
def split_long(t):
    def f(s):
        if len(s.split()) > 22:
            m = re.search(r",\s+(and|which|but|while|so)\s+", s)
            if m: return s[:m.start()] + ". " + s[m.end()].upper() + s[m.end() + 1:]
        return s
    return "\n\n".join(" ".join(f(s) for s in SENT.split(p)) for p in t.split("\n\n"))
def contract(t):
    for a, b in [(" do not ", " don't "), (" is not ", " isn't "), (" it is ", " it's "), (" does not ", " doesn't "), (" cannot ", " can't "), (" are not ", " aren't "), (" will not ", " won't "), (" that is ", " that's ")]:
        t = t.replace(a, b)
    return t
def shuffle_paras(t):
    ps = t.split("\n\n"); 
    import random; random.Random(1).shuffle(ps); return "\n\n".join(ps)
def lowercase_starts_no(t): return t
PROBES = {"nomd": strip_md, "ascii": ascii_punct, "nostock": drop_stock_sentences, "trim": drop_first_last_para,
          "split": split_long, "contract": contract, "shuffle": shuffle_paras,
          "combo": lambda t: contract(split_long(drop_stock_sentences(ascii_punct(strip_md(t)))))}
for name, fn in PROBES.items():
    d = B / f"p_{name}"; d.mkdir(exist_ok=True)
    for f in sorted((B / "orig").glob("*.txt")):
        (d / f.name).write_text(fn(f.read_text(encoding="utf-8")), encoding="utf-8")
print("probes written:", ", ".join(PROBES))
