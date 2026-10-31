#!/usr/bin/env python3
"""Benchmark the humanizing workflow on AI samples rewritten with the skill.

Usage: python dev/humanize_bench.py [--no-deep]

Pairs each dev/corpus/humanized/<model>__<file>.txt with its original in
dev/corpus/ai/<model>/<file>.txt and reports, per genre and overall:
  - AI-likeness before -> after (style model, and style+GPT-2 unless --no-deep)
  - share now reading as human (score < 40) and as AI (score >= 60)
  - verify_rewrite pass rate (facts kept, nothing invented) and length ratio
  - fake-casual tics introduced by the rewrite (ai-patterns.md section 9)
It also scores dev/corpus/ai_edited/ (human text polished by AI), which a
fair detector should mostly put in the middle.

The humanizers used the style scorer while revising, so the style-model drop
is partly "taught to the test"; the GPT-2 score is the more honest check.
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "humanize" / "scripts"))
import ai_score  # noqa: E402
import verify_rewrite  # noqa: E402


def score(text, deep):
    r = ai_score.build(text, deep=deep)
    return r["score"], r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-deep", action="store_true")
    args = ap.parse_args()
    deep = not args.no_deep

    rows = []
    for f in sorted((ROOT / "corpus" / "humanized").glob("*.txt")):
        model, name = f.stem.split("__", 1)
        orig_path = ROOT / "corpus" / "ai" / model / f"{name}.txt"
        if not orig_path.exists():
            print("no original for", f.name)
            continue
        orig, new = orig_path.read_text(encoding="utf-8"), f.read_text(encoding="utf-8")
        s0, r0 = score(orig, False)
        s1, r1 = score(new, False)
        row = {"file": f.name, "genre": name.split("__")[0], "style_before": s0, "style_after": s1}
        if deep:
            row["deep_before"], _ = score(orig, True)
            row["deep_after"], _ = score(new, True)
        v = verify_rewrite.compare(orig, new)
        row.update(verify=v["passed"], length_ratio=v["shape"]["length_ratio"],
                   tics_before=r0["raw"]["performed_voice_tics"], tics_after=r1["raw"]["performed_voice_tics"],
                   frag_after=r1["raw"]["fragment_ratio"], missing=v["missing"], added=v["added"])
        rows.append(row)
        print(f"{f.name:42s} style {s0:3d}->{s1:3d}" + (f"  deep {row['deep_before']:3d}->{row['deep_after']:3d}" if deep else "")
              + f"  verify {'ok' if v['passed'] else 'FAIL'}  len {row['length_ratio']:.2f}", flush=True)

    if not rows:
        sys.exit("no humanized samples found")
    keys = ["style"] + (["deep"] if deep else [])

    def summary(rs):
        out = {"n": len(rs)}
        for k in keys:
            b = [r[f"{k}_before"] for r in rs]
            a = [r[f"{k}_after"] for r in rs]
            out[k] = {"before": round(statistics.mean(b), 1), "after": round(statistics.mean(a), 1),
                      "now_human": round(sum(x < 40 for x in a) / len(a), 3),
                      "still_ai": round(sum(x >= 60 for x in a) / len(a), 3)}
        out["verify_pass"] = round(sum(r["verify"] for r in rs) / len(rs), 3)
        out["length_ratio"] = round(statistics.mean(r["length_ratio"] for r in rs), 3)
        out["tics_added"] = sum(max(0, r["tics_after"] - r["tics_before"]) for r in rs)
        return out

    report = {"overall": summary(rows), "by_genre": {}}
    by = defaultdict(list)
    for r in rows:
        by[r["genre"]].append(r)
    for g, rs in sorted(by.items()):
        report["by_genre"][g] = summary(rs)

    print("\n## Humanizing benchmark")
    o = report["overall"]
    for k in keys:
        print(f"{k:5s}: mean score {o[k]['before']} -> {o[k]['after']} | now reads human {o[k]['now_human']:.0%} | still AI {o[k]['still_ai']:.0%}")
    print(f"facts check passed {o['verify_pass']:.0%} | length ratio {o['length_ratio']:.2f} | fake-casual tics added {o['tics_added']}")
    for g, s in report["by_genre"].items():
        line = f"  {g:13s} n={s['n']:2d} " + " ".join(f"{k} {s[k]['before']:.0f}->{s[k]['after']:.0f}" for k in keys)
        print(line + f"  verify {s['verify_pass']:.0%}")
    fails = [r for r in rows if not r["verify"]]
    if fails:
        print("\nFacts-check failures:")
        for r in fails:
            print(f"  {r['file']}: missing {r['missing']} added {r['added']}")

    edited = sorted((ROOT / "corpus" / "ai_edited").glob("edited__*.txt"))
    if edited:
        sc = [score(f.read_text(encoding="utf-8"), deep)[0] for f in edited]
        report["ai_edited"] = {"n": len(sc), "mean": round(statistics.mean(sc), 1),
                               "flagged_ai": round(sum(s >= 60 for s in sc) / len(sc), 3),
                               "mixed": round(sum(40 <= s < 60 for s in sc) / len(sc), 3)}
        e = report["ai_edited"]
        print(f"\n## AI-polished human text (n={e['n']}): mean score {e['mean']}, "
              f"flagged AI {e['flagged_ai']:.0%}, mixed {e['mixed']:.0%}")

    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs" / "humanize_bench.json").write_text(json.dumps({"report": report, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
