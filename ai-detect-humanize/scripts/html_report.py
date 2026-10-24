#!/usr/bin/env python3
"""Interactive HTML heatmap of how AI-like each sentence reads.

Usage:
  python html_report.py INPUT -o report.html [--deep] [--compare BEFORE_INPUT]

Each sentence is highlighted by its AI-likeness (pattern matches, plus GPT-2
predictability with --deep); hover or tap a sentence to see why. The header
shows the overall calibrated score, what pushed it toward AI or human, and
per-section scores. With --compare, the page shows the original and the
rewritten text side by side with both scores.

ai_score.py --html calls this too. The output is one self-contained file.
"""
import argparse
import html
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ai_score  # noqa: E402
from extract_text import extract  # noqa: E402

SPLIT_RE = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[*]?[A-Z0-9])")


def sentence_ppl(sentence):
    import perplexity
    _, lp, _, _ = perplexity.token_stats(sentence)
    return math.exp(-sum(lp) / len(lp)) if lp else None


def section_scores(text):
    """Model score (0-1) for each paragraph, taken from the section it falls in.
    Sentence colours blend this with local pattern hits, so one stray 'however'
    in a human section doesn't light up and an AI section reads as a block."""
    by_para = {}
    for _, body in ai_score.sections(text):
        s = ai_score.analyze(body)["score"] / 100
        for para in re.split(r"\n\s*\n", body):
            by_para[" ".join(para.split())] = s
    return by_para


def score_sentences(text, deep):
    """[(kind, payload)] blocks; paragraphs carry [(sentence, score, reasons)]."""
    out = []
    sec = section_scores(text)
    for block in re.split(r"\n\s*\n", text):
        b = block.strip()
        if not b:
            continue
        m = re.match(r"^(#{1,6})\s+(.*)", b)
        if m and "\n" not in b:
            out.append(("heading", (len(m.group(1)), m.group(2))))
            continue
        if b.startswith("|") or b.startswith("```"):
            out.append(("raw", b))
            continue
        items = []
        base = sec.get(" ".join(b.split()))
        for line in b.splitlines() if re.search(r"^\s*([-*•]|\d+[.)])\s", b, re.M) else [" ".join(b.split())]:
            for s in SPLIT_RE.split(line.strip()):
                if not s.strip():
                    continue
                weight, reasons = ai_score.sentence_detail(s)
                score = min(1.0, weight / 5.0)
                if base is not None:
                    score = 0.6 * base + 0.4 * score
                    if base >= 0.6:
                        reasons = [f"section reads as AI ({base:.0%})"] + reasons
                if deep and len(s.split()) >= 5:
                    ppl = sentence_ppl(s)
                    if ppl:
                        p = max(0.0, min(1.0, (70 - ppl) / 55))
                        score = 0.5 * score + 0.5 * p
                        if p >= 0.6:
                            reasons = reasons + [f"very predictable to GPT-2 (perplexity {ppl:.0f})"]
                        elif p <= 0.1:
                            reasons = reasons + [f"unpredictable word choices (perplexity {ppl:.0f})"]
                items.append((s, round(score, 3), reasons))
            items.append(("\n", None, None))
        out.append(("para", items[:-1]))
    return out


def level(score):
    if score is None or score < 0.2:
        return 0
    if score < 0.45:
        return 1
    if score < 0.7:
        return 2
    return 3


def render_doc(blocks):
    parts = []
    for kind, payload in blocks:
        if kind == "heading":
            lvl, txt = payload
            parts.append(f"<h{min(lvl + 1, 4)} class='doc-h'>{html.escape(txt)}</h{min(lvl + 1, 4)}>")
        elif kind == "raw":
            parts.append(f"<pre>{html.escape(payload)}</pre>")
        else:
            spans = []
            for s, score, reasons in payload:
                if s == "\n":
                    spans.append("<br>")
                    continue
                tip = html.escape("; ".join(reasons) if reasons else "no AI patterns found")
                spans.append(f"<span class='s l{level(score)}' tabindex='0' data-score='{score}' "
                             f"data-tip='{tip}'>{html.escape(s)}</span> ")
            parts.append("<p>" + "".join(spans) + "</p>")
    return "\n".join(parts)


def summary_html(result, label):
    cal = result.get("calibrated") or {}
    chips_ai = "".join(f"<li class='chip ai'>{html.escape(n)}</li>" for n, _ in cal.get("toward_ai", []))
    chips_hu = "".join(f"<li class='chip hu'>{html.escape(n)}</li>" for n, _ in cal.get("toward_human", []))
    secs = ""
    if len(result.get("sections", [])) > 1:
        rows = "".join(
            f"<tr><td>{html.escape(s['title'])}</td><td class='num'>{s['words']}</td>"
            f"<td><div class='bar'><i style='width:{s['score']}%' class='b{level(s['score'] / 100)}'></i></div></td>"
            f"<td class='num'>{s['score']}</td></tr>" for s in result["sections"])
        secs = f"<details><summary>Sections</summary><table><tr><th>Section</th><th>Words</th><th></th><th>Score</th></tr>{rows}</table></details>"
    deep = result.get("deep")
    deep_line = (f"<p class='muted'>GPT-2 perplexity {deep['perplexity']} &middot; "
                 f"{deep['top10_fraction']:.0%} of words were in its top-10 guesses</p>") if deep else ""
    return f"""
    <section class="card">
      <div class="label">{html.escape(label)}</div>
      <div class="gauge"><span class="big">{result['score']}</span><span class="of">/100</span>
        <span class="verdict v{2 if result.get('mixed') else level(result['score'] / 100)}">{html.escape(result['verdict'])}</span></div>
      <p class="muted">{result['raw']['words']} words &middot; {result['raw']['sentences']} sentences
        {' &middot; short text, low confidence' if result.get('low_confidence') else ''}</p>
      {deep_line}
      {f'<h3>Pushing toward AI</h3><ul class="chips">{chips_ai}</ul>' if chips_ai else ''}
      {f'<h3>Pushing toward human</h3><ul class="chips">{chips_hu}</ul>' if chips_hu else ''}
      {secs}
    </section>"""


CSS = """
:root{--bg:#faf9f7;--fg:#1f1e1c;--muted:#6b6862;--card:#fff;--line:#e6e3dd;
--l1:#fdf0c4;--l2:#fbd38d;--l3:#f6a38a;--ai:#b4472d;--hu:#2f6f4f;--accent:#7a5af8}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#181715;--fg:#ecebe8;--muted:#a19e97;
--card:#22211e;--line:#3a3833;--l1:#4a3f18;--l2:#6b4a17;--l3:#7a2f22;--ai:#f08f74;--hu:#7fc8a0;--accent:#a996ff}}
:root[data-theme="dark"]{--bg:#181715;--fg:#ecebe8;--muted:#a19e97;--card:#22211e;--line:#3a3833;
--l1:#4a3f18;--l2:#6b4a17;--l3:#7a2f22;--ai:#f08f74;--hu:#7fc8a0;--accent:#a996ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.65 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:24px 16px 8px;max-width:1200px;margin:auto}h1{font-size:22px;margin:0 0 4px}
.muted{color:var(--muted);font-size:14px;margin:4px 0}
main{max-width:1200px;margin:auto;padding:0 16px 48px}
.grid{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.label{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.gauge{display:flex;align-items:baseline;gap:6px;flex-wrap:wrap}.big{font-size:44px;font-weight:700}
.of{color:var(--muted)}.verdict{margin-left:8px;padding:2px 10px;border-radius:99px;font-size:14px;font-weight:600}
.v0,.v1{background:color-mix(in srgb,var(--hu) 18%,transparent);color:var(--hu)}
.v2{background:var(--l2);color:var(--fg)}.v3{background:color-mix(in srgb,var(--ai) 20%,transparent);color:var(--ai)}
h3{font-size:13px;margin:12px 0 6px;color:var(--muted);font-weight:600}
.chips{list-style:none;padding:0;margin:0;display:flex;flex-wrap:wrap;gap:6px}
.chip{font-size:13px;padding:2px 9px;border-radius:99px;border:1px solid var(--line)}
.chip.ai{color:var(--ai)}.chip.hu{color:var(--hu)}
details{margin-top:12px}summary{cursor:pointer;font-size:14px;color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:14px;margin-top:8px}td,th{padding:4px 6px;text-align:left;border-bottom:1px solid var(--line)}
.num{text-align:right;font-variant-numeric:tabular-nums}.bar{background:var(--line);border-radius:4px;height:8px;min-width:60px}
.bar i{display:block;height:8px;border-radius:4px}.b0,.b1{background:var(--hu)}.b2{background:#e0a030}.b3{background:var(--ai)}
.toolbar{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin:20px 0 8px;font-size:14px}
.legend{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.legend span{padding:1px 8px;border-radius:4px}
.doc{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:8px 22px;overflow-wrap:anywhere}
.doc-h{margin:18px 0 6px}pre{white-space:pre-wrap;font-size:13px}
.s{border-radius:3px;padding:1px 0;cursor:help;outline:none}.l1{background:var(--l1)}.l2{background:var(--l2)}.l3{background:var(--l3)}
.dim .s{background:none}.dim .s.show{background:var(--l3)}
#tip{position:fixed;z-index:9;max-width:320px;background:var(--fg);color:var(--bg);font-size:13px;line-height:1.4;
padding:8px 10px;border-radius:8px;pointer-events:none;opacity:0;transition:opacity .1s}
.cols{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
.coltitle{font-weight:600;margin:16px 0 6px}
"""

JS = """
const tip=document.getElementById('tip');
function show(e){const t=e.target.closest('.s');if(!t)return;tip.textContent=
 'AI-likeness '+Math.round(t.dataset.score*100)+'%: '+t.dataset.tip;tip.style.opacity=1;
 const r=t.getBoundingClientRect();let x=Math.min(r.left,innerWidth-340);tip.style.left=Math.max(8,x)+'px';
 tip.style.top=(r.bottom+8+tip.offsetHeight>innerHeight?r.top-tip.offsetHeight-8:r.bottom+8)+'px';}
document.addEventListener('mouseover',show);document.addEventListener('focusin',show);
document.addEventListener('mouseout',e=>{if(e.target.closest('.s'))tip.style.opacity=0});
const sl=document.getElementById('thr'),out=document.getElementById('thrv');
sl&&sl.addEventListener('input',()=>{const v=sl.value/100;out.textContent=sl.value+'%';
 document.querySelectorAll('.doc').forEach(d=>d.classList.toggle('dim',v>0));
 document.querySelectorAll('.s').forEach(s=>s.classList.toggle('show',+s.dataset.score>=v&&v>0));});
"""


def page(title, cards, docs):
    toolbar = """<div class="toolbar"><div class="legend">AI-likeness:
      <span class="l1">some</span><span class="l2">notable</span><span class="l3">strong</span></div>
      <label>Only highlight sentences above <input id="thr" type="range" min="0" max="90" value="0" step="10">
      <output id="thrv">0%</output></label></div>"""
    if len(docs) == 1:
        body = f"<div class='doc'>{docs[0][1]}</div>"
    else:
        body = "<div class='cols'>" + "".join(
            f"<div><div class='coltitle'>{html.escape(t)}</div><div class='doc'>{d}</div></div>" for t, d in docs) + "</div>"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>AI-Likeness Heatmap</title>
<style>{CSS}</style></head><body><header><h1>{html.escape(title)}</h1>
<p class="muted">Statistical estimate from writing style &mdash; not proof of authorship. Hover or tap a sentence to see why it's highlighted.</p></header>
<main><div class="grid">{cards}</div>{toolbar}{body}</main><div id="tip" role="tooltip"></div><script>{JS}</script></body></html>"""


def build_report(path, deep=False, compare=None):
    text = extract(path)
    result = ai_score.build(text, deep=deep)
    if compare:
        before_text = extract(compare)
        before = ai_score.build(before_text, deep=deep)
        cards = summary_html(before, f"Before: {Path(compare).name}") + summary_html(result, f"After: {Path(path).name}")
        docs = [("Before", render_doc(score_sentences(before_text, deep))),
                ("After", render_doc(score_sentences(text, deep)))]
        title = f"AI-likeness: {before['score']} → {result['score']}"
    else:
        cards = summary_html(result, Path(path).name)
        docs = [("", render_doc(score_sentences(text, deep)))]
        title = f"AI-likeness report: {Path(path).name}"
    return page(title, cards, docs), result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--compare", help="the original file, to show before/after side by side")
    args = ap.parse_args()
    if args.deep:
        import perplexity
        if not perplexity.available():
            print("NOTE: --deep needs `pip install torch transformers`; continuing without it.", file=sys.stderr)
            args.deep = False
    out, result = build_report(args.input, args.deep, args.compare)
    Path(args.output).write_text(out, encoding="utf-8")
    print(f"Wrote {args.output} (score {result['score']}, {result['verdict']})")


if __name__ == "__main__":
    main()
