---
name: ai-detect-humanize
description: Detect AI-generated writing and rewrite ("humanize") it into natural, human-sounding prose — for anything from one paragraph to huge documents, PDFs, Word files, essays, reports, theses, blog posts and books. Use this skill whenever the user asks to check whether text is AI-written, get an "AI score" or "AI percentage", find robotic or ChatGPT-sounding passages, humanize, de-AI, rewrite to sound more human/natural, make text "undetectable", or improve the flow of AI-drafted writing — even if they just paste text and say "make this sound like me" or "does this look like ChatGPT wrote it?". Handles long inputs by extracting, chunking and processing section by section, and produces a before/after score report.
---

# AI Detect & Humanize

Two jobs, often done together:

1. **Detect** — estimate how AI-generated a text reads, and point at the exact passages that give it away.
2. **Humanize** — rewrite those passages (or the whole text) so they read like a thoughtful person wrote them, keeping the meaning, facts and format intact.

The scripts in `scripts/` do the mechanical parts (reading PDFs/DOCX, splitting huge texts, scoring, rebuilding the output file). You do the judgment and the writing.

## Be honest about what detection can do

No detector — commercial or this one — can prove who wrote a text. The script measures *stylistic signals* that AI text tends to have (uniform sentence length, stock phrases, heavy transitions, few contractions). Report results as "reads as likely AI / mixed / likely human", give the score, and say it's a heuristic. Never tell a user a text is definitely AI-written or definitely human; people get accused unfairly on that basis.

Humanizing is about good writing: varied rhythm, concrete wording, a real voice. If a user says the goal is to submit AI work as their own where that's prohibited (e.g. a graded assignment), it's fine to mention once, briefly, that they should check their institution's rules — then help with the writing. Don't lecture or refuse.

## Step 0 — Get the text

| Input | Do this |
|---|---|
| Pasted text (short) | Use it directly. Save to a `.txt` file if you'll run the scripts. |
| `.pdf`, `.docx`, `.txt`, `.md` | `python scripts/extract_text.py <file> -o work/source.txt` |
| Scanned PDF (extract gives almost no text) | Tell the user it needs OCR; use the `pdf` skill if available. |

Keep a `work/` folder next to the user's file (or in a scratch dir) for intermediate files.

## Step 1 — Detect

```bash
python scripts/ai_score.py work/source.txt --report work/before_report.md --json work/before.json
```

This prints an overall score (0–100, higher = more AI-like), a per-section breakdown, the signal values, and the most suspicious sentences. Then add your own read: skim the flagged sections and note what a human editor would notice that a script can't (generic claims with no specifics, symmetrical "On one hand… on the other" structure, hollow conclusions, a summary that repeats the intro).

If the user only asked for detection, stop here and present:
- Overall verdict + score, with the caveat above (one sentence, not a paragraph)
- Top 3–5 signals that drove the score, in plain language
- The most AI-sounding passages, quoted, with why each one reads that way
- Offer to humanize

## Step 2 — Chunk (long texts only)

Anything over ~1,500 words: split it so each piece gets full attention and nothing gets silently dropped or summarized.

```bash
python scripts/chunk_text.py work/source.txt --out work/chunks --words 900
```

This writes `work/chunks/chunk_001.txt …` split at paragraph boundaries, plus `manifest.json`. Headings, lists and code blocks stay attached to their chunk.

For very long documents (say, 15+ chunks), it's reasonable to use subagents in parallel if you have them — give each one a batch of chunks, the voice notes from Step 3, and the guide in `references/humanizing-guide.md`. Otherwise just work through the chunks in order.

## Step 3 — Humanize

Read `references/humanizing-guide.md` before rewriting — it lists the AI patterns to remove and the techniques that actually make prose read as human.

First, settle the **voice** in 2–3 lines and keep it fixed across all chunks so the document doesn't change personality halfway through:
- Audience and register (academic, business, casual blog, personal statement…)
- Person (I / we / impersonal) — keep whatever the original used unless asked
- If the user gave a writing sample of their own, match its sentence length, vocabulary and quirks

Then rewrite each chunk to `work/chunks/chunk_001.human.txt` etc. Ground rules, because these are where rewrites usually go wrong:

- **Preserve meaning and every fact** — numbers, names, dates, citations, quotes, technical terms. Humanizing that changes what the text says is a failure.
- **Preserve structure** — headings, list items, tables, paragraph order. Markdown/headings in, same headings out.
- **Don't shorten.** Aim for roughly the same length (±15%). Cutting content to sound punchier loses the user's material.
- **Don't overcorrect** into slang, typos or forced jokes. The target is a competent human writer, not a sloppy one.
- **Stay consistent at chunk seams** — read the last paragraph of the previous rewritten chunk before starting the next.

Short text (one or a few paragraphs): skip chunking and just rewrite, then score.

## Step 4 — Rebuild and re-score

```bash
python scripts/build_output.py work/chunks --format docx -o "<original-name>_humanized.docx"
python scripts/ai_score.py "<original-name>_humanized.docx" --report work/after_report.md --compare work/before.json
```

Output format follows the input: `.pdf`/`.docx` → `.docx`; `.md` → `.md`; `.txt` → `.txt`; pasted text → reply in chat (and a file too if it's long). `build_output.py` accepts `--format docx|md|txt`.

If any section still scores high (≳55), revise just those sections and rebuild. Two passes is usually enough; don't loop endlessly chasing a number — a lower score from mangled prose is worse than a slightly higher score from good prose.

## Step 5 — Report back

Keep it short:

```
**AI score:** 78 → 31 (likely AI → likely human)
**What changed:** varied sentence rhythm, removed 42 stock phrases ("delve", "it is important to note"…), replaced generic claims with the specifics already in your text, added contractions where the register allows.
**File:** report_humanized.docx (same headings and length, ~6,200 words)
**Worth a manual look:** section 4 (still reads formal — it's a methods section, which is fine)
```

Plus the before/after report path if they want detail. Remind them in one line that detector scores are estimates and that they should read it through to make sure it sounds like them.

## Scripts reference

| Script | Purpose |
|---|---|
| `extract_text.py FILE -o OUT` | PDF/DOCX/TXT/MD → plain text (keeps headings as `#` lines for DOCX) |
| `chunk_text.py FILE --out DIR --words N` | Split at paragraph boundaries into ~N-word chunks + manifest |
| `ai_score.py FILE [--report MD] [--json J] [--compare J]` | Heuristic AI-likeness score, per-section breakdown, flagged sentences |
| `build_output.py DIR --format docx\|md\|txt -o OUT` | Join `*.human.txt` chunks (falls back to originals if missing — and warns) |

Dependencies: `pypdf` and `python-docx` (`pip install pypdf python-docx`). Scoring and chunking use only the standard library.
