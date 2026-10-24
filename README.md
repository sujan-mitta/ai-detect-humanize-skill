# AI Detect & Humanize — a Claude skill

A [Claude skill](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview) that checks whether writing reads as AI-generated and rewrites it into natural, human-sounding prose. It works on anything from a single paragraph to very large documents, PDFs and Word files.

## What it does

1. **Extracts** text from `.pdf`, `.docx`, `.txt`, `.md` or pasted text, keeping headings, lists and tables.
2. **Detects** AI-style writing with an offline heuristic scorer: an overall 0–100 score, per-section scores, and the most AI-sounding sentences with reasons.
3. **Chunks** long documents at paragraph boundaries (~900 words each) so every part is fully rewritten, not summarized.
4. **Humanizes** each chunk following a writing guide. Facts, numbers, citations, headings and length are preserved, and one voice is kept across the document.
5. **Rebuilds** the output in the input's format (PDF/DOCX → DOCX, MD → MD, TXT → TXT) and re-scores it for a before/after comparison.

> The score is a stylistic estimate, not proof of authorship. No detector can reliably tell who wrote a text.

## Install

**Claude app (claude.ai / desktop):** download [`ai-detect-humanize.skill`](ai-detect-humanize.skill) and upload it under *Settings → Capabilities → Skills*.

**Claude Code:** copy the skill folder into your skills directory:

```bash
cp -r ai-detect-humanize ~/.claude/skills/
```

Python dependencies for the scripts:

```bash
pip install pypdf python-docx
```

## Usage

Just ask Claude, for example:

- "Is this essay AI-written? Give me a score."
- "Humanize this PDF and give me a Word file back."
- "Make this report sound less like ChatGPT, keep all the numbers."

The scripts also work on their own:

```bash
python ai-detect-humanize/scripts/ai_score.py report.pdf --report before.md --json before.json
python ai-detect-humanize/scripts/chunk_text.py report.pdf --out work/chunks --words 900
# ...rewrite each chunk_NNN.txt into chunk_NNN.human.txt...
python ai-detect-humanize/scripts/build_output.py work/chunks --format docx -o report_humanized.docx
python ai-detect-humanize/scripts/ai_score.py report_humanized.docx --compare before.json
```

## How the score works

The scorer measures 15 signals, grouped like the pattern catalog:

| Group | Signals |
|---|---|
| Rhythm | Sentence-length variety (burstiness); same-length paragraphs that end in wrap-up sentences |
| Vocabulary | AI-typical words and phrases ("delve", "it is important to note", "as X continues to evolve"…); stacked hedges, "truly/incredibly", "experts say" |
| Sentence templates | Participial tails (", ensuring…", ", highlighting…"); contrast frames ("It's not X — it's Y", "The result?"); repeated openers ("This…", "By -ing…"); *Furthermore/Moreover* openers; rule-of-three lists |
| Fake-casual voice | Fragment runs ("Simple. Effective. Human."), "Honestly?", "Here's the thing" — what AI writes when told to sound human |
| Punctuation | Few contractions; heavy em-dash use |
| Document | Bold-label bullets, emoji markers; chat leftovers ("I hope this helps!"); repeated phrasing |

The final score blends the weighted average with the strongest few signals, so clear evidence isn't averaged away. **60+** likely AI · **40–59** mixed · **under 40** likely human.

On top of the script, Claude audits the text against [`references/ai-patterns.md`](ai-detect-humanize/references/ai-patterns.md), a catalog of how language models write. It covers vocabulary, sentence templates, grammar, rhythm, paragraph and document structure, formatting, tone, genre-specific tells, and the tics models fall into when told to "sound human".

## Repository layout

```
ai-detect-humanize/
├── SKILL.md                       # instructions Claude follows
├── references/
│   ├── ai-patterns.md             # full catalog of AI writing patterns
│   └── humanizing-guide.md        # rewriting techniques + examples
└── scripts/
    ├── extract_text.py            # PDF/DOCX/TXT/MD → text
    ├── chunk_text.py              # split long texts into chunks
    ├── ai_score.py                # heuristic AI-likeness scorer
    └── build_output.py            # join chunks → DOCX/MD/TXT
ai-detect-humanize.skill           # packaged skill for upload
```
