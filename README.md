# AI Detect & Humanize: a Claude skill

A [Claude skill](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview) that checks whether writing reads as AI-generated, and rewrites it into natural, human-sounding prose without losing a single fact. It handles anything from one pasted paragraph to long PDFs and Word documents.

> Scores are statistical estimates, not proof of authorship. The skill says so every time, and won't call a text "definitely AI" or "definitely human".

## What it does

| Step | What happens |
|---|---|
| **Read** | Extracts `.pdf`, `.docx`, `.txt`, `.md` or pasted text, keeping headings, lists and tables (PDF paragraphs and headings are reconstructed) |
| **Detect** | A trained model gives a 0–100 AI-likeness score, explains what pushed it toward AI or human, scores each section (so a half-human, half-AI document shows up as *mixed*), and lists the most AI-sounding sentences |
| **Deep mode** | Optional GPT-2 perplexity features (how predictable each word is), the signal tools like GPTZero use |
| **Heatmap** | An HTML report that highlights every sentence by how AI-like it reads, with reasons on hover, plus a side-by-side before/after view |
| **Humanize** | Long documents are split into ~900-word chunks so nothing gets summarized away. Each chunk is rewritten following a catalog of AI writing patterns and a writing guide |
| **Verify** | A fact checker compares every number (spelled-out ones too), date, citation, quote, URL and heading between the original and the rewrite, and fails if anything is lost or invented |
| **Voice** | Builds a profile from your own writing (sentence length, contractions, punctuation habits, pet words) and scores how closely a rewrite matches it |
| **Rebuild** | Outputs the same format you gave it (PDF/DOCX → DOCX, MD → MD, TXT → TXT) with a before/after score |

## How accurate is detection?

The model was trained and cross-validated on **1,122 labelled texts**:
- **603 human texts** written before LLMs were common: 1990s Usenet, arXiv abstracts from before 2021, Wikipedia, Project Gutenberg (19th-century books and 1900–1928 fiction and essays: Fitzgerald, Joyce, Woolf, Cather, Mansfield, Chesterton), 75 Paul Graham essays, StackExchange answers from 24 sites (before 2020), Wikinews, and Python PEPs.
- **519 AI texts** from three Claude models plus ChatGPT-style writing, across forum posts, Q&A, essays, academic writing (with and without citations), encyclopedia articles, news, technical writing and **106 pieces of fiction**. Some were prompted to "sound human".

5-fold cross-validation:

| Model | AUC | Accuracy | AI caught | Humans wrongly flagged |
|---|---|---|---|---|
| Old hand-tuned rules (v1) | 0.51 on Claude-only data | — | 16% | — |
| Style model (default) | 0.973 | 90.8% | 90.8% | 9.1% |
| Style + GPT-2 (`--deep`) | **0.981** | **93.4%** | **93.3%** | **6.5%** |

**Verdicts are tuned to avoid false accusations.** The "likely AI" cut-off is set from the data so that only 5% of human texts reach it. Borderline texts get "mixed / uncertain" instead.

| Verdict | Human texts | AI texts |
|---|---|---|
| Deep mode: "likely AI" (score ≥ 57) | 5% | 91% |
| Style mode: "likely AI" (score ≥ 66) | 5% | 86% |
| Style mode: "likely human" (score < 52) | 91% | 9% |

On genres the model never trained on (emails, cover letters, LinkedIn posts, blog posts, business reports), it catches 76% of AI text.

**By genre (deep mode):**
- **AI fiction:** 92–100% caught, up from about 50% before the fiction data was added.
- **Human fiction:** 0% wrongly flagged.
- **Human essays:** 12% wrongly flagged at the raw 50 cut-off (down from 16%); the 5% "likely AI" threshold keeps actual accusations rarer.
- **Human Q&A answers:** 10% wrongly flagged.

AI-polished human text lands anywhere, which is expected.

### How much text does it need? (no graphics card required)

| Text length | Accuracy |
|---|---|
| 50 words | 61% (unreliable) |
| 100 words | 68% (unreliable) |
| 200 words | 80% |
| **300 words (≈1 page)** | **88%** |
| 1,000 words (≈3 pages) | **95%** |
| 3,000 words (≈10 pages) | **95%**, 3% of human documents wrongly flagged |
| 6,000 words (≈20 pages) | 100% on 14 test documents (small sample) |

Long documents are measured by joining held-out texts of the same kind; the model never saw any of them. There's no upper limit on length: long documents are also scored section by section (about 250 words each), so a mixed document shows which parts read as AI.

**Speed on an ordinary laptop CPU:**

| Document | Style score | Deep score (GPT-2) |
|---|---|---|
| 1,000 words | 0.5 s | 3 s |
| 5,000 words | 2 s | 14 s |
| 20,000 words (≈65 pages) | 7 s | 24 s |
| 60,000 words (≈200 pages) | 19 s | 25 s |

Deep mode reads the text in a single GPT-2 pass. Beyond about 9,000 words it scores 12,000 evenly spaced tokens instead of every one, since its features are averages. A whole book therefore takes under half a minute.

### Humanizing benchmark

24 AI texts across six genres were rewritten by agents following the skill:

| | Before | After |
|---|---|---|
| Mean AI score (style) | 96 | 48 |
| Mean AI score (GPT-2 deep) | 98 | 54 |
| Reads as human | 0% | 50% (style) / 38% (deep) |
| Facts check passed | — | 88% (the 3 failures each added a count that wasn't in the original) |
| Length kept | — | 99% on average |
| Fake-casual tics added | — | 0 |

Encyclopedia-style and literary text humanize best. Product copy and formal Q&A stay AI-like more often, because their format itself reads as AI.

### End-to-end tests (with vs. without the skill)

The 11 realistic requests in [`ai-detect-humanize/evals/`](ai-detect-humanize/evals) were each run twice by Claude Sonnet, once with the skill and once without, then graded against 64 checks:

| | With skill | Without skill |
|---|---|---|
| Checks passed | **63 / 64 (98%)** | 57 / 64 (89%) |
| Avg time per task | 43 s | 27 s |
| Avg tokens per task | 70k | 60k |

What the skill fixed:
- **No dropped facts.** Without the skill, the cover-letter rewrite dropped the year from a date, and the long-PDF rewrite lost a number.
- **Long PDF.** With the skill it was chunked and checked; without it, it was rewritten in one pass, unchecked, and came out *more* AI-like (89 → 100).
- **Scoring.** With the skill, every request got a real before/after score.

The one miss with the skill: a formal business report stayed at 96 after humanizing. That genre reads as AI whoever writes it, and the skill says so rather than mangling the prose.

The detection tasks (spotting a pasted chatbot answer, finding which of three files is AI) were passed both with and without the skill: Claude is already good at reading text. The skill's value there is calibrated numbers, per-section scores, the heatmap, and finding the AI half of a mixed document.

## Install

**Claude app (claude.ai / desktop):** download [`ai-detect-humanize.skill`](ai-detect-humanize.skill) and upload it under *Settings → Capabilities → Skills*.

**Claude Code:**

```bash
cp -r ai-detect-humanize ~/.claude/skills/
```

```bash
pip install pypdf python-docx
```

Optional, for deep mode (downloads ~500 MB of GPT-2 the first time):

```bash
pip install torch transformers
```

## Usage

Just ask Claude, for example:

- "Is this essay AI-written? Give me a score."
- "Humanize this PDF and give me a Word file back, keep all the numbers."
- "Rewrite my cover letter so it sounds like me, here's something I wrote."

Or run the scripts yourself:

```bash
python ai-detect-humanize/scripts/ai_score.py report.pdf --deep --report before.md --json before.json --html heatmap.html
python ai-detect-humanize/scripts/chunk_text.py report.pdf --out work/chunks --words 900
# rewrite each chunk_NNN.txt into chunk_NNN.human.txt, then for each one:
python ai-detect-humanize/scripts/verify_rewrite.py work/chunks/chunk_001.txt work/chunks/chunk_001.human.txt
python ai-detect-humanize/scripts/build_output.py work/chunks --format docx -o report_humanized.docx
python ai-detect-humanize/scripts/ai_score.py report_humanized.docx --compare before.json
python ai-detect-humanize/scripts/html_report.py report_humanized.docx --compare report.pdf -o before_after.html
python ai-detect-humanize/scripts/style_profile.py my_old_essay.txt my_emails.txt -o voice.json
```

## How the scorer works

1. **Clean the text.** Code, links, URLs, citations like *(Smith et al., 2019)*, bracketed statistics like *(n = 214)*, acronym definitions and LaTeX are removed first, so they can't be mistaken for writing style.
2. **Measure 30 style features.** Sentence and paragraph rhythm, vocabulary variety, punctuation habits (parentheses, semicolons, em dashes, questions), specificity (numbers, names), first person, contractions, and the AI patterns from the catalog (stock phrases, participial tails, "not X but Y" frames, rule-of-three lists, bold lead-ins, chatbot leftovers, fake-casual tics).
3. **Optionally add five GPT-2 features** (`--deep`): perplexity, top-1 and top-10 token rates, rare-word rate, and perplexity variation across sentences.
4. **Score with gradient-boosted trees**, with the confidence calibrated (Platt scaling) so the score works as a probability. Known AI tells are constrained so they can only ever push the score toward AI. The trees are exported to `model.json` and run in plain Python, so scikit-learn isn't needed at runtime.
5. **Apply a safety net for dense AI tells.** Short texts stuffed with classic phrases ("leveraging", "spearheaded", "Let's connect!") give the model too little to go on. A second score built only from known tells is therefore combined with the model's. It only switches on above 2.7 tells per 100 words, the highest density found in any of the 394 human texts, so it added no false positives on that corpus.
6. **Flag mixed documents.** When some sections read as AI and others as human, the verdict says "mixed" and names the AI sections, instead of averaging them away.
7. **Explain the score.** Each feature is reset to the typical value and the change in score is measured. The report then lists, for example, "fewer parentheses than typical human writing" or "more rule-of-three lists", and marks features a faithful rewrite can't change (numbers, names) as *(content)*.

Claude also audits each text against [`references/ai-patterns.md`](ai-detect-humanize/references/ai-patterns.md), a catalog of how language models write. It covers vocabulary, sentence templates, grammar, rhythm, paragraph and document structure, formatting, tone, genre-specific tells, the tics models fall into when told to "sound human", and what the calibration data showed about newer models.

## Repository layout

```
ai-detect-humanize/                # the skill
├── SKILL.md                       # instructions Claude follows
├── references/
│   ├── ai-patterns.md             # catalog of AI writing patterns
│   └── humanizing-guide.md        # rewriting techniques + examples
├── scripts/
│   ├── extract_text.py            # PDF/DOCX/TXT/MD → text
│   ├── chunk_text.py              # split long texts into chunks
│   ├── ai_score.py                # calibrated AI-likeness scorer + explanations
│   ├── perplexity.py              # GPT-2 deep features
│   ├── html_report.py             # sentence heatmap / before-after page
│   ├── verify_rewrite.py          # fact-preservation check
│   ├── style_profile.py           # voice profile + match score
│   ├── build_output.py            # join chunks → DOCX/MD/TXT
│   └── model.json                 # trained model (boosted trees, style + deep)
└── evals/                         # 11 end-to-end test cases (prompt, inputs, graders)
dev/                               # tooling to rebuild the model
├── build_human_corpus.py          # downloads the human corpus
├── calibrate.py                   # trains + cross-validates, writes model.json
├── humanize_bench.py              # measures humanizing results
└── make_eval_cases.py             # generates eval case folders
ai-detect-humanize.skill           # packaged skill for upload
```

The calibration corpus itself (`dev/corpus/`) isn't committed, since it contains third-party text. `python dev/build_human_corpus.py` re-downloads the human half.
