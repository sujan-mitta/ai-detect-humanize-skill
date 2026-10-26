# AI Detect & Humanize: a Claude skill

A [Claude skill](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview) that checks whether writing reads as AI-generated, and rewrites it into natural, human-sounding prose without losing a single fact. It handles anything from one pasted paragraph to long PDFs and Word documents.

> Scores are statistical estimates, not proof of authorship. The skill says so every time, and won't call a text "definitely AI" or "definitely human".

## What it does

| Step | What happens |
|---|---|
| **Read** | Extracts `.pdf`, `.docx`, `.txt`, `.md` or pasted text, keeping headings, lists and tables (PDF paragraphs and headings are reconstructed) |
| **Detect** | A trained model (three tiers: style, `--deep` with GPT-2, `--max` with neural AI-text classifiers) gives a 0–100 AI-likeness score, explains what pushed it toward AI or human, scores each section (so a half-human, half-AI document shows up as *mixed*), and lists the most AI-sounding sentences |
| **Deep mode** | Optional GPT-2 perplexity features (how predictable each word is), the signal tools like GPTZero use |
| **Heatmap** | An HTML report that highlights every sentence by how AI-like it reads, with reasons on hover, plus a side-by-side before/after view |
| **Humanize** | Long documents are split into ~900-word chunks so nothing gets summarized away. Each chunk is rewritten following a catalog of AI writing patterns and a writing guide |
| **Verify** | A fact checker compares every number (spelled-out ones too), date, citation, quote, URL and heading between the original and the rewrite, and fails if anything is lost or invented |
| **Voice** | Builds a profile from your own writing (sentence length, contractions, punctuation habits, pet words) and scores how closely a rewrite matches it |
| **Rebuild** | Outputs the same format you gave it (PDF/DOCX → DOCX, MD → MD, TXT → TXT) with a before/after score |

## How accurate is detection?

Short version: in `--max` mode, about **8 in 10 AI texts are caught while about 2 in 100 human texts are flagged**, including on kinds of writing the model has never seen, and including output from older open models. It is weakest on technical writing, AI fiction and "casual" rewrites, and it flags 5-10% of formal human prose.

**Training data (2,100 labelled texts, 12 genres):**
- **1,066 human texts** from before LLMs were common: Usenet, pre-2021 arXiv, Wikipedia, Project Gutenberg books, Paul Graham essays, StackExchange, Wikinews, Python PEPs, IETF RFCs, CPython/Linux/Rust/Git documentation, Enron-era emails, US government (GAO/CRS) reports, Yelp reviews and Reddit posts.
- **1,034 AI texts** from three Claude models and ChatGPT-style prompting across the same genres, including fiction, tutorials and protocol write-ups. About 100 are AI text rewritten by an assistant to "sound human" or "casual".
- The style and template models learn from this modern-LLM text only. A separate "legacy" head (below) also uses 300 texts from a public benchmark.

**How it is tested.** The strict test is *leave-one-genre-out*: a whole genre (say, all technical writing, human and AI) is held back, the model trains on the other 11, and then scores the held-out genre. That is the situation when someone hands it a kind of document it has not seen. Verdict thresholds come from these held-out scores so that at most 2% of held-out human texts are called "likely AI".

| Mode | AUC | AI caught at 1% human false positives | Verdict "likely AI": AI caught / human flagged |
|---|---|---|---|
| Style (default, no downloads) | 0.976 | 73% | 78% / 2% |
| Deep (`--deep`, + GPT-2) | 0.975 | 73% | 80% / 2% |
| **Max (`--max`, + neural detectors)** | **0.975** | **77%** | **84% / 2%** |

When genres *are* seen in training but individual sources and authors are held out (the normal deployed situation), max mode gets AUC 0.999 and catches 97% of AI at 1% false positives. Treat that as an upper bound: the AI and human samples share topics and prompt styles.

**By genre, leave-one-genre-out, max mode** (AI caught / human wrongly flagged): academic 99% / 7%, essays 96% / 0%, reviews 100% / 6%, Q&A 94% / 1%, forum posts 93% / 0%, news 97% / 0%, encyclopedia 98% / 5%, emails 87% / 0%, business and government reports 89% / 6%, fiction 70% / 1%, technical writing 59% / 0%. AI-written cover letters, LinkedIn posts and thank-you notes (never trained on): 100% caught. 40 passages from books the model never saw: 2% flagged.

**Older and weaker generators.** Style cues are generator-specific: a style model trained on GPT-2/OPT/LLaMA-1/BLOOM-class output as well as modern text got *worse* on both (AUC 0.96 -> 0.78). So the style model stays modern-only, and `--max` adds a second "legacy" head that sees only predictability features (GPT-2 statistics, Binoculars, the two neural classifiers) and is trained on ours plus 300 benchmark texts. The higher of the two scores is used; the legacy head can only add detections. On 316 benchmark texts (COLING-2025 dev and RAID; none used in training), max mode catches **79% of AI at 3% human false positives (AUC 0.95)**, up from 27% / 1% before the head existed. Split by source: RAID (a benchmark the head never saw) 88% caught / 0% flagged; COLING (same benchmark family as its training sample, so optimistic) 75% / 5%. Style and deep modes do not have this head and stay close to chance on those generators (AUC 0.57 / 0.69). On our own corpus the head costs nothing (84% -> 83% caught, 1.9% -> 1.7% flagged).

**Weak spots, stated plainly:**
- **Technical writing** is the weakest genre: about 59% of AI text caught when the genre is unseen (it was 35% before adding real documentation and tutorial text, 99% when the genre is seen).
- **AI fiction** about 70% and "casual"/humanized AI about 76% on unseen genres.
- **Human false positives are not zero.** Formal, polished human prose is flagged at 5-10% (academic, reviews, government reports).
- **Short text:** under 100 words no verdict is given. Under about 200 words treat the number as a weak signal. Length-versus-accuracy was measured on an earlier version of the model (about 88% at 300 words, 95% at 1,000+ words); the direction still holds, but those figures were not re-measured for this release.
- A detector cannot prove authorship. Report the score as an estimate.

**Formatting does not move the score.** Merging paragraphs, hard-wrapping, curly vs straight quotes, double spaces, Windows line endings or pre-tokenised text changed scores by 0.2 points on average at most (0% verdict flips) on plain prose. Earlier versions moved up to 53 points on such edits, because they were reading layout rather than writing; those shortcuts were found by auditing each feature's single-feature predictive power within a genre, then removed.

There is no upper limit on length: long documents are also scored section by section (about 250 words each), so a mixed document shows which parts read as AI.

**Speed on an ordinary laptop CPU:**

| Document | Style score | Deep score (GPT-2) |
|---|---|---|
| 1,000 words | 0.5 s | 3 s |
| 5,000 words | 2 s | 14 s |
| 20,000 words (≈65 pages) | 7 s | 24 s |
| 60,000 words (≈200 pages) | 19 s | 25 s |

Deep mode reads the text in a single GPT-2 pass. Beyond about 9,000 words it scores 12,000 evenly spaced tokens instead of every one, since its features are averages. A whole book therefore takes under half a minute.

### Humanizing benchmark

24 AI texts across six genres were rewritten by agents following the skill (measured with an earlier version of the scorer; not re-run for this release, so read the scores as indicative):

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

Optional, for `--deep` and `--max` (no graphics card needed). The first run downloads GPT-2 (~500 MB) for `--deep`, plus the fakespot, desklib and Binoculars (SmolLM2) models (~3.7 GB) for `--max`:

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
2. **Measure about 40 style features plus a learned sentence-template score** (part-of-speech skeleton n-grams, trained to separate AI from human structure). Sentence and paragraph rhythm, vocabulary variety, punctuation habits (parentheses, semicolons, em dashes, questions), specificity (numbers, names), first person, contractions, and the AI patterns from the catalog (stock phrases, participial tails, "not X but Y" frames, rule-of-three lists, bold lead-ins, chatbot leftovers, fake-casual tics).
3. **Optionally add five GPT-2 features** (`--deep`): perplexity, top-1 and top-10 token rates, rare-word rate, and perplexity variation across sentences. `--max` also adds the AI probability from two pretrained neural classifiers (fakespot RoBERTa and desklib DeBERTa-v3) and the Binoculars score (SmolLM2-360M base vs instruct), averaged over up to six evenly spaced chunks. Binoculars is weak alone with CPU-sized models, but it lifts AI caught in unseen genres from 79% to 86%.
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
│   ├── neural.py                  # neural AI-text classifiers (--max)
│   ├── html_report.py             # sentence heatmap / before-after page
│   ├── verify_rewrite.py          # fact-preservation check
│   ├── style_profile.py           # voice profile + match score
│   ├── build_output.py            # join chunks → DOCX/MD/TXT
│   └── model.json                 # trained model (boosted trees, style + deep)
└── evals/                         # 11 end-to-end test cases (prompt, inputs, graders)
dev/                               # tooling to rebuild the model
├── build_human_corpus.py          # downloads the human corpus
├── calibrate.py                   # trains + cross-validates, writes model.json
├── neural_features.py             # runs the neural classifiers over the corpus
├── long_doc_test.py               # accuracy on long documents
├── humanize_bench.py              # measures humanizing results
└── make_eval_cases.py             # generates eval case folders
ai-detect-humanize.skill           # packaged skill for upload
```

The calibration corpus itself (`dev/corpus/`) isn't committed, since it contains third-party text. `python dev/build_human_corpus.py` re-downloads the human half.
