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

Short version: in `--max` mode, about **85 in 100 AI texts are caught while about 2 in 100 human texts are flagged**, including on kinds of writing the model has never seen, and about **3 in 4 AI texts from older open models**. It is weakest on technical writing (about 63%), and it flags about 7% of formal government-report prose.

**Training data (3,132 labelled texts, 12 genres):**
- **1,524 human texts** from before LLMs were common: Usenet, arXiv abstracts, Wikipedia (WikiText), Project Gutenberg books, Reddit WritingPrompts stories, Paul Graham essays, StackExchange, Wikinews, Python PEPs, IETF RFCs, CPython/Linux/Rust/Git documentation, Enron-era emails, US government (GAO/CRS) reports, Yelp reviews and Reddit posts.
- **1,608 AI texts** from three Claude models and ChatGPT-style prompting across the same genres, matched to the human topics where possible (the same Wikipedia articles, the same writing prompts, the same RFCs). About 190 are AI text rewritten by an assistant to sound "casual", "natural" or "human".
- The style and template models learn from this modern-LLM text only. A separate "legacy" head (below) also uses 300 texts from a public benchmark.

**How it is tested.** The strict test is *leave-one-genre-out*: a whole genre (say, all technical writing, human and AI) is held back, the model trains on the other 11, and then scores the held-out genre. That is the situation when someone hands it a kind of document it has not seen. Verdict thresholds come from these held-out scores so that at most 2% of held-out human texts are called "likely AI".

| Mode | AUC | AI caught at 1% human false positives | Verdict "likely AI": AI caught / human flagged |
|---|---|---|---|
| Style (default, no downloads) | 0.974 | 74% | 83% / 2% |
| Deep (`--deep`, + GPT-2) | 0.975 | 74% | 83% / 2% |
| **Max (`--max`, + neural detectors)** | **0.978** | **83%** | **86% / 2%** |

When genres *are* seen in training but individual sources and authors are held out (the normal deployed situation), max mode gets AUC 0.998. Treat that as an upper bound: the AI and human samples share topics and prompt styles.

**By genre, leave-one-genre-out, max mode** (AI caught / human wrongly flagged): academic 87% / 3%, essays 96% / 0%, reviews 99% / 3%, Q&A 96% / 1%, forum posts 95% / 0%, news 99% / 0%, encyclopedia 80% / 1%, emails 89% / 0%, business and government reports 94% / 7%, fiction 77% / 1%, technical writing 63% / 0%. AI-written cover letters, LinkedIn posts and thank-you notes (never trained on): 100% caught. 40 passages from books the model never saw: 2% flagged.

**Older and weaker generators.** Style cues are generator-specific: a style model trained on GPT-2/OPT/LLaMA-1/BLOOM-class output as well as modern text got *worse* on both (AUC 0.96 -> 0.78). So the style model stays modern-only, and `--max` adds a second "legacy" head that sees only predictability features (GPT-2 statistics, Binoculars, the two neural classifiers) and is trained on ours plus 480 benchmark texts and 144 Gemini texts. The higher of the two scores is used; the legacy head can only add detections. On 316 benchmark texts (COLING-2025 dev and RAID; none used in training), max mode catches **about 78% of AI at 1.3% human false positives (AUC 0.97)**, up from 27% before the head existed. Split by source: RAID (a benchmark the head never saw) 83% caught / 0% flagged; COLING (same benchmark family as its training sample, so optimistic) 71% / 2%. Style mode does not have this head and stays close to chance on those generators; `--deep` has a weaker GPT-2-only version (see below). On our own corpus the head costs almost nothing (86% -> 85% caught, 1.7% -> 1.6% flagged).

**Modern non-Claude models** (`dev/family_eval.py`; 140 benchmark texts and 90 human texts the model never saw, max mode, at the verdict threshold, 0 of 90 humans flagged): GPT-4/4o 100%, GPT-3.5 96%, Mixtral 93%, Cohere 79%, **Llama-3 68%, Gemma 61%**. Adding 120 more Llama-3/Gemma training texts moved these by at most 4 points, so they look genuinely close to human on the signals measured here. **Gemini** (240 texts written by Gemini Pro and Flash through Antigravity, plain and "make it sound human" prompts, 12 genres): before any Gemini training only 65-72% were above the 60 line. After adding files 01-03 of each genre to the legacy head, the held-out files 04-05 (96 texts, different topics, same prompt templates, so optimistic) are **88% above the 60 line and 78% above the stricter verdict threshold (85)**. Note the two thresholds: "likely AI" (60) catches more but also flags more humans (5% of external human texts); the verdict threshold flags 1.3%. Other prompts or newer Gemini versions may score lower. The legacy head now only adds to a score when its own score is at least 80 (a gate); without it, 10.8% of external human texts were above 60, with it 5.1%, and verdict-level results are unchanged.

**False positives on genres it never saw.** 160 human texts from three genres never used in training (legal bill summaries, statute text, medical textbook passages) were all scored: 0 flagged as likely AI, 97-98% read as human (max mode).

**Weak spots, stated plainly:**
- **Technical writing** is the weakest genre: about 63% of AI text caught when the genre is unseen (it was 23-35% before adding real documentation and tutorial text; 96% when the genre is seen).
- **Formal human prose:** government reports are flagged 7% of the time; academic abstracts 3%.
- **Rewritten AI:** AI text rewritten to sound casual or natural is caught about 78% of the time; encyclopedic AI about 80%.
- **Technical and government-report prose:** I added 120 more AI examples of these genres and retrained. Held-out report AI caught went up (68% to 92% on the new style), but the retrained model's verdict threshold rose and it caught fewer AI texts overall at the verdict on non-Claude generators, so I did not adopt it; the shipped model is unchanged. Unseen-genre technical recall (about 63%) and report false positives (about 7% on unseen genres) are unchanged.
- **Style mode** does not detect older open-model output; use `--max` for that. `--deep` has a smaller GPT-2-only legacy head: on the older-model benchmark it catches 44% of AI instead of 19%, at the cost of about 1 more point of human texts flagged (3.6% -> 4.8% on held-out humans). Still far weaker than `--max`.
- **Short text:** under 60 words no verdict is given, and neither under 100 words without `--max`. With `--max`, 60-99 words get a tentative flag only (measured on chunks of whole sentences: AUC 0.91 at 60-80 words and 0.90 at 80-99; about half of AI chunks flagged above 60 and 3-4% of human chunks; at 35-60 words AUC is 0.79 and only 23% of AI is flagged). A low score never clears a short text. Under about 200 words treat the number as a weak signal. Length-versus-accuracy was measured on an earlier version of the model (about 88% at 300 words, 95% at 1,000+ words); the direction still holds, but those figures were not re-measured for this release.
- A detector cannot prove authorship. Report the score as an estimate.

**Formatting does not move the score.** Merging paragraphs, hard-wrapping, curly vs straight quotes, double spaces, Windows line endings or pre-tokenised text changed scores by 0.7 points at most on average (0% verdict flips) on plain prose. Earlier versions moved up to 53 points on such edits, because they were reading layout rather than writing; those shortcuts were found by auditing each feature's single-feature predictive power within a genre, then removed.

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

Tested on 36 fresh AI texts (six genres) the detector had never seen, in `--max` mode:

| Rewrite method | Mean score | Rewrites that leave "likely AI" (<60) | Rewrites that read as human (<40) |
|---|---|---|---|
| Before (original AI text) | 98 | 0 of 36 | 0 of 36 |
| One pass of the usual edits (plainer words, contractions, varied openers) | 97 | 0 of 36 | 0 of 36 |
| Up to five rounds of clause-structure edits, tracked against the score (6 texts, two models) | 70-78 | 4 of 12 | 0 of 12 |
| Same, but guided by `rewrite_targets.py` (ranks the sentences to change first; same 6 texts, two models) | 60-72 | 6 of 12 | 0 of 12 (one at 41) |

So: the default quick rewrite does not change what a strong detector sees, and even the iterative workflow only moves some texts (essays, news and emails best; blogs and technical text least) out of the "likely AI" band. Pointing each round at the specific sentences to change (`scripts/rewrite_targets.py`) moved more of them than iterating blind. The workflow in `SKILL.md` now describes the iterative method and what worked (`references/humanizing-guide.md` §2c). I also tested the idea that adding the author's own details is what finally makes a text read as human, and it did not hold up: on 12 drafts, weaving in specific personal details (invented by an AI playing the author, so a real author might do better, which I could not test) changed the mean `--max` score only from 98 to 95 (0 of 12 under 60), and details plus the structure method gave 98 to 77 (3 of 12 under 60), no better than the structure method alone. The detector reads sentence structure far more than content. `scripts/author_questions.py` still helps make a draft truly the author's, but it is not a way to lower the score. Earlier figures in this section (for example "96 to 48") were measured with the fast style model that the rewriters were optimising against; they are not representative and have been removed.

Fact preservation (`verify_rewrite.py`) passed on all of the iterative rewrites except for flagged number-word changes such as "a pair" becoming "two", which are for the writer to confirm. Length stayed within 94-113% of the originals.

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
