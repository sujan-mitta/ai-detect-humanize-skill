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

No detector — commercial or this one — can prove who wrote a text. The scorer is a statistical model trained on labelled human and AI texts; the report prints its measured held-out accuracy, and it still misfires on some genres (polished human essays get flagged; carefully prompted AI slips through). Report results as "reads as likely AI / mixed / likely human", give the score, and say it's an estimate. Never tell a user a text is definitely AI-written or definitely human; people get accused unfairly on that basis.

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
python scripts/ai_score.py work/source.txt --deep --report work/before_report.md --json work/before.json --html work/before_heatmap.html
```

- The score (0–100) is the calibrated probability from `scripts/model.json`, a model fitted on ~3,100 labelled texts across 12 genres (forum, Q&A, essays, academic, encyclopedia, news, fiction, technical, email, reports, reviews, posts). The report says which features pushed it toward AI and which toward human, plus the pattern signals, a per-section breakdown and the most suspicious sentences.
- `--deep` adds GPT-2 perplexity — how predictable each word is to a language model, the main signal tools like GPTZero use. It's more accurate than the default style model, mainly on fiction and reports. It needs `pip install torch transformers` and downloads ~500 MB the first time; no graphics card is needed. On a normal CPU it takes ~3 s for 1,000 words and caps at ~25 s for anything longer than ~20,000 words, because long texts are sampled evenly, so use it freely, even on whole books. If torch isn't installed the script says so and uses the style model; mention that deep mode exists if the user wants a stronger check.
- `--max` is the most accurate mode: on genres it never trained on, about **86% of AI text caught with 2% of human texts flagged (AUC 0.978)**, and about 75% of output from older open models at 1% human flagged. Technical writing (~63%) and formal human prose (government reports ~7% flagged) are the weak spots. On top of `--deep` it runs two pretrained neural AI-text classifiers, fakespot (RoBERTa) and desklib (DeBERTa-v3), plus Binoculars (SmolLM2 base vs instruct). It downloads ~3.7 GB once and runs on CPU, reading at most six evenly spaced ~400-word chunks, so it takes about 10–25 s per text and under a minute even for long documents.
- **Default: use `--max` whenever its models are installed** (check with `python -c "import torch, transformers"` and run it; if it reports missing packages or models it can't download, fall back to `--deep`, then the style model). Tell the user which mode produced the score. If they have no space or time for the one-off download, say `--deep` is the fallback and a little less accurate.
- **Evasion-proofing.** Before scoring, the text is Unicode-normalised: zero-width characters are removed and Cyrillic/Greek look-alike letters swapped into Latin words are mapped back. On public benchmarks (RAID), look-alike substitution is the cheapest attack and cut some detectors from 85% to 9%. Here the attacked text scores the same as the original. If any were found, the report says how many. Tell the user, but remember copy-paste from some PDFs and web pages produces stray zero-width characters innocently.
- **Short texts get no verdict.** Under 100 words the report shows the number but says "too short to judge", because accuracy there is poor. Relay that; don't turn the number into an accusation.
- **How much text the score needs.** Accuracy improves with length (roughly 88% at 300 words and 95% at 1,000+ in an earlier measurement) and is poor under 150 words. For anything under ~200 words, say clearly that the number is a weak signal and lean on your own reading of the patterns. For long documents, the section scores matter as much as the overall one.
- `--html` writes a heatmap: every sentence highlighted by how AI-like it reads, with the reasons on hover. Offer it to the user — for long documents it's the most useful thing to look at. `scripts/html_report.py REWRITE -o out.html --compare ORIGINAL` gives a side-by-side before/after.

Then do your own audit with `references/ai-patterns.md` — the full catalog of how language models write, organized by vocabulary, sentence templates, grammar, rhythm, paragraph and document structure, formatting, tone and genre. You are a language model yourself, so you can recognize these habits from the inside; use that. The script catches surface patterns; you catch what it can't: the topic-sentence → support → wrap-up paragraph template, both-sidesing, generic examples, synonym cycling, a conclusion that echoes the intro, uniform warmth, symmetric sections. Run through the checklist in §11 of the catalog and note which categories are "heavy".

If the verdict says **mixed**, lead with the section breakdown: an overall average hides an AI-written half next to a human one, and "which part" is what the user needs. If the text reads as human but you recognise it as a published work (a classic essay, a novel, a well-known article), say so. That's a sourcing question, not an AI one, and a low score doesn't mean the person wrote it.

If the user only asked for detection, stop here and present:
- Overall verdict + score, with the caveat above (one sentence, not a paragraph)
- Top 3–5 signals that drove the score, in plain language
- The pattern categories from your audit that are heavy (e.g. "every paragraph ends with a wrap-up sentence", "8 participial tails in 600 words")
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

Read `references/humanizing-guide.md` (rewriting techniques, register notes, examples) and `references/ai-patterns.md` (what to remove) before rewriting.

Work on structure and rhythm first, vocabulary last. Swapping "delve" for "dig into" while keeping the same paragraph template, triads and wrap-up sentences leaves text that still reads as AI. And watch your own output: when a model is asked to sound human it falls into a second set of tics — fragment overload, "Honestly?", "Here's the thing:", more em dashes, "It's not X. It's Y.", invented anecdotes. §9 of the catalog lists them; check each rewritten chunk against it.

First, settle the **voice** in 2–3 lines and keep it fixed across all chunks so the document doesn't change personality halfway through:
- Audience and register (academic, business, casual blog, personal statement…)
- Person (I / we / impersonal) — keep whatever the original used unless asked
- If the user can share something they wrote themselves (old essays, emails, posts — 500+ words is ideal), build a voice profile and write to it. It's the single best way to make a rewrite sound like *them* rather than like generic "human" text, so ask for a sample when the goal is "make it sound like me":
  ```bash
  python scripts/style_profile.py their_sample1.txt their_sample2.docx -o work/voice.json
  ```
  It prints concrete targets (sentence length and spread, contraction rate, punctuation they never use, pet words, typical openers). Put those in your voice notes.

Then rewrite each chunk to `work/chunks/chunk_001.human.txt` etc. Ground rules, because these are where rewrites usually go wrong:

- **Preserve meaning and every fact** — numbers, names, dates, citations, quotes, technical terms. Humanizing that changes what the text says is a failure.
- **Preserve structure** — headings, list items, tables, paragraph order. Markdown/headings in, same headings out. Bold *lead-in labels* inside paragraphs or bullets ("**Flexibility:** Remote work…") are different: they're one of the strongest AI tells, so turn them into ordinary sentences unless the format genuinely needs them (product spec sheets, FAQs) or the user asked to keep them.
- **Don't invent anything** — no new examples, anecdotes, names, numbers or sensory details, even to sound human. The facts check can't see an invented detail that has no number or name in it, so re-read each chunk against the original for added claims.
- **Don't shorten.** Aim for roughly the same length (±15%). Cutting content to sound punchier loses the user's material.
- **Don't overcorrect** into slang, typos or forced jokes. The target is a competent human writer, not a sloppy one.
- **Stay consistent at chunk seams** — read the last paragraph of the previous rewritten chunk before starting the next.

**Verify every chunk before moving on.** Rewrites quietly drop a statistic or "improve" a citation more often than you'd think, and an invented number is worse than an AI-sounding sentence:

```bash
python scripts/verify_rewrite.py work/chunks/chunk_001.txt work/chunks/chunk_001.human.txt --semantic
```

It compares numbers, percentages, dates, citations, URLs, quotes, code and headings, and exits with an error listing anything missing or newly added, plus the length ratio. It also warns when the number of negation words changes (a flipped claim: "did not suffer" → "suffered"). Add **`--semantic`** to also check meaning with a small embedding model (22M parameters, ~90 MB, CPU-fast): it lists original sentences with no close counterpart in the rewrite (possibly dropped) and rewrite sentences with no counterpart in the original (possibly invented). That is the only check that can see an invented claim containing no number or name. It's advisory, not a gate: on careful rewrites about 1 in 3 gets a flag, mostly rewritten rhetorical sentences, but real drops and inventions show up here, so read each flagged sentence and decide. Fix the chunk and re-run until it passes. Names that appear or disappear are listed as warnings — check them, since a rephrase can legitimately move a name. If you built a voice profile, `python scripts/style_profile.py --profile work/voice.json --check work/chunks/chunk_001.human.txt` shows how close the chunk is, metric by metric.

**Short or single-file texts** (under ~1,500 words) skip chunking and rebuilding:

```bash
python scripts/ai_score.py original.txt --json work/before.json
# rewrite -> work/rewrite.txt
python scripts/verify_rewrite.py original.txt work/rewrite.txt
python scripts/ai_score.py work/rewrite.txt --compare work/before.json
```

Below ~300 words the score is jumpy (a single sentence can move it 10+ points), so treat it as a rough guide and lean on your own audit.

**Using the score to revise.** The report lists what pushed the score toward AI, *compared with typical human writing* — "fewer parentheses" means the text has fewer than people usually write, not that parentheses are bad. Work on the items a rewrite can honestly change (sentence rhythm, rule-of-three lists, bold lead-ins, paragraph wrap-ups, repeated openers), and ignore anything marked *(content)*: numbers and names come from the facts, and you mustn't add or remove those. Things to know when revising:
- Don't chop sentences short to sound casual. Human writing in the training data runs *longer* and more varied than AI text; runs of short sentences read as AI.
- Don't swap in synonyms to avoid repeating a word. Models vary their vocabulary more than people do, and natural repetition reads as human.
- A parenthetical aside or a semicolon is fine where a person would actually use one. Adding them just to move the number is gaming the detector, and it makes the writing worse.

Stop after two revision passes. If a section still scores high but reads well, it's probably the genre (formal reports, encyclopedia-style and academic text score high even when written by people), so say so in the report rather than mangling the prose.

## Step 4 — Rebuild and re-score

```bash
python scripts/build_output.py work/chunks --format docx -o "<original-name>_humanized.docx"
python scripts/verify_rewrite.py work/source.txt "<original-name>_humanized.docx"
python scripts/ai_score.py "<original-name>_humanized.docx" --deep --report work/after_report.md --compare work/before.json
python scripts/html_report.py "<original-name>_humanized.docx" --compare work/source.txt -o work/before_after.html
```

Output format follows the input: `.pdf`/`.docx` → `.docx`; `.md` → `.md`; `.txt` → `.txt`; pasted text → reply in chat (and a file too if it's long). `build_output.py` accepts `--format docx|md|txt`.

If any section still scores high (≳60), revise just those sections and rebuild, following "Using the score to revise" above. Two passes at most; a lower score from mangled prose is worse than a slightly higher score from good prose. For a long document, `--deep` on the final file is the more honest check, since the style score is the one you revised against.

## Step 5 — Report back

Keep it short:

```
**AI score:** 78 → 31 (likely AI → likely human)
**What changed:** varied sentence rhythm, removed 42 stock phrases ("delve", "it is important to note"…), replaced generic claims with the specifics already in your text, added contractions where the register allows.
**Facts check:** passed — all 37 numbers, 12 citations and 4 quotes preserved, nothing added
**File:** report_humanized.docx (same headings and length, ~6,200 words)
**Worth a manual look:** section 4 (still reads formal — it's a methods section, which is fine)
```

Plus the before/after heatmap and report paths. Remind them in one line that detector scores are estimates and that they should read it through to make sure it sounds like them.

## Scripts reference

| Script | Purpose |
|---|---|
| `extract_text.py FILE -o OUT` | PDF/DOCX/TXT/MD → plain text (keeps headings as `#` lines for DOCX) |
| `chunk_text.py FILE --out DIR --words N` | Split at paragraph boundaries into ~N-word chunks + manifest |
| `ai_score.py FILE [--deep] [--report MD] [--json J] [--compare J] [--html H]` | Calibrated AI-likeness score with explanations, pattern signals, sections, flagged sentences |
| `html_report.py FILE -o OUT [--deep] [--compare ORIGINAL]` | Sentence heatmap; side-by-side before/after with `--compare` |
| `perplexity.py FILE` | GPT-2 perplexity features on their own (used by `--deep`) |
| `neural.py FILE` | Neural detector features on their own: fakespot, desklib, Binoculars (used by `--max`) |
| `verify_rewrite.py ORIGINAL REWRITE` | Fact-preservation check; exit code 1 on missing/added facts |
| `style_profile.py SAMPLES… [-o J]` / `--profile J --check FILE` | Voice profile from the user's own writing; match score for a rewrite |
| `build_output.py DIR --format docx\|md\|txt -o OUT` | Join `*.human.txt` chunks (falls back to originals if missing — and warns) |
| `model.json` | Trained scorer weights (regenerate with `dev/calibrate.py --write` in the repo) |

Dependencies: `pypdf` and `python-docx` (`pip install pypdf python-docx`). Deep mode adds `torch` and `transformers`. Everything else uses the standard library.
