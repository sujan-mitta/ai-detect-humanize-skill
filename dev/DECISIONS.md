# Decision log

## 2026-10-26: external data hurt the style tier

Leave-one-genre-out (LOGO), 15 genres incl. 3 external benchmark sources held out whole:

| Config | style AUC | template-only AUC | style: AI caught at 2% human FPR |
|---|---|---|---|
| ours only (12 genres) | 0.959 | 0.973 | 70% |
| ours + 3,000 COLING-train texts (39 generators) | 0.784 | 0.741 | 25% |

Per-genre recall on OUR genres fell too (academic 92->36, literary 54->22, technical 27->10), and held-out external sources were still hard (x:mage 5%).

**Diagnosis.** Style cues are generator-specific. Modern chat LLMs (our corpus: Claude-family + ChatGPT-style prompt) are fluent, hedged and template-y; the benchmark generators include GPT-2-class, OPT, BLOOM, flan-t5, whose text is repetitive/incoherent in different ways. The two distributions put the same features on opposite sides, so a single style model cannot fit both and is worse on each. GPT-2-predictability and the neural classifiers (deep AUC 0.950, max 0.964) are less affected because they measure predictability, not style conventions.

**Decision (gate 4: ablate before keeping).** Style and template models are trained on modern-LLM text only (ours). External data is NOT used for the style/template tiers. It stays as an external test set. The deep/max tiers may later add a feature-complete subset of external rows if the external evaluation shows a large generalisation gap on modern generators.

## Shortcuts found and removed (gate 3)
- Typography: curly quotes 140 human/0 AI; double space after stop 163/0 -> `canon_typography`.
- Layout/length: within-genre AUC of paragraph count 0.83, word count 0.78 -> removed paragraph-layout features, 250-word window for repeated 4-grams, 200-word rate floor. Previous model moved 53 points when an AI text was merged into one block.
- Pre-tokenised human text (HC3: "word , word .") -> detokenisation in `canon_typography`.
- Binoculars was implemented with models swapped: AUROC 0.48 -> 0.79 after the fix (0.84 on the full corpus).

## Open weaknesses (LOGO, max tier)
- technical writing recall 31%, AI fiction 36%, "casual"/humanized AI ~55%.
- Human government-report prose flagged 16% at the verdict threshold; reviews 8%.

## 2026-10-26: weak-spot data (held aside until CHAIN2 finishes)
Added 100 more GAO/CRS report passages and 60 IETF RFC passages (human) as `dev/corpus/human_plus.jsonl`. Matched AI texts (same topics, Claude/gptstyle prompts) to be generated; only adopted if grouped/LOGO numbers for technical/report improve without regressing others (gate 4). Held out of human.jsonl while CHAIN2 runs so the grouped run uses the same corpus as the LOGO run.

## 2026-10-26: ours-only retrain (v1) results, LOGO
style AUC 0.960 (AI caught 70% at 2% human FP), deep 0.962 (76%), max 0.970 (81%); template-only 0.972. Recovered from 0.784 after reverting external data -> confirms the earlier diagnosis.
Weakest: technical 23-27%, literary 52-60%, report 37-76%. Human FP at verdict: academic 8-10%, review 6-8%, report 0-6%.
Invariance (plain prose): worst mean change 0.6 pts (goal <2), 0% verdict flips.
External (COLING dev + RAID, mostly legacy open models): style AUC 0.57, deep 0.69, max 0.92 (27% caught at shipped threshold, human FP 1%). Style/template tiers target modern chat LLMs and fail on GPT-2/OPT/LLaMA-1/BLOOM-class output; max tier (Binoculars etc.) generalises. Report honestly.
Grouped run crashed on a print (GENRES None) - fixed.

## v2: adding 160 human (GAO/CRS reports, RFC) + 160 matched AI (Haiku/Sonnet/Opus) -> backups: runs/model_ours_v1.json. Adopt only if technical/report recall rises without raising human FP elsewhere.

## v2 adopted (gate 4 passed): +160 human (GAO/CRS, RFC) +160 matched AI
LOGO max: AUC 0.970->0.972; AI caught at 1% human FP 73%->79%, 0.5% 68%->74%; report recall 76->88%, technical 27->35%; human FP: academic 10->3%, report 6->5%, encyclopedic 5->10% (worse), review 6%. Style: AUC .960->.969, technical 23->37, report 37->69. Technical remains weakest (35%). Grouped protocol (genres seen): max AUC 0.998, 98% AI caught at 1% FP; optimistic since topics/prompt styles are shared - report as upper bound.

## v3 (weak-spot round)
- Technical: +98 human tutorial/doc passages (CPython 3.9, Linux 5.4, Rust book 2019, Pro Git) + 98 matched AI. LOGO technical recall: style 37->65%, max 35->59%; overall max AUC 0.972->0.975, AI caught at 1% FP 79%->77% (noise-level; report FP unchanged at 6%).
- Legacy head (predictability-only, ours + 300 COLING-train texts, max-combined = max(modern, rescaled legacy)): external 316 texts 30%->79% AI caught, human flagged 2.5%->3.2%; RAID (unseen) 88%/0%; own corpus 84%->83% caught, 1.9%->1.7% flagged. Gate passed.
- Not fixed: formal human prose FP (5-10%), fiction (~70%), casual/humanized (~76%), style/deep modes on legacy generators.

## v4 (second weak-spot round) - adopted
Added human: 100 WikiText articles, 78 arXiv abstracts, 100 Yelp, 80 GAO, 100 WritingPrompts stories. Added AI (matched topics): 100 fiction, 100 encyclopedic, 78 academic, 100 reviews, 80 reports, plus 117 casual/natural rewrites of AI texts.
LOGO max: AUC 0.975->0.978; AI caught at 1% FP 77->83%; verdict 84->86% at 2%; human FP encyclopedic 5->1%, academic 7->3%, review 6->3%; fiction 70->77%; technical 59->63%; cost: AI encyclopedic 98->80% (formal human data raised the bar), report FP 6->7%.
Legacy head retrained: external 316 -> 75% caught at 1% human FP (RAID 83%/0%); own corpus 86->85% caught.

## v5 (technical docs + 120 more GAO reports + 233 matched AI) - NOT adopted; v4 stays shipped
LOGO: style tier improved (AUC .974->.976, AI caught at 1% FP 74->78%, technical 72->80%), but max tier fell: AI caught at 1% FP 83->80%, 0.5% 78->75%, verdict threshold rose 85->91 (harder formal negatives), so at the shipped threshold the benchmark recall fell 75->49% (human FP 1->0%; AUC actually rose .963->.973). Per-genre max recall fell for academic/email. Decision: gate failed on the recommended mode; model.json = v4. The v5 data stays in the corpus (dev/runs/model_v5.json keeps the model). Report FP stays 7% - more of the same GAO data does not move it; it looks like a real overlap between formal human prose and AI.
Held-out end-to-end (36 fresh AI texts, never trained on): v4 max flagged 36/36 (detection recall 100%). Humanizing: one pass 98->97, 0/36 human; iterative structure edits 98->70/78, 4/12 out of the AI band.
Scorer now prints a log-odds margin when the score saturates (>=90) so rewrite rounds can be tracked.

## Unseen-genre false-positive test (shipped v4 model, max mode)
160 human texts from genres never in training: legal (CRS bill summaries) 60, statute text 40, medical textbooks 60. Flagged as likely AI: 0 of 160 (0%); read as human: 97-98%. Error analysis (dev/error_analysis.py) shows the leave-one-genre-out "report 7%" is a held-out-genre artefact driven by the sentence-template score; with the genre in training, report false positives are 0-1%. Conclusion: more of the same data is not the lever; breadth is, and breadth of human genres already looks good.

## Frozen-split comparison (v4 data vs v4+v5 data), identical held-out groups, 5 random splits, max tier
AUC 0.999 vs 0.999; AI caught at 1% human FP 97.6% vs 97.9% (paired diff +0.3 pts, sd 0.2); technical 95.6 vs 95.1; report 98.7 vs 98.3. The v5 data changes nothing when genres are seen; the earlier +-3 point LOGO movements were test-set composition and threshold noise. Shipped model stays v4 (smaller, no loss). Lesson: compare on a frozen test set; LOGO numbers are only comparable between runs that hold out the same rows.

## Sentence-target tool for rewrites (scripts/rewrite_targets.py)
Same six held-out texts, two models, --max scores. Iterating blind: mean 98->70 (Opus) / 78 (Sonnet), 4/12 under 60. Iterating with the target list: 60 / 72, 6/12 under 60, best case 41. Verifier: trailing punctuation inside quotes no longer counts as a changed quote; remaining flags are digits for spelled-out counts and changed quote boundaries, left for the writer to confirm.

## Family test and legacy head v2
- family_eval.py on held-out COLING-train texts: GPT-4/4o 100, GPT-3.5 89->96, Mixtral 93, Cohere 86->79, Llama-3 68, Gemma 57->61; humans 0/90 at verdict.
- Legacy head retrained with +180 rows (Llama-3, Gemma, human): external 74->78.5% caught at 1.3% flagged. Previous model: dev/runs/model_v4_pre_extra.json.
- restructure.py (mechanical clause splitting): 98->96 in max mode; removed.

## Gemini
- 240 Gemini Pro/Flash texts (dev/corpus/gemini, gitignored?). Untrained model flagged 65-72%. Legacy head retrained with files 01-03 (144 rows): held-out 04-05 flagged 96%. Family test humans over 0.60: 4%->6%, 0/90 over verdict. Previous model: dev/runs/model_v4b_pre_gemini.json.
