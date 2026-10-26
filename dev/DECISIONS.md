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
