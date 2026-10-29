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

## False-positive creep, deep legacy head
- Human texts above the 60 line (external, 158): 7.0% (v4) -> 8.2% -> 10.8% after the Llama/Gemma and Gemini retrains. Cause: the legacy head's rescaled score lifted humans into the 60-85 band. Fix: legacy only adds when its own score >= 80 (gate in model.json): 5.1%, verdict-level results unchanged, Gemini held-out 96% -> 88% above 60, 78% at verdict. Correction: earlier '96% Gemini' was the 60 line, not the verdict.
- legacy_deep (GPT-2 features only, gate 80): deep tier on older-model benchmark AI caught 19% -> 44%, held-out humans flagged 3.6% -> 4.8%.
- Not addressed: short text (<100 words), technical/government prose data, humanizer ceiling. Round-2 Gemini prompt written (dev/ANTIGRAVITY_PROMPT2.md) for a fair held-out test.

## Round 7: short text, author details, technical/report data
- Short text (max mode, whole-sentence chunks): 60-80 words AUC 0.91, 80-99 AUC 0.90, 35-60 AUC 0.79. Rule: --max gives a tentative flag at 60-99 words, never clears; else abstain.
- Author details (AI-simulated authors): 98 -> 95 (0/12 under 60); details + structure method 98 -> 77 (3/12). No better than structure alone. The claim that author details are the reliable fix was wrong and removed. author_questions.py kept as a quality aid.
- 120 new AI technical/report texts (sonnet7, haiku7). Frozen split: base test unchanged (AUC 0.998 both), report AI +3.4 pts, new-style report AI 68 -> 92 (optimistic, same sources), new technical already 100.
- Full retrain (v7, 3717 texts): LOGO AUC 0.977 (same). But verdict threshold rose 85 -> 90, so at the verdict external AI caught 78 -> 66%, Gemini heldout 78 -> 71%, family results equal or lower; fresh humans over 60 went 1.2 -> 3.6%. NOT ADOPTED. Shipped model stays the previous one (dev/runs/model_v4c_pre_v7.json). v7 kept as dev/runs/model_v7_not_adopted.json. The new texts remain in dev/corpus/ai/{sonnet7,haiku7} for a future retrain.

## Round 8: robustness pass
- Edge-case run of the scorer (empty, blank, 1 word, BOM, Windows-1252, CRLF, emoji, no punctuation, missing file, .doc). Found and fixed: missing file printed a Traceback (now "file not found"); a UTF-8 BOM stayed in the text; Windows-1252 files were decoded with replacement characters (now cp1252 fallback); .doc gave no explanation (now says to save as .docx).
- dev/smoke_test.py: 12 fast checks (style tier, no downloads). All pass.
- No model change this round. Remaining accuracy gaps are unchanged and listed in the README (Llama-3/Gemma, technical prose, humanizer ceiling, Gemini round 2 needs the user to run ANTIGRAVITY_PROMPT2.md).

## Round 9: v7 adopted with a pinned verdict threshold
- dev/regression_floor.py (cached features, seconds): floors on external, Gemini, fresh humans and per-family recall. The rejected v7 failed it, the previous model passed.
- Finding: v7 was rejected only because the 2% held-out-genre quantile moved the verdict threshold 85 -> 90. At 85 v7 beats the previous model on every outside benchmark at equal human false positives (external AI 77.8 -> 80.4%, Gemini 78.1 -> 80.2%, Llama-3 65 -> 69%; external humans 1.3%, fresh humans 0%; unseen-genre humans 0/160).
- Legacy gate swept 60-85: no effect at the verdict, only on the 60 line; kept 80. (85 would give fresh humans over 60 of 1.2% but Gemini over 60 of 86.5%.)
- calibrate.py: VERDICT_AI_CAP = 85. LOGO max at 85: AUC 0.977, 85% AI caught, 3% humans flagged (report 9%, academic 4%, others <=2%).
- Adopted: model.json = v7 with threshold 85, gate 80. Previous: dev/runs/model_v4c_pre_v7.json.
- Llama-3/Gemma: all 47 texts of each generator in the COLING train file are already cached and used; no more data of that kind is available here.

## Round 9b: invariance and a rewrite-from-facts humanizer trial
- Invariance on the adopted model (60 texts, 8 formatting edits): worst mean change 0.3 points, 0% verdict flips (runs/invariance_v8.txt).
- Rewrite-from-facts method (write a fact list, set the original aside, write fresh under structure rules, then up to 3 target rounds), 12 fresh texts, max mode: mean 98 -> 80, 1 of 12 under 60 (essay 65/59, news 60/60 are borderline), blog 96, story 92, technical 88. Caveat: five of six agents were stopped before finishing their rounds because six parallel max-mode scorers starved the CPU, so this under-measures the method. Even so it is clearly behind iterating on the original with the target list (mean 60-72, 6 of 12 under 60), so the guide is unchanged.
- Practical note: parallel agents each running --max slow everything to a crawl and trigger memory-allocation failures; run max-mode scoring one at a time.

## Round 10: fair Gemini test (new prompts) and legacy-head retraining, not adopted
- 192 new Gemini Pro/Flash texts (casual and persona prompts, different from the round-1 templates). Shipped model: casual 75% above 60 / 50% at verdict; persona 54% / 26%. The earlier 80-92% Gemini figure only held for the training templates.
- Retrained the legacy head with these rows: casual-only, persona-only, both. Cross-condition (train one style, test the other): above 60 casual 75 -> 78-83%, persona 54 -> 67%; but verdict-level recall fell (casual 50 -> 38-39%, persona 26 -> 21-22%), external AI caught 80 -> 62-64%, gemini round-1 80 -> 74%, GPT-3.5 92 -> 85-88%, fresh humans over 60 up to 4.8%. Every candidate fails regression_floor.py. NOT ADOPTED; shipped model unchanged. Candidates: dev/runs/model_g2_{casual,persona,casual_persona}.json.
- Conclusion: casually-prompted modern Gemini output is mostly indistinguishable on these signals; more training rows of this kind trade away other recall. README and SKILL.md now say so.
- Scripts: dev/gemini2_features.py, dev/gemini2_eval.py, legacy_head.py --gem2/--out.

## Round 11: false positives on real human reports (80 pre-2021 web texts), retrain not adopted
- dev/corpus/human_web (40 report, 40 technical; manifest inside) scored with the shipped model, max mode: reports 10% at verdict / 20% above 60, technical 2.5% / 5%. A real weak spot beyond the 9% leave-one-genre-out estimate.
- Candidate trained with the even-numbered half added as humans, tested on the odd half (n=20 per genre): report 10 -> 5%, technical 5 -> 0% at verdict, human external 0%; but external AI at verdict 80 -> 70%, Gemini round-1 80 -> 64%, GPT-3.5 92 -> 85%. Fails regression_floor.py. NOT ADOPTED (dev/runs/model_web_half.json). human.jsonl restored; shipped model unchanged.
- dev/humanweb_eval.py scores any model on these texts (--held-odd for the odd half).
