# Handoff: state when the usage limit was reached

Read `dev/PLAN.md` first (goals, anti-memorisation gates, phases).

## Do NOT yet
- Do **not** package or push. `skills/humanize/scripts/model.json` is still the **older** v5 model (trained on uncleaned typography, no business genres). README/SKILL.md numbers describe that older model.
- Nothing from this session's recent work is committed (last commit: 419dfe0 on main, which is also what is on GitHub).

## What changed in the working tree since 419dfe0 (uncommitted)
- `ai_score.py`: typography canonicalisation in `prose_blocks`, Unicode/homoglyph `sanitize`, short-text abstain (<100 words), displayed-score remap (`display`), learned template feature, humanizer-fingerprint features.
- `neural.py`: Binoculars orientation fixed (AUROC 0.48 -> 0.79); `perplexity.py`: one-pass features + Fast-DetectGPT.
- `templates.py` (new), `verify_rewrite.py` (`--semantic`, negation warning), `extract_text.py` (bold kept in docx).
- `dev/calibrate.py`: leave-one-genre-out evaluation, recall by AI type/genre, fixed-FPR tables, version-stamped caches. `dev/build_business_corpus.py`, `dev/fix_business_rows.py`, `dev/build_external.py`, `dev/neural_features.py` (stale-aware), `dev/PLAN.md`.

## Data state
- Human: 4 new genres (email 56, report 49, review 50, post 50); layout repaired.
- AI on the same topics written by 3 models: ~209 files (`dev/corpus/ai/*/{email,report,review,post}__*`).
- Typography shortcuts found: curly quotes 140 human / 0 AI; double space 163 / 0. Fixed by `canon_typography`. 464 cached rows were stale.

## Background jobs that may still be running on the machine
1. `python -u dev/neural_features.py` (log `dev/runs/neural_pass1.txt`): recomputes fakespot/desklib/Binoculars for stale + new rows. It is resumable. **Re-run it once more after it finishes** to pick up the AI business texts written after it started.
2. `python -u dev/build_external.py` has FINISHED. Result:
   - `coling_eval.jsonl`: 220 texts (110 human / 110 AI), 38 generators incl. GPT-4o, Llama-3, Gemma-2, Mixtral, OPT, BLOOM, GLM. Usable as the external generalisation test.
   - **`raid_eval.jsonl` / `raid_pool.jsonl` are UNUSABLE: they contain only AI texts (88 and 274) and zero human texts.** Cause: in RAID's train split the human rows are not spread through the file, so 260 random offsets hit none of them (and only 700 of 10,000 sampled rows were un-attacked). Fix before using RAID: fetch human rows from the START of the split (offset 0 upward; the first rows are `model == "human"`, `attack == "none"`), then fetch AI rows whose `source_id` matches those humans (or sample AI rows separately by domain/model) and keep eval/pool disjoint by `source_id`. Until then evaluate on COLING only.

## Next steps, in order
1. Wait for / re-run `dev/neural_features.py` until it reports 0 texts to compute.
2. `python -u dev/calibrate.py --write > dev/runs/logo_calibration.txt` (recomputes GPT-2 features for all rows first, ~30 min; then leave-one-genre-out results, thresholds, per-type/genre recall).
3. Phase 1 of the plan: score the new model on `dev/corpus/external/*.jsonl` (needs a small scorer script: style + deep + max per text, report TPR at 5/1/0.5% FPR per generator and domain).
4. Phases 2-5 per `dev/PLAN.md`; keep a decision log in `dev/runs/decision_log.md`.
5. Regenerate README/SKILL.md numbers from the final run only, repackage (`package_skill`), commit locally. User pushes.

## Known weaknesses to report honestly
- Genres with no human data (cover letters, LinkedIn, personal statements): recall only, FP unknown.
- Under ~150 words any mode is unreliable.
- The AI side is still mostly Claude-family text plus a ChatGPT-style prompt; external benchmarks are the only generator-diversity evidence.
