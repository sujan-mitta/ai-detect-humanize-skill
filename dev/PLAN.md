# Improvement plan: make the detector generalise, not memorise

## Where we are and why it disappoints

| Symptom | Root cause (evidence) |
|---|---|
| CV accuracy looks great (96-98%) but real documents disappoint | CV mixed genres, authors and rewrites in every fold, so it measured memorisation. Leave-one-genre-out exposed a 15-23 point recall drop on new genres. |
| Models read formatting, not writing | Curly quotes: 140 human / 0 AI texts. Double space after a full stop: 163 / 0. Paragraph layout of human emails/posts was destroyed by the source. |
| Blind spots | No human data for emails, reports, reviews, posts, so false positives there were unmeasurable. |
| Narrow AI side | Only Claude models + one "ChatGPT-style" prompt. Real inputs come from GPT-4/5, Gemini, Llama, Mistral, DeepSeek... |
| Narrow human side | ~15 sources, many passages per author. |
| Small data | ~1.4k texts; deep/max tiers have fewer usable rows. |
| Evasion | Humanizers and paraphrasers shift exactly the surface cues that detectors use. |

## Rules every change must pass (the anti-memorisation gates)

1. **Name the pattern.** Before adding a feature or data source, write one sentence on the linguistic or statistical regularity it captures and why that regularity should hold for text we haven't seen.
2. **Unseen-everything evaluation.** Report on data the model never touched, split by the thing that must generalise: genre (leave-one-genre-out), generator (leave-one-generator-out), source/author (grouped), request style (leave-one-humanizer-request-out), and finally an *external* benchmark I did not write (RAID, COLING-MGT, MAGE).
3. **Shortcut audit.** For any property that should not matter (length, paragraph count, markdown, digits, typography, topic), check it does not separate the classes on its own; neutralise it if it does.
4. **Ablate before keeping.** A feature/model stays only if it improves the *out-of-distribution* number, not just CV. Complexity that doesn't generalise gets deleted.
5. **Fixed-FPR reporting.** Quote TPR at 5% / 1% / 0.5% human false-positive rates with confidence intervals; never accuracy at a 50% cut-off. False accusations are the costly error.
6. **Document what failed.** Negative results go in the README, with the reason.

## Roadmap

**Phase 0: finish the corrected baseline** (running)
- Typography canonicalised once in the shared reader; stale caches recomputed.
- Human emails (Enron/AESLC), reports (GAO/CRS), reviews (Yelp), posts (Reddit), plus AI texts on the same topics.
- Leave-one-genre-out evaluation, recall by AI type and genre, fixed-FPR tables.
- Success: honest numbers; no genre has a false-positive rate above ~8% at the verdict threshold.

**Phase 1: external reality check**
- Score the current model on RAID (many generators/domains) and COLING-2025 MGT: data I neither wrote nor curated.
- Success: a measured out-of-distribution TPR at fixed FPR per generator and domain; a ranked list of failure modes.

**Phase 2: widen the data**
- Train on external multi-generator data; keep generators and domains out for evaluation (leave-one-generator-out, leave-one-domain-out). Features expensive to compute (GPT-2, neural) on a subset; cheap ones on all.
- Success: unseen-generator TPR at 1% FPR improves; no regression on our LOGO numbers.

**Phase 3: model design for generalisation**
- Compare, under the OOD protocol: stacked boosted trees vs. a simple average of independent detectors (style, templates, neural, Fast-DetectGPT, Binoculars); embeddings as a probe (topic-leak check). Pick what generalises, not what fits.
- Per-length calibration + abstain band; sentence-level scoring for mixed text.
- Success: smaller gap between CV and external numbers.

**Phase 4: robustness**
- Adversarial/humanized data from more request styles; leave-one-request-style-out; homoglyph/zero-width defence (done).
- Success: humanized-text recall at the verdict threshold, out-of-fold.

**Phase 5: humanizer skill**
- Re-test end-to-end after the detector changes; make sure rewriting advice targets patterns that survived the OOD tests (structure, specificity), not surface tells.

**Phase 6: honest write-up**
- README/SKILL.md numbers regenerated from the final run only; weaknesses stated; packaged; committed locally. **No push**: the user pushes.

## Decision log
Updated in `dev/runs/decision_log.md` as results arrive.
