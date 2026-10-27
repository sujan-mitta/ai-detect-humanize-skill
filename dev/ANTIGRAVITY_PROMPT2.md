# Task for Antigravity (round 2): Gemini texts with DIFFERENT prompts, for an honest held-out test

## Context
Round 1 (`dev/corpus/gemini/`) was used to train a detector, so it can no longer be used to
test it. Round 2 must differ from round 1 in prompt wording and in topics, so the test is fair.
You only generate and save texts. Do not run or edit any code, and do not look at or reuse
anything under `dev/corpus/gemini/` (neither its topics nor its prompts).

## What to do
Create `dev/corpus/gemini2/`. For EACH Gemini model available to you (use the same models as
round 1 if you can, and ALSO any other Gemini versions you can reach, e.g. an older or newer
Flash/Pro, or a "thinking" variant), make one folder named `<model>__<condition>`.

Conditions (one folder per model per condition):
- `casual`: a short, messy, realistic user request, e.g. `can u write something about <topic> for my <context>` (vary the wording every time; do NOT use the phrase "About 300-400 words"; let the model choose the length, but if it is under 150 words ask once for a longer version)
- `persona`: `You are <a specific kind of person, e.g. a retired nurse, a grad student, a small-shop owner>. Write <genre description> about <topic> in your own voice.`
- `edit`: first write a short rough human-style draft yourself (3-5 sentences, with ordinary imperfections), then ask the model: `Improve and expand this into a polished piece.` and save ONLY the model's output.

Genres (4 texts per genre per folder = 48 texts per folder; file names `<genre>__01.txt` ... `__04.txt`):
blog, email, essay, news, story, technical, report, review, academic, forum, qa, encyclopedic
(same genre list as round 1, but write the genre description yourself in different words).

## Rules
1. A new, unusual topic for every text. No topic may appear twice anywhere in `gemini2/`, and none from round 1.
2. One fresh, separate request per text, with no earlier outputs in context.
3. Save exactly what the model returned as UTF-8 plain text. Remove only a leading "Sure, here is..." or trailing "Would you like...?" lines. Do not edit anything else.
4. Do not use any detector or humanizer tool.
5. Write `dev/corpus/gemini2/MANIFEST.md`: per folder, the exact model name and version, how you accessed it, date, word-count range, and anything unusual.
6. If you cannot finish everything, finish whole folders in this order: (strongest model, casual), (fastest model, casual), (strongest, persona), (fastest, persona), then `edit` folders, then any extra models.

When done, reply with the folder list, file counts and the MANIFEST contents. Do not commit or push.
