# Task for Antigravity: generate Gemini-written test texts for an AI-text detector

## Context
I am testing an AI-text detector (a Python tool). It is already tested on Claude, GPT, Llama,
Gemma, Mixtral and Cohere output. It has NEVER seen Gemini output. Your job is to produce
Gemini-written texts as files. You only generate and save texts. Do not run or edit any code.

## What to do
Work in this repo folder: `dev/corpus/gemini/` (create it).

For EACH Gemini model available to you (try as many as you can; at minimum the strongest
and the fastest, e.g. Gemini Pro and Gemini Flash, whatever the newest versions are), and
for EACH of the two conditions below, create one folder named
`<model>__<condition>` and put the texts inside.

Examples: `gemini-pro__plain/`, `gemini-pro__human/`, `gemini-flash__plain/`, `gemini-flash__human/`
(lowercase, no spaces; use the real model name/version you used).

### Conditions
- `plain`: ask the model exactly as a normal user would, with no mention of detection or style.
  Prompt template: `Write a <genre description> about <topic>. About 300-400 words.`
- `human`: same, but add this sentence: `Make it sound like a real person wrote it, not an AI.`

### Genres (5 texts per genre per folder = 60 texts per folder)
File names: `<genre>__01.txt` ... `<genre>__05.txt`

| genre (file prefix) | genre description for the prompt |
|---|---|
| blog | a personal blog post |
| email | a work email to a colleague |
| essay | a reflective personal essay |
| news | a local news article |
| story | a short story |
| technical | a technical explanation for developers |
| report | a formal report section about a public policy topic |
| review | a product or restaurant review |
| academic | an academic-style paragraph summary of a research area |
| forum | a forum post asking for or giving advice |
| qa | a detailed answer to a how/why question |
| encyclopedic | an encyclopedia-style article section |

### Rules (important, they decide whether the test is valid)
1. Use a DIFFERENT topic for every text. Pick varied, ordinary topics yourself. Do not reuse topics across files or across folders.
2. One fresh, separate request per text. Do not generate several texts in one chat reply, and do not give the model earlier outputs as examples.
3. Save exactly what the model returned, as plain text in UTF-8. Do not edit, shorten, polish or rewrite it. If the model adds a preamble such as "Sure, here is..." or a closing offer, delete ONLY those lines.
4. Keep the model's own formatting (headings, bullets) if it produced them. Do not add formatting.
5. Length: each text should be 250-450 words. If one is under 150 words, regenerate it once.
6. Do not use any detector or "humanizer" tool yourself. The point is raw model output.
7. Write a file `dev/corpus/gemini/MANIFEST.md` listing, per folder: the exact model name/version, date, how you accessed it, and anything unusual (e.g., a model refused or you could only do some genres).

If you cannot finish everything, finish whole folders (all 12 genres) in this priority order:
1. strongest model, `plain`
2. fastest model, `plain`
3. strongest model, `human`
4. fastest model, `human`
A smaller complete set beats a large partial one.

## When you are done
Reply with: the list of folders created, number of files in each, and the contents of MANIFEST.md. Do not commit or push anything.
