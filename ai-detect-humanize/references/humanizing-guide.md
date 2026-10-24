# Humanizing Guide

What makes text read as AI-written, and how to rewrite it so it reads as a person wrote it. Read the whole thing once before a rewrite; it's short.

## Contents
1. The patterns that give AI text away
2. Rewriting techniques
3. Register-specific notes
4. Before/after examples
5. Final checklist

---

## 1. The patterns that give AI text away

This is the short version. The full catalog — sentence templates, grammar habits, paragraph and document structure, formatting, tone, genre-specific tells, and the tics models fall into when told to "sound human" — is in `ai-patterns.md`.

**Rhythm**
- Sentences all 18–25 words long. Paragraphs all 3–4 sentences. Everything the same shape.
- Every paragraph opens with a topic sentence and closes with a mini-summary.

**Stock vocabulary** (each fine alone; in clusters they're a fingerprint)
delve, tapestry, testament, realm, landscape, intricate, multifaceted, pivotal, crucial, vital, robust, seamless, leverage, utilize, foster, underscore, showcase, navigate (figuratively), embark, unlock, unleash, harness, elevate, empower, holistic, synergy, paradigm, nuanced, meticulous, comprehensive, ever-evolving, dynamic, cutting-edge, game-changer, vibrant, bustling, myriad, plethora.

**Stock phrases**
"It is important to note that", "It's worth noting", "In today's fast-paced world", "In the ever-evolving landscape of", "plays a crucial role in", "a wide range of", "Whether you're X or Y", "When it comes to", "Not only… but also", "serves as a testament to", "In conclusion", "In summary", "Overall,", "Ultimately,", "At the end of the day", "This highlights the importance of", "Let's dive in", "Here's the thing".

**Structure tics**
- Sentence-initial transitions everywhere: Furthermore, Moreover, Additionally, However, Consequently, Notably.
- Lists of exactly three ("efficient, scalable, and secure") again and again.
- "It's not just X — it's Y." / "X isn't about A. It's about B."
- Balanced hedging: "While X has benefits, it also presents challenges."
- Em dashes used as the default punctuation.
- Conclusions that restate the intro without adding anything.
- Rhetorical question followed immediately by its answer.

**Voice**
- No contractions in a register where a person would use them.
- No concrete detail — claims stay general ("many experts agree", "significant impact").
- Uniformly positive, enthusiastic, or diplomatic tone. Nobody is ever wrong, nothing is ever boring.
- No stance. The writer never commits to an opinion.

## 2. Rewriting techniques

Use these in combination; one alone isn't enough.

1. **Vary sentence length deliberately — in both directions.** Mix a 5-word sentence with a 35-word one that takes a clause or two to land. Let one paragraph be a single sentence if it earns it. The common mistake is only adding short sentences: human writing in the calibration data runs *longer* on average than AI text, so a page of punchy fragments reads as AI. Combine as often as you split.
2. **Cut the scaffolding.** Delete "It is important to note that", "In conclusion," and most sentence-initial transitions. The logic usually survives without them; where it doesn't, use a plain connector (but, so, and, still, yet) or restructure.
3. **Repeat words naturally.** Don't rotate synonyms to avoid using a word twice ("the company… the firm… the organization"). Models vary vocabulary more than people do; a person just says "the company" again.
3b. **Swap inflated words for plain ones.** utilize → use, leverage → use/build on, facilitate → help, crucial → important (or just say why it matters), delve into → look at/dig into, robust → solid/reliable (or the specific property).
4. **Make it concrete.** Replace general claims with the specifics already in the source: the number, the name, the example. Don't invent facts — if no specific exists, a plainer, shorter claim is better than an inflated one.
5. **Break the rule of three.** Use two items, or four, or one with a reason.
6. **Commit.** Where the original hedges both ways, keep the writer's actual position if one is implied ("This approach is slower, but for most teams the trade-off is worth it").
7. **Use contractions** where the register allows (not in formal academic or legal prose).
8. **Vary paragraph openings.** Start with the subject, a clause, a number, a short fact — not "Additionally" five times.
9. **Endings: stop when you're done.** Replace summary conclusions with a final point, implication, or next step.
10. **Punctuation variety.** Trade most em dashes for commas, parentheses, colons, or full stops.
11. **Allow a little texture.** An aside, a mild opinion, a sentence that starts with "And" or "But". A human-sounding text has small asymmetries.

## 3. Register-specific notes

- **Academic / thesis:** Keep formal tone, citations, passive voice where conventional (methods sections). Focus on rhythm, removing stock phrases and empty transitions, and making claims precise. No contractions, no jokes.
- **Business / report:** Short sentences, active voice, numbers up front, plain verbs. Kill buzzwords hard.
- **Blog / marketing:** Contractions, direct address ("you"), short paragraphs, a point of view. Avoid hype adjectives.
- **Personal statement / cover letter:** First person, specific anecdotes from the source, modest confidence. Remove superlatives ("passionate", "deeply committed") in favour of what the person actually did.
- **Technical docs:** Clarity beats style. Mainly remove filler and fix rhythm; don't touch code, commands, or terms.

## 4. Before/after examples

**Example 1 — generic essay**

Before:
> In today's rapidly evolving digital landscape, artificial intelligence plays a pivotal role in transforming industries. Furthermore, it is important to note that AI not only enhances efficiency but also fosters innovation. Ultimately, organizations that leverage AI will be well-positioned to thrive.

After:
> AI is changing how a lot of industries work. Some of that is plain efficiency: tasks that took a team a week now take an afternoon. The more interesting part is what companies do with the time they get back. The ones experimenting now will have a head start.

**Example 2 — academic**

Before:
> This study delves into the multifaceted impact of social media on adolescent mental health. Moreover, the findings underscore the crucial importance of digital literacy programs.

After:
> This study examines how social media use relates to adolescent mental health across three measures: anxiety, sleep quality and self-esteem. The results point to digital literacy programs as a practical intervention.

(The three measures came from the source's methods section — specifics pulled forward, nothing invented.)

**Example 3 — business**

Before:
> Our robust and scalable solution seamlessly integrates with your existing workflows, empowering teams to unlock their full potential.

After:
> It plugs into the tools your team already uses, so nobody has to change how they work to get started.

## 5. Final checklist (per chunk)

- [ ] Every fact, number, name, citation and quote from the original is still there and unchanged
- [ ] Headings, lists and paragraph order preserved
- [ ] Length within ~15% of the original
- [ ] Sentence lengths visibly vary; no run of 3+ same-length sentences
- [ ] No more than one sentence-initial transition word per paragraph
- [ ] Stock words/phrases from section 1 removed or justified
- [ ] Voice matches the voice notes and the previous chunk
- [ ] Participial tails (", ensuring…") and contrast frames ("not X, but Y") mostly gone
- [ ] Paragraphs don't all end in a wrap-up sentence
- [ ] None of the "sound human" tics from ai-patterns.md §9 crept in (fragment runs, "Honestly?", extra em dashes, invented details)
- [ ] Reads naturally aloud — no slang, typos or forced quirks added
