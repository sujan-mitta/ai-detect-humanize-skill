# The AI Writing Pattern Catalog

A field guide to how large language models write — written from the inside. These are the habits models like Claude and ChatGPT fall into by default: the words they reach for, the sentence shapes they repeat, the way they build paragraphs and whole answers, and the tone they settle into.

Use it two ways:
- **Detecting:** audit a text against every section, not just the vocabulary list. Vocabulary is the easiest tell to fix and the least reliable on its own; structure and rhythm give AI text away even after the buzzwords are gone.
- **Humanizing:** check your own rewrite against it, especially section 9. You are a language model too; when you rewrite, you will reintroduce these patterns unless you watch for them.

A single pattern proves nothing — humans use em dashes and say "crucial". What gives text away is **density and co-occurrence**: five or six of these habits stacked in every paragraph.

## Contents
1. Vocabulary
2. Sentence-level templates
3. Grammar and punctuation habits
4. Rhythm
5. Paragraph structure
6. Whole-document structure
7. Formatting habits
8. Tone and stance
9. Second-order tells: what AI does when told to "sound human"
10. Genre-specific tells (essays, emails, creative writing, academic, marketing)
11. Quick audit checklist
12. What the calibration data showed (newer models write differently)
13. How AI "humanizes" text, measured
14. Which signals are strong and which are weak

---

## 1. Vocabulary

Why it happens: models pick high-probability words that sound polished and safe in many contexts. Mid-register "elevated" words win because they were common in the formal, edited text models learn to imitate, and fine-tuning rewarded sounding helpful and articulate.

**Signature words** (overused far beyond human rates)
- Verbs: delve, embark, navigate, foster, leverage, harness, unlock, unleash, elevate, empower, underscore, showcase, streamline, bolster, enhance, ensure, facilitate, optimize, resonate, thrive, embrace, cultivate, spearhead, revolutionize, transform, reshape, redefine, illuminate, unveil
- Nouns: tapestry, testament, realm, landscape, journey, ecosystem, framework, insights, synergy, paradigm, nuance, intricacies, interplay, cornerstone, beacon, catalyst, symphony, kaleidoscope, labyrinth, treasure trove, game-changer, double-edged sword
- Adjectives: crucial, pivotal, vital, essential, key, robust, seamless, comprehensive, multifaceted, intricate, nuanced, holistic, dynamic, vibrant, bustling, ever-evolving, cutting-edge, innovative, transformative, invaluable, unparalleled, meticulous, profound, noteworthy, compelling
- Adverbs: seamlessly, effortlessly, meticulously, notably, significantly, ultimately, arguably, truly, genuinely, incredibly, remarkably, deeply, profoundly

**Register inflation** — formal word where a plain one fits: individuals (people), utilize (use), commence (start), purchase (buy), endeavor (try), numerous (many), various (some/different), assist (help), demonstrate (show), in order to (to), prior to (before), a myriad of / a plethora of (many).

**Doublets and padding pairs**: "clear and concise", "safe and secure", "each and every", "first and foremost", "various and diverse", "efficient and effective", "thoughtful and intentional".

**Intensifier and sincerity adverbs**: truly, genuinely, really, incredibly, absolutely, deeply. Models use them to sound warm; humans in writing mostly don't.

**Hedge stacking**: "can potentially help", "may often", "it's possible that this could", "generally tends to", "in many cases". One hedge is honest; two in a clause is a model avoiding commitment.

**Vague attribution**: "experts agree", "studies show", "research suggests", "many believe", "it is widely recognized" — with no expert, study, or number named.

**Abstract over concrete**: "significant impact", "various factors", "key considerations", "a range of benefits", "positive outcomes". The specific thing is never named.

**Nominalization**: "the implementation of", "the utilization of", "provide an explanation of", "make a decision" — a verb turned into a noun phrase.

**Stock openers and closers** (phrases)
- "In today's fast-paced / digital / ever-changing world…"
- "In the realm/world/landscape of X…"
- "When it comes to X…"
- "Whether you're a seasoned X or just starting out…"
- "From X to Y, …" (From startups to Fortune 500 companies…)
- "Imagine a world where…"
- "In conclusion / In summary / Overall / Ultimately / At the end of the day…"
- "As X continues to evolve, …" — the single most common AI closing move
- "The future of X is bright / looks promising."
- "It's important to note / It's worth noting / It's worth mentioning…"
- "plays a crucial/pivotal/vital role in"
- "a wide range of / a variety of / a diverse array of"
- "serves as a testament to / stands as a testament to"
- "paving the way for", "the key to unlocking", "a deeper understanding"

## 2. Sentence-level templates

Why: models reuse sentence shapes that worked in training — they're rhetorically satisfying, sound insightful and are easy to fill with any content. These templates survive vocabulary swaps, so they're often the strongest remaining tell.

**The participial tail** — a main clause followed by ", -ing …" that comments on itself:
> "The city invested in bike lanes, **making it** easier for residents to commute."
> "She revised the plan, **ensuring** every team had a voice."
Common tails: ensuring, highlighting, making, allowing, enabling, providing, reflecting, creating, fostering, showcasing, underscoring, emphasizing, demonstrating, offering, paving the way, contributing to, reinforcing, solidifying. One per page is normal; one per paragraph is AI.

**Contrast frames** (negation-then-correction)
> "It's not just a tool — it's a mindset."
> "This isn't about technology. It's about people."
> "Not only does it save time, but it also reduces costs."
> "Less about X, more about Y." / "Rather than X, it's Y."
> "X isn't a bug; it's a feature."

**The colon or question reveal**
> "The result? A 40% drop in costs."
> "Here's the thing: …" / "The best part?" / "The catch?" / "The bottom line:"
> "So, what does this mean for you?" followed immediately by the answer.

**"This + abstract noun" openers** — starting a sentence by pointing back at the last one: "This approach…", "This shift…", "This highlights…", "This means that…", "These insights…". Humans do it; models do it in nearly every paragraph.

**"By + gerund" openers**: "By leveraging data, companies can…", "By understanding X, you can…".

**Fronted transitions**: Furthermore, Moreover, Additionally, However, Consequently, Notably, Importantly, Ultimately, Overall, In addition, As a result, On the other hand. Models open a large share of sentences this way; human writers mostly let order carry the logic.

**Balanced concession**: "While X offers benefits, it also presents challenges." "Although X, Y." Used reflexively to sound fair.

**Triads**: "fast, reliable, and secure"; "innovation, collaboration, and growth"; three parallel clauses in a row; three examples every time. The rule of three, applied mechanically.

**Universal-scope claims**: "Everyone has experienced…", "We've all been there.", "In a world where…".

**Definition openers**: "X refers to…", "X is a process by which…", "At its core, X is…".

## 3. Grammar and punctuation habits

- **Flawless, uniform grammar.** No typos, no fragments (unless deliberately styled), no run-ons, no comma splices, consistent Oxford comma, consistent tense. Real writing, even edited writing, has small irregularities and personal habits.
- **Em dashes as default punctuation** — for asides, for reveals, in pairs. Several per paragraph is a strong tell, especially in casual text where a human would use a comma, parentheses, or a new sentence.
- **Colons before lists and reveals**, frequently.
- **Few semicolons, few parentheses, few exclamation marks** in neutral prose (the opposite in "enthusiastic" prose).
- **Low contraction rate** in contexts where people contract ("it is", "do not", "you will").
- **Subject-first, medium-complexity sentences**: one main clause, one subordinate clause or participial phrase. Rarely very simple, rarely sprawling.
- **Heavy comma use with appositives and introductory phrases**.
- **Passive and "there is/are" constructions** in explanatory text.
- **Consistent noun repetition or synonym cycling**: models either repeat the key noun exactly in every sentence, or rotate synonyms ("the company… the firm… the organization… the business") to avoid repetition — both read as mechanical.
- **Pronoun avoidance**: "the user" instead of "you", "individuals" instead of "they".
- **Title Case Headings** and capitalized Key Concepts.

## 4. Rhythm

The most robust statistical tell, and the hardest to fake.

- **Low burstiness**: sentence lengths cluster around 15–25 words. Humans swing from 4 words to 40.
- **Uniform paragraphs**: 3–5 sentences each, all roughly the same length.
- **Predictable cadence**: claim, elaboration, implication — repeated.
- **No dead air**: every sentence does a "job". Humans include asides, false starts, small redundancies.
- **Low perplexity**: each next word is the expected word. Phrases are complete clichés rather than slightly unusual choices.

## 5. Paragraph structure

The AI paragraph template:
1. Topic sentence that announces the point (often starting with a transition)
2. Two or three supporting sentences, often with a triad or example
3. Wrap-up sentence restating the point's significance ("This makes X essential for Y.")

Other paragraph habits:
- **Mini-conclusions** at the end of each paragraph.
- **One idea per paragraph, exactly**, with no digressions.
- **Generic examples**: "a small business owner", "Company X", "imagine you're a student", round numbers ("increase productivity by 50%").
- **Over-explaining**: defining terms the reader obviously knows; spelling out implications already clear.
- **Both-sidesing**: every advantage gets a balancing disadvantage, whether or not the writer would really weigh them equally.

## 6. Whole-document structure

- **Intro that restates the topic** and previews what's coming ("In this article, we'll explore…", "Let's dive in.").
- **Symmetric sections**: same length, same number of bullets (usually 3–5), same internal shape.
- **Listicle skeleton** even for topics that need argument: numbered points with bold labels.
- **Conclusion that summarizes the body** and adds nothing, often ending with a future-looking line ("As X continues to evolve…") or a call to action.
- **Even coverage**: every subtopic gets equal weight; nothing is skipped, nothing gets passionate extra attention.
- **No narrative thread or callbacks**: sections don't build on each other or refer back to earlier specifics.

Chat-response artifacts (often left in pasted text):
- Openers: "Certainly!", "Great question!", "Absolutely!", "Sure! Here's…", "Here's a [revised/detailed] version…"
- Closers: "I hope this helps!", "Let me know if you'd like…", "Feel free to reach out…", "Would you like me to…?", "Happy writing!"
- Meta: "As an AI language model…", "I don't have personal opinions, but…", "[Insert name here]", placeholder brackets.
- Unprompted disclaimers: "It's always a good idea to consult a professional."

## 7. Formatting habits

- **Headers everywhere**, even in short pieces; Title Case.
- **Bold-label bullets**: `**Flexibility:** Remote work allows employees to…` — extremely characteristic.
- **Bullets for everything**, each a full sentence, parallel in structure, often ending with periods.
- **Bolding key phrases** mid-paragraph.
- **Emoji section markers** (🚀 ✅ 💡) in casual/marketing text.
- **"Key Takeaways", "TL;DR", "Final Thoughts", "Conclusion"** sections.
- **Comparison tables** for things a sentence could cover.
- **Horizontal rules** between sections.

When the text came from a PDF or Word file, most formatting is lost in extraction — weight this section less there, but look for leftover "Label: sentence" lines that were bullets.

## 8. Tone and stance

- **Relentlessly positive and encouraging**; problems are "challenges" or "opportunities".
- **Diplomatic neutrality**: no strong opinion, no one is wrong, every view "has merit".
- **Empathy formulas**: "It's completely understandable to feel…", "You're not alone."
- **Grand stakes**: revolutionize, reshape, game-changer, unprecedented, "the future of".
- **No lived experience**: no specific memory, place, date, name, sensory detail, or irritation — or, when asked for them, suspiciously tidy ones.
- **No humor**, or safe, flat humor.
- **Reader-flattery**: "As a savvy marketer, you know…"
- **Uniform warmth** from first line to last — no shifts in mood.

## 9. Second-order tells: what AI does when told to "sound human"

This section matters most when *you* are doing the humanizing. Asked to sound casual or human, models swap one set of tics for another:

- **Fragment overload**: "Simple. Effective. Done." "Big mistake." "Game changer." in every paragraph.
- **Fake spontaneity**: "Honestly?", "Look,", "Here's the thing:", "Let's be real.", "I'll be honest —", "Spoiler:", "Plot twist:".
- **Contrast frames and colon reveals multiply** ("It's not X. It's Y." becomes the default sentence).
- **Em dashes increase** rather than decrease.
- **Manufactured anecdotes and specifics**: invented names, numbers or memories that weren't in the source. Never add facts to sound human.
- **Forced informality**: "gonna", "kinda", slang, or deliberate typos in a register that doesn't call for them.
- **Overcorrected rhythm**: alternating long/short sentences in a regular pattern — which is just a new uniformity.
- **Synonym swapping only**: replacing "delve" with "dig into" and "crucial" with "key" while keeping the same paragraph template, triads and wrap-up sentences. The text still reads as AI.
- **Stripping all structure**: removing headings and paragraphs the original needed, or cutting length.

Good humanizing changes structure and rhythm first, vocabulary second, and adds nothing that isn't in the source.

## 10. Genre-specific tells

**Student essays**: "Throughout history, …" openers; thesis listing exactly three points; each body paragraph = one point; "In conclusion, it is clear that…"; quotes introduced with "As X once said,".

**Emails / cover letters**: "I hope this email finds you well", "I am writing to express my interest", "I am confident that my skills and experience make me an ideal candidate", "passionate about", "Thank you for your time and consideration", "I look forward to the opportunity to discuss…".

**LinkedIn / social**: one-sentence paragraphs; hook line + line break; "Here's what I learned:" + numbered list; "Agree?" closer; emoji bullets; "Let that sink in."

**Marketing / product copy**: "Elevate your X", "Unlock the power of", "Say goodbye to X and hello to Y", "Whether you're X or Y", "designed with you in mind", "seamless experience", "take your X to the next level".

**Academic**: "This paper delves into", "sheds light on", "a comprehensive analysis", "plays a pivotal role", "further research is needed to fully understand", "has garnered significant attention", "in recent years" with no years named, hedged findings stacked three deep.

**Creative writing / fiction**: names like Elara, Lyra, Kael, Aria, Sarah Chen, Dr. Evelyn; "the air was thick with", "a testament to", "eyes sparkling with mischief", "a smile tugging at the corners of her lips", "voice barely above a whisper", "the weight of unspoken words", "little did she know", "something shifted", "a mix of X and Y" for emotions; a neat, hopeful resolution; every sense described once in the opening paragraph.

**Summaries / reports**: "This report provides a comprehensive overview", "key findings include", bullet lists of findings with bold labels, "Overall, the results indicate…".

## 11. Quick audit checklist

Score each as none / some / heavy. Three or more "heavy" is a strong AI signal; one or two may just be a writer's style or a genre convention.

- [ ] Signature vocabulary and stock phrases (§1)
- [ ] Participial tails (", ensuring…", ", making…") (§2)
- [ ] Contrast frames ("not X, but Y") and colon/question reveals (§2)
- [ ] Transition-word sentence openers (§2)
- [ ] Triads everywhere (§2)
- [ ] Em-dash density; no contractions in casual text (§3)
- [ ] Uniform sentence and paragraph length (§4)
- [ ] Topic sentence → support → wrap-up paragraphs (§5)
- [ ] Generic examples, vague attribution, no specifics (§1, §5)
- [ ] Intro preview + summary conclusion + "as X continues to evolve" (§6)
- [ ] Chat artifacts left in ("I hope this helps", "Certainly!") (§6)
- [ ] Bold-label bullets, header-heavy, emoji markers (§7)
- [ ] Uniformly positive, opinion-free tone (§8)

## 12. What the calibration data showed (newer models write differently)

The scorer was fitted on ~370 labelled texts: human writing from 1990s Usenet, pre-2021 arXiv abstracts, Wikipedia, Project Gutenberg and Paul Graham's essays, against text from three current Claude models answering the same kinds of requests. Some findings cut against the folklore above, and they matter for how much weight to give each pattern:

- **The classic "ChatGPT" tells were nearly absent from current-model output.** Across 140 AI samples there were almost no em dashes, no "delve", no "crucial", and very few *Furthermore/Moreover* openers. A detector built only on those words (like this one's first version) scored at chance on real data. Treat sections 1–2 as strong evidence *when present* (older models, ChatGPT-style text, lazy prompting) but never treat their absence as evidence of a human writer.
- **What did separate AI from human text** was quieter and structural:
  - Humans use more **parentheses, semicolons, specific numbers and proper names** — concrete, specific, slightly messy writing.
  - AI text had **more even sentence lengths**, more **rule-of-three lists**, more **bold lead-ins** ("**Define your use case.** If you…"), and more "wrap-up" sentences closing paragraphs.
  - AI vocabulary was *more* varied word-to-word (a higher moving type-token ratio) — models avoid repeating words, while people happily reuse the same word.
  - People **repeat phrases** across a text more than models do.
- **GPT-2 perplexity** (how predictable each word is to a small language model) separates the two well: AI text is more predictable, with more of its words falling in the model's top-10 guesses.
- **Hard cases:** polished human essays and expert Q&A answers get flagged as AI more often than other human genres (10–16% at the raw cut-off). AI fiction used to slip through about half the time; after training on 106 AI stories and modern human fiction, 92–100% is caught. AI fiction rarely uses the famous clichés ("barely above a whisper" appeared in only 4% of samples). What gives it away is predictability to GPT-2 and tidy structure: every scene resolves, emotions are named, and sentences are evenly paced.

So when auditing, weigh rhythm, specificity and structure above vocabulary, and when humanizing, the highest-leverage changes are: vary sentence length, add the specifics the source already has (numbers, names, examples), allow the odd parenthesis or semicolon where it fits the register, break up triads, and drop bold lead-ins and paragraph wrap-ups.

## 13. How AI "humanizes" text, measured

76 AI texts were rewritten by three Claude models, under the requests people actually type: "make this undetectable by GPTZero/Turnitin", "humanize this, add natural imperfections", "paraphrase this", "rewrite it casually", "make it sound like a student wrote it". The skill's own careful rewrites were included for comparison. Each rewrite was compared with its original, and a technique counts when it moved its measure clearly. Share of rewrites using each technique:

| Technique | All | "humanize" | "paraphrase" | "casual" | this skill |
|---|---|---|---|---|---|
| Plainer, shorter words | **61%** | 62% | 54% | 92% | 33% |
| Adds contractions | **33%** | 62% | 15% | 62% | 17% |
| Breaks up rule-of-three lists | **32%** | 38% | 31% | 31% | 25% |
| Adds parentheses | **25%** | 0% | 0% | 23% | 54% |
| Shorter sentences on average | **25%** | 0% | 0% | 46% | 42% |
| Starts sentences with And / But / So | **24%** | 38% | 8% | 69% | 12% |
| Adds casual fillers (basically, honestly, pretty much) | **24%** | 38% | 0% | 54% | 12% |
| Forces short/long sentence alternation | **24%** | 0% | 23% | 23% | 33% |
| More sentence-length variety | **22%** | 23% | 0% | 23% | 38% |
| Removes Furthermore/Moreover openers | **18%** | 23% | 46% | 23% | 8% |
| Talks to the reader ("you") | **16%** | 31% | 8% | 31% | 0% |
| Adds personal hedges (I think, probably) | **11%** | 38% | 8% | 8% | 0% |
| Adds rhetorical questions | **8%** | 8% | 0% | 23% | 4% |
| Removes stock AI phrases | **8%** | 23% | 8% | 15% | 0% |
| Fake typos, dropped apostrophes, ellipses, em-dash changes | **0–1%** | | | | |

What this tells us about how AI rewrites:
- **Humanizing is mostly a surface operation.** The most common moves are word-level: simpler words, contractions, breaking a list of three. The paragraph template, the even pacing, the generic claims and the wrap-up endings usually survive, which is why the rewrites stay detectable.
- **"Casual" and "humanize" requests produce a recognisable second register**: And/But/So openers, fillers, contractions, "you", hedges like "I think". Together these form a fingerprint of their own (see §9). Real casual human writing has these features too, but rarely all at once in an otherwise tidy, well-organised text.
- **"Paraphrase" changes the least.** It swaps words and drops transition openers, keeps everything else, and is the most detectable.
- **AI rewriters almost never fake errors.** No typos, no dropped apostrophes, no lowercase "i", even when asked for "natural imperfections". They also leave punctuation habits like em dashes alone.
- **Detection after humanizing.** Before the detector was trained on humanized text, it caught 21% (style) to 58% (max) of the skill's own rewrites. After training with author-grouped cross-validation, it catches most of them (see the README for current figures). Humanizing reliably lowers a score; it doesn't reliably make text read as human to a trained detector.

For the rewriting side, the lesson is the same as in `humanizing-guide.md`: change structure, rhythm and specificity, not just vocabulary. Don't add the casual fingerprint (fillers + And/But openers + "you" + hedges) to a text whose register doesn't call for it.

## 14. Which signals are strong and which are weak

Each group of signals scored on its own (cross-validated AUC: 0.5 = useless, 1.0 = perfect), with the share of AI texts in which the group fires at all:

| Signal group | AUC alone | Fires on |
|---|---|---|
| Stock vocabulary ("delve", "crucial", "tapestry") | 0.52 | 17% |
| Fake-casual tics ("Honestly?", "Here's the thing") | 0.50 | 3% |
| Formatting (bold lead-ins, chatbot leftovers) | 0.57 | 8% |
| Paragraph template (wrap-ups, even paragraphs) | 0.71 | 54% |
| Rhythm (sentence length and variety) | 0.70 | 85% |
| Hand-written sentence templates | 0.73 | 39% |
| Specificity and voice | 0.74 | 43% |
| Vocabulary statistics | 0.76 | 75% |
| GPT-2 predictability | 0.75 | 100% |
| Punctuation habits | 0.82 | 34% |
| Neural detectors | 0.87 | 100% |
| **Learned sentence templates** (skeleton n-grams) | **0.97 across unseen genres** | 100% |

How to use this when auditing by hand:
- **Never conclude much from vocabulary alone.** The famous words are strong evidence when several cluster together, but most modern AI text has none of them.
- **Formatting tells are decisive when present** (pasted chatbot answers, bold-label bullets) but absent from most documents.
- **The strongest signals are distributional**: punctuation habits, rhythm, and above all *grammatical templates*. These are the habitual sentence shapes the learned template model picks up: list constructions, `w , and w`, `not just w but`, `by w-ing w , w can`, `this w w that`, and the steady clause-then-clause cadence. These survive word-level humanizing, which is why humanized text stays detectable.
