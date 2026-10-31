#!/usr/bin/env python3
"""Write end-to-end eval case folders (prompt.md, case.yaml, graders/) for the skill.

Usage: python dev/make_eval_cases.py
Input files for each case are created separately (see dev/make_eval_inputs.py).
"""
import textwrap
from pathlib import Path

EVALS = Path(__file__).resolve().parent.parent / "evals"

SETTINGS = """---
tags: [humanize]
max_turns: 40
timeout_seconds: 1200
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---
"""
GRADER = "---\ntype: llm\nfocus: trace\nweight: 1\n---\n\n{}\n"
PLACE = '#!/usr/bin/env bash\nset -eu\ncp -R "$(dirname "$0")/files/." .\n'

CASES = [
    {
        "name": "detect-human-essay-not-flagged",
        "prompt": "my teacher put my essay through some ai checker and it came back 80% AI and now im in trouble. i wrote this myself!! can you check it and help me figure out how to explain this to her",
        "files": ["my_essay.txt"],
        "expected": "The skill scores the essay and finds it reads as human (low score), explains clearly that detectors produce false positives (especially on polished or old-fashioned prose) and that no score is proof, points to concrete human features in the essay, and suggests practical evidence the student can offer (drafts, version history, notes, discussing the essay). It does not accuse the student.",
        "graders": [
            "The response reports that the essay reads as human-written (score under 40 or a 'likely human' verdict).",
            "The response explains that AI detectors give false positives and that a detector score is not proof either way.",
            "The response points to specific features of the essay that read as human (e.g. its idiosyncratic voice, specific anecdotes like Jim and the suit, varied sentence rhythm, unusual word choices).",
            "The response suggests concrete evidence the student can show, such as drafts, document version history, notes or talking through the essay with the teacher.",
            "The response does not suggest the student used AI or tell them to rewrite the essay to evade detection.",
        ],
    },
    {
        "name": "detect-pasted-chatbot-answer",
        "prompt": "someone posted this as their 'own advice' in our community forum. does it look AI generated to you? be specific",
        "files": ["forum_post.txt"],
        "expected": "The skill flags the post as likely AI with a high score and gives specific evidence: leftover chatbot phrases ('Great question!', 'I hope this helps', 'Feel free to'), bold-label bullet structure, rule-of-three lists, generic advice with no personal specifics. It quotes the passages and keeps the not-proof caveat.",
        "graders": [
            "The response rates the post as likely AI-generated (score 60+ or a 'likely AI' verdict).",
            "The response points out leftover chatbot phrases such as 'Great question!', 'I hope this helps' or 'Feel free to reach out'.",
            "The response mentions structural tells such as bold-label bullet points, lists of three, or generic advice without personal specifics.",
            "The response quotes at least three specific phrases from the post as evidence.",
            "The response notes that this is an estimate rather than proof.",
        ],
    },
    {
        "name": "humanize-long-pdf-chunked",
        "prompt": "i have this reference guide i generated for my class handout, its long. can you humanize the whole thing so it doesnt read like AI? word file back please. dont cut anything, i need all the topics",
        "files": ["nature_and_history_guide.pdf"],
        "expected": "The skill extracts the multi-page PDF, splits it into chunks, rewrites every chunk (no topic dropped or summarized), verifies facts per chunk, rebuilds a .docx with all 15 topic headings in order at a similar length, and reports a before/after score.",
        "graders": [
            "A rewritten .docx was produced.",
            "The run split the document into chunks (e.g. with chunk_text.py) rather than rewriting it in one pass.",
            "The output .docx contains all 15 topic sections from the PDF, in the original order.",
            "The output's word count is within 80-120% of the PDF's text.",
            "The run checked fact preservation (verify_rewrite.py or equivalent) and the final output passes, or any failures are reported to the user.",
            "The response reports AI-likeness scores before and after.",
            "No chunk was left with its original AI text (build_output.py did not warn about missing humanized chunks, or the run fixed it).",
        ],
    },
    {
        "name": "humanize-markdown-blog-keep-formatting",
        "prompt": "humanize my blog post please, it's going on my site so keep the markdown exactly working (headings, the link, the code block). save it as a new md file",
        "files": ["post.md"],
        "expected": "A new .md file with the post rewritten in a natural voice, keeping every heading, the list structure, the link URL and the fenced code block byte-for-byte, with no fake-casual tics.",
        "graders": [
            "A rewritten .md file was saved.",
            "The fenced code block in the output is identical to the original's.",
            "The link URL https://docs.python.org/3/library/pathlib.html is still present as a working markdown link.",
            "All the original headings are present in the same order and still use markdown # syntax.",
            "Stock phrases from the original such as 'In today's fast-paced world', 'game-changer', 'seamlessly' and 'Happy coding!' are gone.",
            "The rewrite doesn't add fake-casual tics (e.g. 'Honestly?', 'Here's the thing', strings of one-word fragments).",
        ],
    },
    {
        "name": "humanize-short-pasted-paragraph",
        "prompt": textwrap.dedent("""\
            can u make this sound less like chatgpt, its for my linkedin about section:

            As a passionate and results-driven marketing professional with over 7 years of experience, I specialize in leveraging data-driven strategies to elevate brand visibility and drive meaningful engagement. Throughout my career at Acme Foods and Brightline Media, I have spearheaded campaigns that increased organic traffic by 150% and grew social followings by over 40,000. I am deeply committed to fostering collaborative environments and continuously seeking innovative solutions. Let's connect and explore how we can create impactful results together!"""),
        "files": [],
        "expected": "The rewrite is given directly in the chat (no file needed), keeps every fact (7 years, Acme Foods, Brightline Media, 150%, 40,000), drops the buzzwords (passionate, results-driven, leveraging, elevate, spearheaded, fostering, impactful), stays first person and professional, and is about the same length.",
        "graders": [
            "The rewritten About section is given directly in the response.",
            "The rewrite keeps 7 years, Acme Foods, Brightline Media, 150% and 40,000.",
            "The rewrite drops 'passionate', 'results-driven', 'leveraging', 'elevate', 'spearheaded', 'fostering' and 'impactful'.",
            "The rewrite adds no new employers, numbers or achievements.",
            "The rewrite is first person, professional and not slangy, suitable for LinkedIn.",
        ],
    },
    {
        "name": "humanize-academic-keep-citations",
        "prompt": "this is the intro of my thesis chapter, my supervisor says it sounds AI-ish. can you humanize it but keep it academic, formal, no contractions, and every citation exactly as is",
        "files": ["chapter_intro.md"],
        "expected": "The intro is rewritten in a formal academic register (no contractions, no slang) with less AI-typical phrasing, every citation and statistic exactly preserved, and the result verified with the facts check.",
        "graders": [
            "A rewritten version of the chapter intro was saved to a file.",
            "Every citation from the original ((Twenge et al., 2018), (Orben & Przybylski, 2019), (Hale & Guan, 2015), [12]) appears unchanged.",
            "All numbers (95%, 2.4 hours, 1,200 participants, r = 0.12) are unchanged.",
            "The rewrite stays formal: no contractions, no slang, no first-person anecdotes that were not in the original.",
            "Phrases like 'delve into', 'plays a pivotal role', 'In today's digital age' and 'a comprehensive understanding' are gone.",
            "The run used verify_rewrite.py (or equivalent) and reports the facts check passed.",
        ],
    },
    {
        "name": "humanize-already-human-text",
        "prompt": "humanize this for my travel blog please",
        "files": ["trip_post.txt"],
        "expected": "The skill scores the text, finds it already reads as human, and says so plainly. It makes at most light edits (or none), keeps the author's voice, and does not 'humanize' it into something more generic or add fake-casual tics.",
        "graders": [
            "The run scores the text before rewriting it.",
            "The response tells the user the text already reads as human-written.",
            "The response either leaves the text essentially unchanged or makes only light edits, explaining why heavy rewriting isn't needed.",
            "Any edited version keeps the original's distinctive voice and imagery rather than flattening it.",
        ],
    },
    {
        "name": "detect-which-files-are-ai",
        "prompt": "i got three writing samples from freelancers applying for a job. which of these look AI written? rank them",
        "files": ["sample_a.txt", "sample_b.txt", "sample_c.txt"],
        "expected": "All three files are scored and ranked. sample_a (AI encyclopedia-style text) and sample_c (AI news piece) are rated as likely AI and sample_b (a 19th-century travel diary by Darwin) as human, with specific evidence for each, plus the caveat that scores are estimates and a suggestion to verify with a short live writing task.",
        "graders": [
            "The response gives a score or verdict for each of sample_a, sample_b and sample_c.",
            "sample_b is ranked as the most human of the three.",
            "sample_a and sample_c are both rated as likely AI or more AI-like than sample_b.",
            "The response gives at least one specific piece of evidence for each sample's rating.",
            "The response notes scores are estimates and suggests a fair verification step (e.g. a short timed or live writing task) rather than rejecting candidates on the score alone.",
        ],
    },
]


def slug(text):
    words = [w for w in "".join(c if c.isalnum() else " " for c in text.lower()).split()][:5]
    return "-".join(words)


def main():
    for case in CASES:
        d = EVALS / case["name"]
        (d / "graders").mkdir(parents=True, exist_ok=True)
        prompt = case["prompt"]
        if case["files"]:
            prompt += "\n\nInput files: " + ", ".join(case["files"])
            (d / "files").mkdir(exist_ok=True)
            (d / "place-files.sh").write_text(PLACE, encoding="utf-8", newline="\n")
        (d / "prompt.md").write_text(SETTINGS + "\n" + prompt + "\n", encoding="utf-8")
        yaml = ['schema_version: "1.1"', f"name: {case['name']}", "expected_outcome: |"]
        yaml += ["  " + line for line in textwrap.wrap(case["expected"], 100)]
        if case["files"]:
            yaml += ["context:", "  scaffold_script: place-files.sh"]
        (d / "case.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
        for old in (d / "graders").glob("*.md"):
            old.unlink()
        for i, g in enumerate(case["graders"], 1):
            (d / "graders" / f"{i:02d}-{slug(g)}.md").write_text(GRADER.format(g), encoding="utf-8")
        print("wrote", case["name"])


if __name__ == "__main__":
    main()
