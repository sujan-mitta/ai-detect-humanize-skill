#!/usr/bin/env python3
"""Questions only the author can answer: the reliable way to make a generic text read as theirs.

Usage: python author_questions.py INPUT [--max-q 8] [--json]

Rewriting cannot add what the author knows. This finds the paragraphs that are most generic
(few concrete markers - numbers, names, places, quotes, first-person experience - and strong
AI-skeleton heat) and writes one targeted question for each. Ask the author these questions,
then put their answers into the text in their own words (see SKILL.md, step 4b). The answers
must come from the author; never invent them.

Specificity = concrete markers per 100 words. Texts people wrote about their own life usually
sit well above AI drafts of the same topic."""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ai_score  # noqa: E402
from extract_text import extract  # noqa: E402
from rewrite_targets import split_sentences, template_heat  # noqa: E402

NUM_RE = re.compile(r"\b\d[\d,.:%/-]*\b")
NAME_RE = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b")
QUOTE_RE = re.compile(r"[\"“”][^\"“”]{3,}[\"“”]")
FIRST_RE = re.compile(r"\b(I|I'm|I've|I'd|my|me|we|our|us)\b")
GENERIC_RE = re.compile(r"\b(important|crucial|essential|key|significant|various|numerous|many|often|can|may|helps?|ensure|enable|leverage|landscape|ecosystem|journey|today's|modern|overall|ultimately)\b", re.I)

ASK = {
    "essay": "What actually happened to you around this point? Name the moment, the place or the person, in one or two sentences.",
    "blog": "Give one real example from your own experience for this point: who, where, when, and what you did or noticed.",
    "email": "Is there a concrete detail (a date, a name, a number, something said in a meeting) that belongs here?",
    "news": "What is the source for this (who said it, what document, what number, what date)? Add a direct quote if you have one.",
    "technical": "What did you observe when you tried this yourself: exact version, command, error message, number, or a case where it did not work?",
    "report": "Which specific figure, date, case or named source supports this? What does it contradict or leave out?",
    "story": "What would only you notice here: a smell, a sentence someone said, an object, something slightly off?",
    "default": "What is a specific detail here that only you know (a name, a number, a place, a moment, an opinion you hold)?",
}


def genre_guess(text):
    t = text.lower()
    if re.search(r"^(dear|hi|hello|subject:)", t.strip()) or "regards" in t[-200:]:
        return "email"
    if re.search(r"\b(according to|said in a statement|officials)\b", t):
        return "news"
    if re.search(r"\b(function|api|config|install|run the|version|error)\b", t):
        return "technical"
    if re.search(r"\b(pursuant|recommend|agency|fiscal|committee)\b", t):
        return "report"
    if len(FIRST_RE.findall(text)) > 3:
        return "blog"
    return "default"


def specificity(par):
    n = max(1, len(par.split()))
    c = len(NUM_RE.findall(par)) + len(NAME_RE.findall(par)) + 2 * len(QUOTE_RE.findall(par)) + len(FIRST_RE.findall(par))
    return 100.0 * c / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--max-q", type=int, default=8)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    text = extract(a.input)
    pars = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.split()) >= 25 and not p.lstrip().startswith("#")]
    tm = ai_score.load_models().get("templates")
    rows = []
    for i, p in enumerate(pars, 1):
        sents = split_sentences(p)
        heat = sum(template_heat(sents, tm)) / max(1, len(sents)) if tm and sents else 0.0
        gen = len(GENERIC_RE.findall(p)) * 100.0 / len(p.split())
        spec = specificity(p)
        rows.append({"paragraph": i, "words": len(p.split()), "specificity": round(spec, 1), "generic": round(gen, 1),
                     "heat": round(heat, 3), "starts": " ".join(p.split()[:12]) + " ...", "_rank": gen + 8 * heat - spec})
    rows.sort(key=lambda r: -r["_rank"])
    g = genre_guess(text)
    out = []
    EXTRA = ["", " Or: where do you disagree with this, or what did it get wrong in your case?",
             " Or: what surprised you, or what did you almost leave out?"]
    for k, r in enumerate(rows[:a.max_q]):
        first = split_sentences(pars[r["paragraph"] - 1])
        claim = " ".join(first[0].split()[:22]) + (" ..." if first and len(first[0].split()) > 22 else "") if first else ""
        r["question"] = f'The draft says: "{claim}" - {ASK[g]}{EXTRA[k % 3]}'
        r.pop("_rank")
        out.append(r)
    out.sort(key=lambda r: r["paragraph"])
    overall = specificity(" ".join(pars)) if pars else 0.0
    if a.json:
        print(json.dumps({"genre_guess": g, "specificity_per_100w": round(overall, 1), "questions": out}, indent=1))
        return
    print(f"Genre guess: {g}. Specificity {overall:.1f} concrete markers per 100 words. Ask the AUTHOR these (answers must be theirs):\n")
    for r in out:
        print(f"Paragraph {r['paragraph']} (\"{r['starts']}\") - specificity {r['specificity']}, generic words {r['generic']}/100w")
        print(f"  Q: {r['question']}\n")


if __name__ == "__main__":
    main()
