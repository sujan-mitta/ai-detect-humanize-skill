#!/usr/bin/env python3
"""Heuristic AI-likeness score for a text (0 = reads human, 100 = reads AI).

Usage:
  python ai_score.py INPUT [--report REPORT.md] [--json OUT.json] [--compare BEFORE.json]

INPUT can be .txt/.md/.pdf/.docx. The score combines stylistic signals that
AI-generated prose tends to show, grouped the same way as
references/ai-patterns.md:

  rhythm      - uniform sentence length, uniform paragraphs + wrap-up sentences
  vocabulary  - stock words/phrases, hedges, intensifiers, vague attribution
  templates   - participial tails, contrast frames / reveals, repeated openers,
                transition openers, rule-of-three lists
  punctuation - few contractions, heavy em-dash use
  document    - markdown formatting habits, chatbot artifacts, repeated phrasing

This is a heuristic, not proof of authorship. Formal human writing (legal,
academic methods sections) can score moderately high; casual AI text that
was prompted for a voice can score low.
"""
import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extract_text import extract  # noqa: E402

STOCK_WORDS = [
    "delve", "delves", "delving", "tapestry", "testament", "realm", "intricate",
    "intricacies", "multifaceted", "pivotal", "crucial", "robust", "seamless",
    "seamlessly", "leverage", "leveraging", "utilize", "utilizing", "utilization",
    "foster", "fostering", "underscore", "underscores", "showcase", "showcasing",
    "embark", "unlock", "unleash", "harness", "harnessing", "elevate", "empower",
    "empowering", "holistic", "synergy", "paradigm", "nuanced", "meticulous",
    "meticulously", "comprehensive", "ever-evolving", "cutting-edge",
    "game-changer", "game-changing", "vibrant", "bustling", "myriad", "plethora",
    "landscape", "navigate", "navigating", "streamline", "streamlined",
    "transformative", "invaluable", "noteworthy", "commendable", "furthermore",
    "moreover", "additionally", "consequently", "notably", "indispensable",
    "paramount", "endeavor", "endeavour", "bolster", "resonate", "resonates",
    "beacon", "cornerstone", "catalyst", "interplay", "garner", "garnered",
    "illuminate", "unveil", "unveiling", "revolutionize", "revolutionizing",
    "spearhead", "cultivate", "unparalleled", "effortlessly", "profound",
    "kaleidoscope", "labyrinth", "symphony", "spearheaded", "spearheading", "fostered", "elevated",
    "elevating", "leveraged", "harnessed", "empowered", "streamlining", "bolstered", "underscored",
    "showcased", "impactful", "results-driven", "passionate", "synergies", "actionable",
]

STOCK_PHRASES = [
    r"it is important to note", r"it'?s important to note", r"it is worth noting",
    r"it'?s worth (noting|mentioning)", r"in today'?s (fast-paced|digital|modern|rapidly|ever)",
    r"ever-(evolving|changing) (landscape|world)", r"plays? an? (crucial|pivotal|vital|key|significant) role",
    r"a (wide|diverse) (range|array|variety) of", r"whether you'?re", r"when it comes to",
    r"not only\b.{1,80}\bbut also", r"(serves?|stands?) as a testament", r"in conclusion",
    r"in summary", r"at the end of the day", r"this highlights the importance",
    r"let'?s (dive|explore|delve)", r"dive (deep|deeper) into", r"in the (realm|world|landscape) of",
    r"a deep(er)? understanding", r"the (power|importance|significance) of",
    r"navigat\w+ the complexities", r"rich tapestry", r"key takeaways?", r"in essence",
    r"ultimately,", r"overall,", r"by doing so", r"in order to",
    r"(unlock|unleash) (the|its|their|your) (full )?potential",
    r"(paves?|paving) the way", r"(a|an) (unique|valuable) (opportunity|insight)",
    r"as [\w ,]{1,40}continues? to (evolve|grow|advance|shape|develop)",
    r"the future of [\w ]{1,30} (is|looks) (bright|promising)", r"imagine a world",
    r"first and foremost", r"each and every", r"sheds? light on", r"has garnered",
    r"further research is needed", r"i hope this (email|message) finds you well",
    r"take [\w ]{1,20} to the next level", r"say goodbye to", r"let that sink in",
    r"throughout history", r"in recent years", r"a (sense|mix) of",
    r"data-driven (strategies|decision|insights|approach)", r"innovative solutions", r"drive meaningful",
    r"let'?s connect", r"i am (writing to express|confident that my)", r"ideal candidate",
    r"aligns? perfectly with", r"thank you for your time and consideration", r"i look forward to hearing",
    r"deeply committed", r"forward-thinking", r"continuous improvement", r"actionable insights",
]

# ", -ing" comment tails: "..., ensuring that every team has a voice."
PARTICIPLE_RE = re.compile(
    r",\s+(ensuring|highlighting|making|allowing|enabling|providing|reflecting|creating|"
    r"fostering|showcasing|underscoring|emphasizing|demonstrating|offering|paving|"
    r"contributing|reinforcing|solidifying|leading|resulting|helping|giving|turning|"
    r"setting|marking|signaling|cementing)\b", re.I)

# "It's not X, it's Y" / "not just X but Y" / "The result? ..." and friends
CONTRAST_RES = [re.compile(p, re.I) for p in [
    r"\bnot (just|only|merely|simply)\b",
    r"\b(isn|wasn|aren)'?t (just |only |merely )?about\b",
    r"\bit'?s not\b[^.?!]{0,50}[.,;—–-]+\s*it'?s\b",
    r"\b(it|this|that) (is|was)n'?t\b[^.?!]{0,50}[.;—]\s*(it|this|that)('s| is| was)\b",
    r"\bless about\b[^.?!]{0,40}\bmore about\b",
    r"(^|[.!?]\s+)(the (result|catch|best part|answer|truth|kicker|bottom line|takeaway|secret|"
    r"problem|twist|reality)|here'?s (the thing|why|what|how)|the reason)\s*[?:]",
    r"\b(honestly|spoiler|plot twist|let'?s be (real|honest))\s*[?:,!]",
]]

HEDGE_RES = [re.compile(p, re.I) for p in [
    r"\b(can|could|may|might) (potentially|possibly|often|generally|help to)\b",
    r"\b(generally|typically|often) tends? to\b",
    r"\bit'?s possible that\b", r"\bin many cases\b", r"\bto some extent\b",
    r"\b(truly|genuinely|incredibly|absolutely|deeply|profoundly|remarkably|undeniably)\b",
    r"\b(experts|studies|research|scientists|many people|some people|many) "
    r"(say|says|agree|suggest|suggests|show|shows|believe|indicate|indicates)\b",
    r"\bit is (widely )?(recognized|believed|accepted|known)\b",
]]

# AI fiction tells (ai-patterns.md section 10, creative writing)
FICTION_RES = [re.compile(p, re.I) for p in [
    r"\b(elara|lyra|kael|aria|seraphina|thorne|elowen|caelum|zephyr|evelyn chen|sarah chen|marcus chen)\b",
    r"\bthe air (was|hung) (thick|heavy) with\b", r"\b(a|the) smile tugg(ed|ing) at\b", r"\bbarely above a whisper\b",
    r"\bthe weight of\b", r"\b(something|everything) shifted\b", r"\blittle did (she|he|they) know\b",
    r"\beyes (sparkling|glinting|gleaming) with\b", r"\ba (mix|mixture) of [a-z]+ and [a-z]+\b",
    r"\b(couldn'?t|could not) help but\b", r"\bunspoken\b", r"\b(a|the) testament to\b", r"\bfor the first time in (years|forever|a long time)\b",
    r"\b(heart|pulse) (raced|pounded|hammered)\b", r"\bhe let out a breath (he|she) didn'?t know\b",
    r"\bbreath (she|he|they) didn'?t (know|realize)\b", r"\bsend(ing)? shivers down\b", r"\bin that moment\b",
    r"\b(her|his|their) voice (cracked|softened|trembled)\b", r"\bsomething (like|akin to) (hope|grief|relief)\b",
    r"\bwhispered,? (as if|like)\b", r"\bthe silence (stretched|hung|settled)\b",
]]

# What models write when asked to "sound human" (ai-patterns.md section 9)
PERFORMED_VOICE_RES = [re.compile(p, re.I) for p in [
    r"(^|[.!?]\s+)(look|listen|honestly|seriously|truth is)[,?:]", r"\bhere'?s the (thing|kicker|deal)\b",
    r"\bnot even close\b", r"\blet'?s be (real|honest)\b", r"\b(spoiler|plot twist)( alert)?:",
    r"\bit turns out\b|\bturns out,", r"\bgame[- ]changer\b", r"\bbig mistake\b",
    r"\bi'?ll be honest\b", r"\bhear me out\b", r"\bthe kind that\b",
]]

CHATBOT_RES = [re.compile(p, re.I) for p in [
    r"\b(great|excellent|good) question\b", r"^\s*(certainly|absolutely|sure)[!,.]",
    r"\bi hope this helps\b", r"\bfeel free to\b", r"\blet me know if\b",
    r"\bas an ai\b", r"\bi'?d be happy to\b", r"\bwould you like me to\b",
    r"\bhere'?s an? (revised|improved|updated|detailed|polished|rewritten) (version|draft)\b",
    r"\[(insert|your|company|name)[^\]]*\]", r"\bhappy (writing|coding|learning)!",
]]

TRANSITIONS = {
    "furthermore", "moreover", "additionally", "however", "consequently",
    "therefore", "thus", "hence", "notably", "importantly", "ultimately",
    "overall", "indeed", "nevertheless", "nonetheless", "similarly",
    "conversely", "subsequently", "accordingly", "firstly", "secondly",
    "lastly", "finally", "in addition", "in conclusion", "in summary",
    "as a result", "on the other hand", "for instance", "for example",
}

OPENER_TEMPLATE_RE = re.compile(
    r"^[\"'(]?(this|these) \w+|^by \w+ing\b|^whether\b|^from [\w ]{1,30} to |^imagine\b|"
    r"^while\b|^although\b|^in (today|the (world|realm|age|era))\b|^when it comes\b", re.I)
WRAPUP_RE = re.compile(
    r"^(this|these|overall|ultimately|in short|in the end|together|as a result)\b|"
    r"\b(essential|crucial|vital|key|invaluable|indispensable) (for|to|in)\b", re.I)

CONTRACTION_RE = re.compile(r"\b\w+'(s|t|re|ve|ll|d|m)\b", re.I)
TRIPLET_RE = re.compile(r"\b[\w-]+(?: [\w-]+)?, [\w-]+(?: [\w-]+)?,? and [\w-]+", re.I)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
BOLD_LABEL_RE = re.compile(r"^\s*(?:(?:[-*•]|\d+[.)])\s+)?\*\*[^*\n]{1,60}\*\*", re.M)
FIRST_PERSON_RE = re.compile(r"\b(i|me|my|mine|i'm|i've|i'd|i'll)\b", re.I)
PROPER_RE = re.compile(r"(?<=[a-z,;] )[A-Z][a-z]+")
LABEL_LINE_RE = re.compile(r"^\s*(?:[-*•]\s+)?[A-Z][\w /&-]{1,35}:\s+[A-Z]", re.M)
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")

# name, weight, human-readable explanation
SIGNALS = [
    ("burstiness", 0.15, "Sentence length variety (uniform lengths read as AI)"),
    ("stock_language", 0.14, "AI-typical words and phrases"),
    ("performed_voice", 0.06, "Fake-casual tics: fragment runs, 'Honestly?', 'Here's the thing'"),
    ("participial_tails", 0.09, "', ensuring/highlighting/making...' comment tails"),
    ("transitions", 0.08, "Sentences opening with Furthermore/Moreover/Additionally..."),
    ("contrast_frames", 0.07, "'Not X, but Y' frames and 'The result?' reveals"),
    ("opener_patterns", 0.07, "Repeated sentence openers (This..., By -ing..., While...)"),
    ("paragraph_template", 0.06, "Same-length paragraphs ending in wrap-up sentences"),
    ("hedges_intensifiers", 0.05, "Stacked hedges, 'truly/incredibly', 'experts say'"),
    ("triplets", 0.05, "Rule-of-three lists ('X, Y, and Z')"),
    ("chatbot_artifacts", 0.05, "Chat leftovers ('I hope this helps', 'Certainly!')"),
    ("contractions", 0.04, "Few contractions (don't, it's...)"),
    ("em_dashes", 0.04, "Heavy em-dash use"),
    ("formatting", 0.03, "Bold-label bullets, 'Label: sentence' lines, emoji markers"),
    ("repetition", 0.02, "Repeated 4-word phrases across the text"),
]

# The final score blends the weighted mean of all signals with the mean of the
# strongest few, so a handful of maxed-out signals isn't averaged away by the
# many that happen to be quiet for a given genre.
TOP_K = 4
TOP_K_BLEND = 0.5


MODEL_PATH = Path(__file__).parent / "model.json"

# plain-language names for model features, used to explain a score
FEATURE_LABELS = {
    "mean_sentence_length": "average sentence length", "sentence_length_cv": "sentence-length variety",
    "paragraph_length_cv": "paragraph-length variety", "wrapup_paragraph_share": "paragraphs ending in a wrap-up",
    "stock_hits_per_1k": "AI-typical words/phrases", "hedges_per_1k": "hedges and intensifiers",
    "participial_tails_per_1k": "', ensuring...' tails", "contrast_frames_per_1k": "'not X but Y' frames",
    "transition_start_ratio": "Furthermore/Moreover openers", "template_opener_ratio": "template openers",
    "top_opener_share": "repeated first word", "triplets_per_1k": "rule-of-three lists",
    "fragment_ratio": "short fragments", "performed_voice_tics": "fake-casual tics",
    "contractions_per_1k": "contractions", "em_dashes_per_1k": "em dashes", "chatbot_artifacts": "chatbot leftovers",
    "repeated_4gram_ratio": "repeated phrases", "mattr": "vocabulary variety", "avg_word_length": "word length",
    "commas_per_sentence": "commas per sentence", "semicolons_per_1k": "semicolons", "parens_per_1k": "parentheses",
    "questions_per_1k": "questions", "exclaims_per_1k": "exclamation marks", "first_person_per_1k": "first person (I/me/my)",
    "digits_per_1k": "specific numbers", "proper_nouns_per_1k": "names and proper nouns",
    "bold_leads_per_1k": "bold lead-ins", "fiction_cliches_per_1k": "AI fiction clichés",
    "dialogue_ratio": "dialogue", "log_perplexity": "GPT-2 perplexity (predictability)",
    "top1_fraction": "words GPT-2 guessed exactly", "top10_fraction": "words in GPT-2's top 10",
    "rare_fraction": "surprising word choices", "sentence_ppl_cv": "perplexity variety across sentences",
}
_MODELS = None


def load_models():
    global _MODELS
    if _MODELS is None:
        _MODELS = json.loads(MODEL_PATH.read_text(encoding="utf-8")) if MODEL_PATH.exists() else {}
    return _MODELS


def model_score(raw, deep=None, kind=None):
    """Calibrated probability (0-100) from model.json, with per-feature contributions.

    kind: "style", "deep" (style + GPT-2) or "max" (style + GPT-2 + neural
    detectors); by default "deep" when extra features are given, else "style".
    Returns None when no trained model is present (heuristic score is used).
    """
    models = load_models()
    kind = kind or ("deep" if deep else "style")
    if kind not in models:
        kind = "deep" if deep and "deep" in models else "style"
    m = models.get(kind)
    if not m:
        return None
    logs = set(m.get("log_features", []))
    values = [raw.get(name, (deep or {}).get(name)) for name in m["features"]]
    values = [math.log1p(max(0.0, v)) if (n in logs and v is not None) else v for n, v in zip(m["features"], values)]
    if m.get("type") == "gb":
        return gb_score(m, values, kind)
    contrib = []
    z = m["intercept"]
    for name, v, mu, sd, c in zip(m["features"], values, m["mean"], m["scale"], m["coef"]):
        if v is None:
            v = mu  # missing (e.g. too few sentences for a CV): treat as average
        term = c * (float(v) - mu) / (sd or 1)
        z += term
        contrib.append((name, term, float(v) > mu))
    contrib.sort(key=lambda t: -abs(t[1]))
    prob = 1 / (1 + math.exp(-max(-30, min(30, z))))
    return {
        "score": round(100 * prob),
        "model": "deep" if m is models.get("deep") else "style",
        "toward_ai": [(describe(n, above), round(t, 2)) for n, t, above in contrib if t > 0.15][:6],
        "toward_human": [(describe(n, above), round(t, 2)) for n, t, above in contrib if t < -0.15][:6],
    }


# Safety net for dense, classic AI tells. The model learns from data in which
# such phrases are rare on both sides, and short texts give it little to go on,
# so a 77-word paragraph stuffed with "leveraging", "spearheaded" and "Let's
# connect" could slip through. In the 394 human calibration texts, tell density
# never exceeded 2.7 per 100 words, so the net starts above that.
TELL_FLOOR, TELL_FULL = 2.7, 7.0


def tell_density(text, prose, sents):
    n = len(WORD_RE.findall(prose))
    tells = (len(stock_hits(prose)) + 2 * len(find_all(CHATBOT_RES, text)) + len(find_all(CONTRAST_RES, prose))
             + len(PARTICIPLE_RE.findall(prose)) + sum(starts_with_transition(s) for s in sents))
    return 100 * tells / max(n, 60)


def apply_tell_net(score, density):
    p_tell = clamp((density - TELL_FLOOR) / (TELL_FULL - TELL_FLOOR))
    return round(100 * (1 - (1 - score / 100) * (1 - p_tell))), p_tell


def gb_raw(m, x):
    """Log-odds from exported boosted trees (node = [feature, threshold, left,
    right, is_leaf, value, missing_goes_left])."""
    z = m["baseline"]
    for tree in m["trees"]:
        node = tree[0]
        while not node[4]:
            v = x[node[0]]
            go_left = node[6] if v is None or v != v else v <= node[1]
            node = tree[node[2] if go_left else node[3]]
        z += node[5]
    return z


def gb_score(m, values, kind):
    """Score plus explanations: each feature's effect is how much the log-odds
    change when that feature is reset to the typical (median) training value."""
    a, b = m.get("platt", [1.0, 0.0])
    z = gb_raw(m, values)
    contrib = []
    for i, (name, v, med) in enumerate(zip(m["features"], values, m["medians"])):
        if v is None:
            continue
        alt = list(values)
        alt[i] = med
        contrib.append((name, a * (z - gb_raw(m, alt)), v > med))
    z = a * z + b
    contrib.sort(key=lambda t: -abs(t[1]))
    prob = 1 / (1 + math.exp(-max(-30, min(30, z))))
    return {
        "score": round(100 * prob),
        "model": kind,
        "toward_ai": [(describe(n, above), round(t, 2)) for n, t, above in contrib if t > 0.1][:6],
        "toward_human": [(describe(n, above), round(t, 2)) for n, t, above in contrib if t < -0.1][:6],
    }


# Features a faithful rewrite can't really change: they come from the facts
# (numbers, names) or from the language model's view of the content.
CONTENT_FEATURES = {"digits_per_1k", "proper_nouns_per_1k", "log_perplexity", "top1_fraction",
                    "top10_fraction", "rare_fraction"}


def describe(name, above):
    """'fewer parentheses', 'higher average sentence length' - the direction matters:
    a feature is listed because the text has more OR less of it than typical."""
    label = FEATURE_LABELS.get(name, name)
    countable = name not in {"mean_sentence_length", "sentence_length_cv", "paragraph_length_cv", "mattr",
                             "avg_word_length", "commas_per_sentence", "log_perplexity", "sentence_ppl_cv"}
    word = ("more" if above else "fewer") if countable else ("higher" if above else "lower")
    text = f"{word} {label}"
    if name in CONTENT_FEATURES:
        text += " (content)"
    return text


def clamp(x):
    return max(0.0, min(1.0, x))


CITATION_RE = re.compile(
    r"\((?:see |e\.g\.,? |cf\. )?[A-Z][A-Za-z'\-]+(?: et al\.?|,? (?:and|&) [A-Z][A-Za-z'\-]+)?,? \d{4}[a-z]?"
    r"(?:[;,] ?(?:p+\. ?)?[\w\s.,&'\-]*?\d{4}[a-z]?)*(?:, p+\. ?\d+(?:[-–]\d+)?)?\)"
    r"|\[\d+(?:[,–-]\s*\d+)*\]")
STAT_PAREN_RE = re.compile(r"\((?=[^)]*[=<>\d])[^()a-z]*(?:\b[a-zA-Z]{1,3}\b[^()a-z]*)*\)")


def strip_markup(text):
    """Remove things that aren't prose style: code, links, URLs, citations and
    bracketed statistics. Otherwise e.g. '(Smith et al., 2019)' or '(n = 214)'
    would count as a human-style parenthetical and a handful of digits."""
    t = re.sub(r"```.*?```", " ", text, flags=re.S)
    t = re.sub(r"`[^`\n]+`", "code", t)
    t = re.sub(r"!?\[([^\]]*)\]\([^)]+\)", r"\1", t)  # markdown links/images -> link text
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = CITATION_RE.sub("", t)
    t = re.sub(r"(?<=[A-Z][a-z]) (?:et al\.? )?\(\d{4}[a-z]?\)", "", t)  # Author (2019)
    t = STAT_PAREN_RE.sub("", t)
    t = re.sub(r"\([A-Z][A-Za-z0-9\-]*[A-Z0-9]s?\)", "", t)  # acronym definitions: (NLP), (RNNs)
    t = re.sub(r"\$[^$\n]{1,80}\$", "x", t)  # inline LaTeX math
    return t


def prose_blocks(text):
    """Paragraphs excluding headings, tables, code, links and citations."""
    blocks = []
    for block in re.split(r"\n\s*\n", strip_markup(text)):
        b = block.strip()
        if not b or b.startswith("|"):
            continue
        lines = [l for l in b.splitlines() if not re.match(r"^#{1,6}\s", l)]
        if lines:
            blocks.append(" ".join(l.strip() for l in lines))
    return blocks


def sentences(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\b(e\.g|i\.e|etc|vs|Dr|Mr|Mrs|Ms|Fig|al)\.", lambda m: m.group(0).replace(".", "․"), text)
    parts = re.split(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])", text)
    return [p.replace("․", ".").strip() for p in parts if len(WORD_RE.findall(p)) >= 3]


def stock_hits(text):
    low = text.lower()
    hits = []
    for w in STOCK_WORDS:
        for _ in re.finditer(r"\b" + re.escape(w) + r"\b", low):
            hits.append(w)
    for p in STOCK_PHRASES:
        for m in re.finditer(p, low):
            hits.append(m.group(0))
    return hits


def count_all(patterns, text):
    return sum(len(p.findall(text)) for p in patterns)


def find_all(patterns, text):
    out = []
    for p in patterns:
        out += [m.group(0).strip(" .,;:") for m in p.finditer(text)]
    return out


def starts_with_transition(sentence):
    low = sentence.lower().lstrip("\"'( ")
    return any(low.startswith(t + ",") or low.startswith(t + " ") for t in TRANSITIONS)


def first_word(sentence):
    m = WORD_RE.search(sentence)
    return m.group(0).lower() if m else ""


def analyze(text):
    blocks = prose_blocks(text)
    prose = "\n\n".join(blocks)
    para_sents = [sentences(b) for b in blocks]
    sents = [s for ps in para_sents for s in ps]
    words = WORD_RE.findall(prose)
    n_words = max(1, len(words))
    n_sents = max(1, len(sents))
    # Densities use at least 400 words as the denominator so one incidental
    # match in a short paragraph doesn't max out a signal.
    per_k = 1000.0 / max(n_words, 400)

    # rhythm
    lengths = [len(WORD_RE.findall(s)) for s in sents]
    cv = statistics.pstdev(lengths) / max(1e-9, statistics.mean(lengths)) if len(lengths) >= 3 else 0.55
    para_lengths = [len(WORD_RE.findall(b)) for b in blocks if len(WORD_RE.findall(b)) >= 15]
    pcv = statistics.pstdev(para_lengths) / max(1e-9, statistics.mean(para_lengths)) if len(para_lengths) >= 4 else 0.5
    real_paras = [ps for ps in para_sents if len(ps) >= 3]
    wrapups = sum(bool(WRAPUP_RE.search(ps[-1])) for ps in real_paras)
    wrapup_share = wrapups / len(real_paras) if real_paras else 0.0

    # vocabulary
    hits = stock_hits(prose)
    hedges = find_all(HEDGE_RES, prose)

    # sentence templates
    participles = PARTICIPLE_RE.findall(prose)
    contrasts = find_all(CONTRAST_RES, prose)
    trans = sum(starts_with_transition(s) for s in sents)
    trans_ratio = trans / n_sents
    template_openers = sum(bool(OPENER_TEMPLATE_RE.search(s.lstrip("\"'( "))) for s in sents)
    firsts = Counter(w for w in map(first_word, sents) if w not in {"the", "a", "i", "we", "you"})
    top_first, top_count = firsts.most_common(1)[0] if firsts else ("", 0)
    top_share = top_count / n_sents if len(sents) >= 6 else 0.0
    triplets = len(TRIPLET_RE.findall(prose)) * per_k

    # performed voice: runs of 1-3 word fragments plus fake-casual tics
    all_parts = [p for b in blocks for p in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", b)) if WORD_RE.search(p)]
    fragments = sum(len(WORD_RE.findall(p)) <= 3 for p in all_parts)
    frag_ratio = fragments / max(1, len(all_parts)) if len(all_parts) >= 6 else 0.0
    tics = find_all(PERFORMED_VOICE_RES, prose)

    # punctuation (contractions use the true word count: it's a rate, not a count of tells)
    contractions = len(CONTRACTION_RE.findall(prose)) * 1000.0 / n_words
    dashes = (prose.count("—") + len(re.findall(r"\w -- \w|\w--\w", prose))) * per_k

    # document level (raw text keeps the markdown)
    chat = find_all(CHATBOT_RES, text)
    bold_labels = len(BOLD_LABEL_RE.findall(text))
    label_lines = len(LABEL_LINE_RE.findall(text))
    emojis = len(EMOJI_RE.findall(text))
    lw = [w.lower() for w in words]
    grams = Counter(tuple(lw[i:i + 4]) for i in range(len(lw) - 3))
    repeated = sum(c for c in grams.values() if c >= 2)
    rep_ratio = repeated / max(1, sum(grams.values()))

    # general stylometry: vocabulary richness, word length, punctuation, specificity
    win = 50
    if len(lw) > win:
        mattr = statistics.mean(len(set(lw[i:i + win])) / win for i in range(0, len(lw) - win, 10))
    else:
        mattr = len(set(lw)) / max(1, len(lw))
    true_k = 1000.0 / n_words

    raw = {
        "words": len(words),
        "sentences": len(sents),
        "mean_sentence_length": round(statistics.mean(lengths), 1) if lengths else 0,
        "sentence_length_cv": round(cv, 3),
        "paragraph_length_cv": round(pcv, 3),
        "wrapup_paragraph_share": round(wrapup_share, 3),
        "stock_hits_per_1k": round(len(hits) * per_k, 2),
        "hedges_per_1k": round(len(hedges) * per_k, 2),
        "participial_tails_per_1k": round(len(participles) * per_k, 2),
        "contrast_frames_per_1k": round(len(contrasts) * per_k, 2),
        "transition_start_ratio": round(trans_ratio, 3),
        "template_opener_ratio": round(template_openers / n_sents, 3),
        "most_common_opener": f"{top_first} ({top_count})",
        "top_opener_share": round(top_share, 3),
        "triplets_per_1k": round(triplets, 2),
        "fragment_ratio": round(frag_ratio, 3),
        "performed_voice_tics": len(tics),
        "contractions_per_1k": round(contractions, 2),
        "em_dashes_per_1k": round(dashes, 2),
        "chatbot_artifacts": len(chat),
        "bold_label_bullets": bold_labels,
        "label_lines": label_lines,
        "emojis": emojis,
        "repeated_4gram_ratio": round(rep_ratio, 3),
        "mattr": round(mattr, 4),
        "avg_word_length": round(statistics.mean(len(w) for w in words), 3) if words else 0,
        "commas_per_sentence": round(prose.count(",") / n_sents, 3),
        "semicolons_per_1k": round(prose.count(";") * true_k, 2),
        "parens_per_1k": round(prose.count("(") * true_k, 2),
        "questions_per_1k": round(prose.count("?") * true_k, 2),
        "exclaims_per_1k": round(prose.count("!") * true_k, 2),
        "first_person_per_1k": round(len(FIRST_PERSON_RE.findall(prose)) * true_k, 2),
        "digits_per_1k": round(len(re.findall(r"\d+", prose)) * true_k, 2),
        "proper_nouns_per_1k": round(len(PROPER_RE.findall(prose)) * true_k, 2),
        "bold_leads_per_1k": round(bold_labels * true_k, 2),
        "fiction_cliches_per_1k": round(len(find_all(FICTION_RES, prose)) * true_k, 2),
        "dialogue_ratio": round(sum(len(q) for q in re.findall(r"[\"“][^\"”\n]{2,400}[\"”]", prose))
                                / max(1, len(prose)), 3),
    }
    sig = {
        "burstiness": clamp((0.65 - cv) / 0.40),
        "stock_language": clamp(len(hits) * per_k / 12.0),
        "participial_tails": clamp(len(participles) * per_k / 6.0),
        "transitions": clamp((trans_ratio - 0.03) / 0.15),
        "contrast_frames": clamp(len(contrasts) * per_k / 8.0),
        "performed_voice": 0.6 * clamp((frag_ratio - 0.10) / 0.25) + 0.4 * clamp(len(tics) / 3.0),
        "opener_patterns": 0.6 * clamp((template_openers / n_sents - 0.05) / 0.20)
                           + 0.4 * clamp((top_share - 0.12) / 0.20),
        "paragraph_template": 0.6 * clamp((0.60 - pcv) / 0.45) + 0.4 * clamp((wrapup_share - 0.15) / 0.5),
        "hedges_intensifiers": clamp(len(hedges) * per_k / 8.0),
        "triplets": clamp(triplets / 6.0),
        "chatbot_artifacts": clamp(len(chat) / 2.0),
        "contractions": clamp(1 - contractions / 15.0),
        "em_dashes": clamp(dashes / 8.0),
        "formatting": clamp((bold_labels + 0.7 * label_lines + 0.5 * emojis) * per_k / 10.0),
        "repetition": clamp((rep_ratio - 0.02) / 0.08),
    }
    weighted = sum(sig[name] * w for name, w, _ in SIGNALS)
    strongest = sorted(sig.values(), reverse=True)[:TOP_K]
    heuristic = round(100 * ((1 - TOP_K_BLEND) * weighted + TOP_K_BLEND * sum(strongest) / TOP_K))
    calibrated = model_score(raw)
    score = calibrated["score"] if calibrated else heuristic
    raw["tells_per_100w"] = round(tell_density(text, prose, sents), 2)
    if calibrated:
        score, p_tell = apply_tell_net(score, raw["tells_per_100w"])
        calibrated["tell_net"] = round(p_tell, 2)
    return {
        "score": score,
        "heuristic_score": heuristic,
        "calibrated": calibrated,
        "verdict": verdict(score),
        "low_confidence": len(words) < 150,
        "signals": {k: round(v, 3) for k, v in sig.items()},
        "raw": raw,
        "top_stock_terms": Counter(hits).most_common(15),
        "examples": {
            "participial_tails": [p.lower() for p in participles][:8],
            "contrast_frames": contrasts[:8],
            "hedges_intensifiers": hedges[:8],
            "performed_voice": tics[:8],
            "chatbot_artifacts": chat[:8],
        },
        "_sentences": sents,
    }


def bands(kind="style"):
    """(likely-AI threshold, likely-human threshold). Set in calibration so that
    at most 5% of human texts reach 'likely AI'; defaults if no model is present."""
    th = load_models().get(kind, {}).get("thresholds") if MODEL_PATH.exists() else None
    return (th["ai"], th["human"]) if th else (60, 40)


def verdict(score, kind="style"):
    t_ai, t_human = bands(kind)
    if score >= t_ai:
        return "likely AI"
    if score >= t_human:
        return "mixed / possibly AI-assisted"
    return "likely human"


def sections(text, target=250):
    """Split at headings, and within long stretches into ~target-word groups of
    whole paragraphs, labelled by paragraph number so a reader can find them."""
    out, heading, buf, count, para_no, first = [], None, [], 0, 0, 1

    def flush():
        if buf:
            label = f"¶{first}" if para_no == first else f"¶{first}-{para_no}"
            out.append((f"{heading} ({label})" if heading else f"Paragraphs {label[1:]}", "\n\n".join(buf)))

    for block in re.split(r"\n\s*\n", text):
        b = block.strip()
        if not b:
            continue
        m = re.match(r"^#{1,6}\s+(.*)", b)
        if m and "\n" not in b:
            flush()
            heading, buf, count, first = m.group(1).strip()[:50], [], 0, para_no + 1
            continue
        para_no += 1
        buf.append(b)
        count += len(b.split())
        if count >= target:
            flush()
            buf, count, first = [], 0, para_no + 1
    flush()
    # merge tiny sections into the previous one
    merged = []
    for t, body in out:
        if merged and len(body.split()) < 60:
            merged[-1] = (merged[-1][0] + " + " + t, merged[-1][1] + "\n\n" + body)
        else:
            merged.append((t, body))
    return merged


def sentence_detail(s):
    """(weight, reasons) of AI patterns in one sentence."""
    reasons, weight = [], 0
    hits = stock_hits(s)
    if hits:
        reasons.append("stock: " + ", ".join(sorted(set(hits))[:4]))
        weight += 2 * len(hits)
    checks = [
        (PARTICIPLE_RE.search(s), "participial tail", 2),
        (any(p.search(s) for p in CONTRAST_RES), "contrast frame / reveal", 2),
        (starts_with_transition(s), "transition opener", 1),
        (OPENER_TEMPLATE_RE.search(s.lstrip("\"'( ")), "template opener", 1),
        (any(p.search(s) for p in HEDGE_RES), "hedge / intensifier", 1),
        (any(p.search(s) for p in PERFORMED_VOICE_RES), "fake-casual tic", 2),
        (any(p.search(s) for p in CHATBOT_RES), "chatbot artifact", 3),
        ("—" in s, "em dash", 1),
        (TRIPLET_RE.search(s), "rule of three", 1),
        (BOLD_LABEL_RE.search(s), "bold lead-in", 1),
    ]
    for hit, label, w in checks:
        if hit:
            reasons.append(label)
            weight += w
    return weight, reasons


def sentence_flags(sents, limit=12):
    scored = []
    for s in sents:
        weight, reasons = sentence_detail(s)
        if weight >= 2:
            scored.append((weight, s, reasons))
    scored.sort(key=lambda x: -x[0])
    return [{"sentence": s[:300], "reasons": r} for _, s, r in scored[:limit]]


def build(text, deep=False, max_mode=False):
    overall = analyze(text)
    if deep or max_mode:
        import perplexity
        if not perplexity.available():
            print("NOTE: --deep/--max need `pip install torch transformers`; using the style model.", file=sys.stderr)
        else:
            feats = perplexity.features(text)
            kind = "deep"
            if feats and max_mode:
                import neural
                nf = neural.features(text)
                if nf:
                    overall["neural"] = nf
                    feats = {**feats, **nf}
                    kind = "max" if "max" in load_models() else "deep"
            if feats:
                overall["deep"] = {k: v for k, v in feats.items() if k != "sentence_ppl"}
                cal = model_score(overall["raw"], feats, kind)
                if cal:
                    cal["score"], p_tell = apply_tell_net(cal["score"], overall["raw"]["tells_per_100w"])
                    cal["tell_net"] = round(p_tell, 2)
                    overall["calibrated"] = cal
                    overall["score"] = cal["score"]
                    overall["verdict"] = verdict(cal["score"], cal["model"])
    secs = []
    for title, body in sections(text):
        a = analyze(body)
        secs.append({"title": title, "words": a["raw"]["words"], "score": a["score"], "verdict": a["verdict"]})
    overall["sections"] = secs
    # A document whose sections clearly disagree is reported as mixed: averaging a
    # human half with an AI half would otherwise hide the AI half.
    solid = [s for s in secs if s["words"] >= 100]
    t_ai, t_human = bands()
    ai_secs = [s for s in solid if s["score"] >= t_ai]
    # Below the AI threshold any clearly human section makes it "mixed"; above it,
    # only a strongly human section does (so one plain paragraph in an AI document
    # doesn't soften the verdict).
    human_cut = t_human if overall["score"] < t_ai else t_human * 0.4
    if ai_secs and any(s["score"] < human_cut for s in solid):
        overall["mixed"] = {"ai_sections": [s["title"] for s in ai_secs], "of": len(solid)}
        overall["verdict"] = f"mixed - {len(ai_secs)} of {len(solid)} sections read as AI"
    overall["flagged_sentences"] = sentence_flags(overall.pop("_sentences"))
    return overall


def render(result, source, before=None):
    r = result["raw"]
    lines = [f"# AI-likeness report: {Path(source).name}", ""]
    if before:
        d = result["score"] - before["score"]
        lines.append(f"**Score:** {before['score']} -> **{result['score']}** ({d:+d})  |  "
                     f"{before['verdict']} -> **{result['verdict']}**")
    else:
        lines.append(f"**Score:** {result['score']}/100 - **{result['verdict']}**")
    if result.get("mixed"):
        lines.append("**Mixed document:** these sections read as AI while others read as human: "
                     + "; ".join(result["mixed"]["ai_sections"]))
    lines.append(f"Words: {r['words']}, sentences: {r['sentences']}"
                 + ("  (short text - low confidence)" if result["low_confidence"] else ""))
    cal = result.get("calibrated")
    if cal:
        models = load_models()
        cv = models.get("cv", {}).get(f"{cal['model']}_cv", {})
        how = {"deep": "style + GPT-2 perplexity", "max": "style + GPT-2 + neural AI-text detectors"}.get(cal["model"], "writing style")
        lines.append(f"Model: calibrated on {models.get('trained_on', {}).get('n_train', '?')} labelled texts using {how}"
                     + (f" (cross-validated accuracy {cv['accuracy']:.0%}, AUC {cv['auc']:.2f})" if cv else ""))
        if result.get("deep"):
            d = result["deep"]
            lines.append(f"GPT-2: perplexity {d['perplexity']}, {d['top10_fraction']:.0%} of words in its top-10 guesses")
        if cal["toward_ai"]:
            lines.append("**Pushing toward AI** (compared with typical human writing): "
                         + ", ".join(n for n, _ in cal["toward_ai"]))
        if cal["toward_human"]:
            lines.append("**Pushing toward human:** " + ", ".join(n for n, _ in cal["toward_human"]))
        if cal.get("tell_net", 0) > 0:
            lines.append(f"**Dense AI tells:** {r['tells_per_100w']} per 100 words (no human text in the calibration set "
                         f"exceeded {TELL_FLOOR}) - this raised the score.")
        if any("(content)" in n for n, _ in cal["toward_ai"]):
            lines.append("_(content) = comes from the facts or subject matter; a faithful rewrite can't change it._")
        if r["words"] < 300:
            lines.append(f"_Short text ({r['words']} words): expect the score to move 10+ points from small edits._")
    lines += ["", "_Statistical estimate from writing style. Not proof of authorship._", "",
              "## Pattern signals", "", "| Signal | AI-ness (0-1) | Measured |", "|---|---|---|"]
    measured = {
        "burstiness": f"sentence-length CV {r['sentence_length_cv']} (human ~0.5-0.9)",
        "stock_language": f"{r['stock_hits_per_1k']} per 1k words",
        "participial_tails": f"{r['participial_tails_per_1k']} per 1k words",
        "transitions": f"{r['transition_start_ratio']:.0%} of sentences",
        "contrast_frames": f"{r['contrast_frames_per_1k']} per 1k words",
        "performed_voice": f"{r['fragment_ratio']:.0%} of sentences are 1-3 word fragments; {r['performed_voice_tics']} tics",
        "opener_patterns": f"{r['template_opener_ratio']:.0%} template openers; top opener '{r['most_common_opener']}'",
        "paragraph_template": f"paragraph-length CV {r['paragraph_length_cv']}; {r['wrapup_paragraph_share']:.0%} end in a wrap-up",
        "hedges_intensifiers": f"{r['hedges_per_1k']} per 1k words",
        "triplets": f"{r['triplets_per_1k']} per 1k words",
        "chatbot_artifacts": f"{r['chatbot_artifacts']} found",
        "contractions": f"{r['contractions_per_1k']} per 1k words",
        "em_dashes": f"{r['em_dashes_per_1k']} per 1k words",
        "formatting": f"{r['bold_label_bullets']} bold-label bullets, {r['label_lines']} label lines, {r['emojis']} emoji",
        "repetition": f"{r['repeated_4gram_ratio']:.1%} of 4-word phrases repeat",
    }
    ranked = sorted(SIGNALS, key=lambda s: -result["signals"][s[0]] * s[1])
    for name, _, desc in ranked:
        bv = f" (was {before['signals'].get(name, '-')})" if before else ""
        lines.append(f"| {desc} | {result['signals'][name]}{bv} | {measured[name]} |")
    if result["top_stock_terms"]:
        lines += ["", "**Most frequent AI-typical terms:** "
                  + ", ".join(f"{t} ({n})" for t, n in result["top_stock_terms"])]
    labels = {"participial_tails": "Participial tails", "contrast_frames": "Contrast frames / reveals",
              "hedges_intensifiers": "Hedges / intensifiers", "performed_voice": "Fake-casual tics",
              "chatbot_artifacts": "Chatbot artifacts"}
    for key, label in labels.items():
        ex = result["examples"].get(key)
        if ex:
            lines.append(f"**{label}:** " + ", ".join(f"\"{e}\"" for e in dict.fromkeys(ex)))
    if len(result["sections"]) > 1:
        lines += ["", "## Sections", "", "| # | Section | Words | Score | Verdict |", "|---|---|---|---|---|"]
        for i, s in enumerate(result["sections"], 1):
            lines.append(f"| {i} | {s['title']} | {s['words']} | {s['score']} | {s['verdict']} |")
    if result["flagged_sentences"]:
        lines += ["", "## Most AI-sounding sentences", ""]
        for f in result["flagged_sentences"]:
            lines.append(f"- \"{f['sentence']}\"  \n  _{'; '.join(f['reasons'])}_")
    lines += ["", "_The script can't see structural patterns like paragraph templates, both-sidesing, "
              "generic examples or summary conclusions - audit those with references/ai-patterns.md._"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--report", help="write a markdown report here")
    ap.add_argument("--json", help="write full results as JSON here")
    ap.add_argument("--compare", help="JSON from an earlier run, to show before/after")
    ap.add_argument("--deep", action="store_true", help="add GPT-2 perplexity features (needs torch + transformers)")
    ap.add_argument("--max", action="store_true", help="most accurate: GPT-2 + neural detectors (downloads ~2.5 GB once)")
    ap.add_argument("--html", help="also write an interactive sentence-heatmap report here")
    args = ap.parse_args()

    text = extract(args.input)
    result = build(text, deep=args.deep, max_mode=args.max)
    before = json.loads(Path(args.compare).read_text(encoding="utf-8")) if args.compare else None
    report = render(result, args.input, before)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(report, encoding="utf-8")
    if args.html:
        import html_report
        page, _ = html_report.build_report(args.input, deep=args.deep and "deep" in result)
        Path(args.html).write_text(page, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
