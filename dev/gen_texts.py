#!/usr/bin/env python3
"""Generate fresh AI texts through provider APIs (keys come from the environment, never from files).

Models x prompt styles x tasks -> corpus/api/<model>/<style>__<task>.txt (skips files that exist).
Providers run in parallel threads, each rate-limited. Usage: python dev/gen_texts.py"""
import json, os, re, sys, threading, time, urllib.request, urllib.error
from pathlib import Path
OUT = Path(__file__).resolve().parent / "corpus" / "api"

TASKS = {
    "remote_work": "the pros and cons of remote work for software teams",
    "sourdough": "how a sourdough starter works and why it sometimes fails",
    "ev_adoption": "why electric vehicle adoption differs between countries",
    "landlord_email": "an email to a landlord asking for a broken heater to be repaired before winter",
    "dns": "how the Domain Name System resolves a website address",
    "library_news": "a local news report about a public library reopening after renovation",
    "inflation": "what causes inflation and how central banks respond",
    "gym_review": "a review of a neighbourhood gym after three months of membership",
    "sleep": "why sleep matters for memory and learning",
    "bike_commute": "a personal essay about switching from driving to cycling to work",
}
STYLES = {
    "plain": "Write about {t}. Aim for about 350 words.",
    "casual": "Write about {t}, the way a regular person would write it casually online, about 350 words. No headings or bullet points.",
    "persona": "You are a middle-aged tradesperson with strong opinions and not much patience for jargon. Write about {t}, in your own voice, about 350 words. No headings or bullet points.",
    "human": "Write about {t}, about 350 words. It must not read as AI-written: vary sentence length a lot, avoid stock phrases and tidy summaries, no headings or lists, sound like one specific person.",
}

def post(url, headers, body, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, json.dumps(body).encode(), {**headers, "Content-Type": "application/json", "User-Agent": "Mozilla/5.0 (compatible; textgen/1.0)"})
            return json.load(urllib.request.urlopen(req, timeout=120))
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:200]
            if e.code in (429, 500, 503):
                time.sleep(20 * (i + 1)); continue
            print("HTTP", e.code, url[:50], msg, flush=True); return None
        except Exception as e:
            time.sleep(10)
    return None

def gemini(model):
    key = os.environ["GEMINI_API_KEY"]
    def f(prompt):
        r = post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                 {"x-goog-api-key": key}, {"contents": [{"parts": [{"text": prompt}]}],
                                           "generationConfig": {"maxOutputTokens": 8000}})
        try:
            return "".join(p.get("text", "") for p in r["candidates"][0]["content"]["parts"] if not p.get("thought"))
        except Exception:
            return None
    return f

def openai_style(url, key_env, model):
    key = os.environ[key_env]
    def f(prompt):
        r = post(url, {"Authorization": "Bearer " + key}, {"model": model, "max_tokens": 8000,
                                                           "messages": [{"role": "user", "content": prompt}]})
        try:
            return r["choices"][0]["message"]["content"]
        except Exception:
            return None
    return f

MODELS = {  # name -> (provider thread, caller, seconds between calls)
    "gemini-flash-latest": ("gemini", gemini("gemini-flash-latest"), 7),
    "gemini-3.5-flash": ("gemini", gemini("gemini-3.5-flash"), 7),
    "gemini-3.1-flash-lite": ("gemini", gemini("gemini-3.1-flash-lite"), 7),
    "gpt-oss-120b": ("groq", openai_style("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY", "openai/gpt-oss-120b"), 6),
    "qwen3.8-27b": ("groq", openai_style("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY", "qwen/qwen3.8-27b"), 6),
    "nemotron-ultra": ("openrouter", openai_style("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY", "nvidia/nemotron-3-ultra-550b-a55b:free"), 8),
}

def clean(t):
    t = re.sub(r"<think>.*?</think>", "", t or "", flags=re.S).strip()
    return t

def worker(provider, items):
    for name, call, wait, style, task, path in items:
        if path.exists():
            continue
        prompt = STYLES[style].format(t=TASKS[task])
        t = clean(call(prompt))
        if len(t.split()) >= 100:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(t, encoding="utf-8")
            print(provider, name, style, task, len(t.split()), flush=True)
        else:
            print("skip", provider, name, style, task, len(t.split()), flush=True)
        time.sleep(wait)

def main():
    by = {}
    for name, (prov, call, wait) in MODELS.items():
        for task in TASKS:
            for style in STYLES:
                by.setdefault(prov if prov != "gemini" else name, []).append((name, call, wait, style, task, OUT / name / f"{style}__{task}.txt"))
    ths = [threading.Thread(target=worker, args=(p, it)) for p, it in by.items()]
    [t.start() for t in ths]; [t.join() for t in ths]
    print("done", flush=True)

if __name__ == "__main__":
    main()


