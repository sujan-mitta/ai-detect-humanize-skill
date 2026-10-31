# Humanize (AI Detect & Humanize)

A Claude skill that estimates how AI-generated a text reads, shows which sentences give it away, and rewrites AI-sounding prose into natural writing while keeping every fact. Works on pasted text, PDF, Word and Markdown files, including long documents.

- Default mode runs locally with only `pypdf` and `python-docx`; nothing you write leaves your computer.
- Optional `--deep` and `--max` modes use local models (`torch`, `transformers`); `--max` downloads about 3.7 GB of open models from Hugging Face once.
- Detector scores are estimates, not proof of authorship. Accuracy numbers and limits are in the repository README: https://github.com/sujan-mitta/ai-detect-humanize-skill

MIT licensed.

## Install (Claude Code)

```
/plugin marketplace add sujan-mitta/ai-detect-humanize-skill
/plugin install humanize@humanize-marketplace
```

One-step form on Claude Code 2.1.275 or later: `/plugin install humanize --marketplace sujan-mitta/ai-detect-humanize-skill`
