# Humanize (AI Detect & Humanize)

A Claude skill that estimates how AI-generated a text reads, shows which sentences give it away, and rewrites AI-sounding prose into natural writing while keeping every fact. Works on pasted text, PDF, Word and Markdown files, including long documents.

- Default mode runs locally with only `pypdf` and `python-docx`; nothing you write leaves your computer.
- Optional `--deep` and `--max` modes use local models (`torch`, `transformers`); `--max` downloads about 3.7 GB of open models from Hugging Face once.
- Detector scores are estimates, not proof of authorship. Accuracy numbers and limits are in the repository README: https://github.com/sujan-mitta/ai-detect-humanize-skill

MIT licensed. [Privacy](https://github.com/sujan-mitta/ai-detect-humanize-skill/blob/main/plugin/README.md#privacy-and-data) | [Source and issues](https://github.com/sujan-mitta/ai-detect-humanize-skill)

## Install (Claude Code)

```
/plugin marketplace add sujan-mitta/ai-detect-humanize-skill
/plugin install humanize@humanize-marketplace
```

One-step form on Claude Code 2.1.275 or later: `/plugin install humanize --marketplace sujan-mitta/ai-detect-humanize-skill`

## Privacy and data

- **Everything runs on your own computer.** The skill reads the text or files you give it (which may contain personal data such as names or emails) and writes its outputs, such as rewritten text, chunks and reports, only to your own working folder. It has no server and keeps no copy anywhere else.
- **No user text is sent anywhere.** The only network access is the optional one-time download of open-source model files from Hugging Face when you first use `--deep` or `--max` mode; no text you write is part of that request.
- **No credentials are read or required.** The skill does not use API keys, tokens or accounts.
- **Not directed at children.** It is a general-purpose writing tool.
