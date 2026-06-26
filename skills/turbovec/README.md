# TurboVec Skill

TurboVec skill package focused on index and query workflows.

## Core Purpose

- Build indexes from directories, files, JSONL, or URLs
- Query indexes with stable IDs and optional filtering patterns
- Support RAG integration with practical code-first outputs

## Main Files

- `SKILL.md` - trigger rules and operating guide
- `scripts/` - reusable CLI helpers
- `references/indexing_guide.md` - detailed usage patterns
- `evals/evals.json` - starter prompts

## Common Commands

Run from this directory:

```bash
python -m scripts build-from-directory ./data --output ./index
python -m scripts build-from-file ./documents.jsonl --output ./index
python -m scripts build-from-urls "https://example.com/a,https://example.com/b" --output ./index
python -m scripts search ./index --query "example query" --k 10
python -m scripts selftest
```

For full command list:

```bash
python -m scripts --help
```
