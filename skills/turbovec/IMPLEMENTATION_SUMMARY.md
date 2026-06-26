# TurboVec Skill - Implementation Summary

This skill package is intentionally simple and focused on index/query workflows.

## What It Includes

- `SKILL.md`: concise trigger and execution guidance
- `scripts/`: reusable CLI for indexing and searching
- `references/indexing_guide.md`: detailed usage reference
- `references/examples.py`: runnable usage examples

## Primary CLI Commands

```bash
python -m scripts build-from-directory ./data --output ./index
python -m scripts build-from-file ./documents.jsonl --output ./index
python -m scripts build-from-urls "https://example.com/a,https://example.com/b" --output ./index
python -m scripts search ./index --query "example query" --k 10
python -m scripts selftest
```

## Design Choices

- Index and query correctness first
- Stable IDs and metadata persistence by default
- Minimal dependency surface (stdlib helpers + optional `turbovec` runtime)
- Reusable scripts for common data sources and retrieval integration

## Validation

- Script modules compile
- CLI command surface is intact
- Existing subcommands remain available

