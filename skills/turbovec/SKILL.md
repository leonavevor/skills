---
name: turbovec
description: Use when users ask for TurboVec/TurboQuant or need help building vector indexes, running vector queries, managing IDs, filtering results, or integrating retrieval in RAG pipelines. Prioritize index and query correctness over benchmarking. Compression is a benefit, not the main objective.
---

# TurboVec Skill

Use this skill for practical TurboVec index-and-query work.

## When To Trigger

Trigger when the user asks to:
- build or update a TurboVec index
- query/search vectors, tune `k`, or add filtering
- manage stable IDs (`IdMapIndex`) and deletes/updates
- integrate TurboVec with RAG/retrieval pipelines

Do not trigger for:
- generic ML/model training questions
- prompt-writing/content-only tasks
- generic vector DB comparisons without TurboVec implementation intent

## Working Flow

1. Clarify inputs and constraints
   - embedding dimension, metric, corpus size, update pattern
   - required recall/correctness and memory budget
2. Build or inspect index
   - choose index type (`TurboQuantIndex` vs `IdMapIndex`)
   - verify ID mapping, updates, and deletes
3. Implement query path
   - embed query, run search, apply filters/allowlist
   - map results back to source documents safely
4. Validate behavior
   - empty query, missing IDs, no-match filters, small allowlist
   - confirm output schema expected by downstream code

## Built-In Utilities

From this skill directory:

```bash
python -m scripts build-from-directory ./data --output ./index --dim 384 --bit-width 4
python -m scripts build-from-file ./documents.jsonl --output ./index
python -m scripts build-from-urls "https://example.com/a,https://example.com/b" --output ./index
python -m scripts search ./index --query "example query" --k 10
python -m scripts integration-scaffold --framework langchain --output ./langchain_example.py
python -m scripts benchmark-compare --input ./runs.json --output ./report.md
python -m scripts summarize-results --input ./runs.json --output ./summary.json
python -m scripts selftest
```

Data loaders support directories, files, JSONL, and URLs. See `references/indexing_guide.md` for details.

## Output Contract

For diagnostics, return:
- what is wrong (or risky) in index/query behavior
- why it happens
- concrete fix steps (ordered by impact and risk)

For implementation, return:
- files changed and key logic decisions
- validation performed (tests/commands)
- follow-up tuning options if relevant

## Example Trigger Prompts

- "Build a TurboVec index from these docs and wire it into my RAG retriever."
- "My TurboVec results are wrong after deletes. Fix ID mapping."
- "Add allowlist filtering to TurboVec search for tenant-scoped retrieval."
