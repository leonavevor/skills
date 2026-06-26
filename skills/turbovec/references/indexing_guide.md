# TurboVec Indexing Guide

Use this guide to build and query indexes from directories, files, JSONL, and URLs.

## Install

```bash
pip install turbovec
```

## CLI Quick Start

Run commands from the `turbovec` skill directory.

```bash
python -m scripts build-from-directory ./data --output ./index
python -m scripts build-from-file ./documents.jsonl --output ./index
python -m scripts build-from-urls "https://example.com/a,https://example.com/b" --output ./index
python -m scripts search ./index --query "example query" --k 10
```

## Important Flags

- `--dim`: embedding dimension (default `384`)
- `--bit-width`: quantization bits (`2` or `4`, default `4`)
- `--no-recursive`: disable recursive directory crawl
- `--extensions`: comma-separated extensions for directory indexing

Example:

```bash
python -m scripts build-from-directory ./docs --output ./index --dim 1536 --bit-width 4 --extensions ".md,.txt"
```

## Python API

```python
from pathlib import Path
from scripts.data_loaders import FileLoader
from scripts.index_builder import IndexBuilder, IndexConfig, DummyEmbedder

# 1) Load documents
documents = list(FileLoader.from_directory(Path("./data")))

# 2) Build index
config = IndexConfig(dim=384, bit_width=4, use_id_map=True)
embedder = DummyEmbedder(dim=384)
builder = IndexBuilder(config, embedder)
builder.add_documents(documents)
index = builder.build_index()

# 3) Persist
index.write("./index/index.tq")
builder.save_metadata(Path("./index"))
```

## Search Example

```python
from pathlib import Path
from turbovec import IdMapIndex
from scripts.index_builder import IndexBuilder, IndexSearcher, DummyEmbedder

config, documents = IndexBuilder.load_metadata(Path("./index"))
index = IdMapIndex.load("./index/index.tq")
searcher = IndexSearcher(index, documents, DummyEmbedder(dim=config.dim))

results = searcher.search("example query", k=5)
for r in results:
    print(r.rank, r.score, r.document.id, r.document.path)
```

## Data Input Notes

- Directory mode supports: `.txt`, `.md`, `.json`, `.py`, `.js`, `.yaml`, `.yml`, `.html`
- JSONL mode expects one object per line with `content`; `id` is optional
- URL mode stores each URL as one document with URL metadata

## Troubleshooting

- `turbovec not installed`: run `pip install turbovec`
- no results: verify metadata and embedding dimensions match the saved index
- empty URL index: verify URLs are reachable from your environment

## Related Files

- `scripts/README.md`
- `references/examples.py`
- `SKILL.md`
