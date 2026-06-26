# TurboVec Indexing Scripts

Reusable Python utilities for building and querying TurboVec vector indexes from common data sources.

## Quick Start

### Build an index from a directory

```bash
python -m scripts build-from-directory ./docs --output ./my_index
```

### Search the index

```bash
python -m scripts search ./my_index --query "vector search" --k 10
```

## Available Commands

### Data Indexing

- **build-from-directory** — Index all files in a directory (supports `.txt`, `.md`, `.json`, `.py`, `.js`, `.yaml`, `.html`)
- **build-from-file** — Index a single file or JSONL dataset
- **build-from-urls** — Index content fetched from URLs
- **search** — Query a built index

### Utilities

- **benchmark-compare** — Generate markdown benchmark comparison reports
- **summarize-results** — Aggregate benchmark results to JSON
- **integration-scaffold** — Generate boilerplate for LangChain, LlamaIndex, Haystack, or Agno
- **selftest** — Run validation

## Modules

### `data_loaders.py`

Load documents from filesystem, URLs, and text strings.

**Classes:**
- `Document` — Represents a document (id, content, metadata)
- `FileLoader` — Load from directories and single files
- `URLLoader` — Fetch from URLs
- `TextLoader` — Create documents from strings

### `index_builder.py`

Build and search TurboVec indexes.

**Classes:**
- `IndexConfig` — Index configuration (dim, bit_width, use_id_map)
- `IndexBuilder` — Build indexes from documents
- `IndexSearcher` — Search a built index
- `SearchResult` — Individual result
- `EmbeddingProvider` — Abstract base for embedders
- `DummyEmbedder` — Simple embedding for testing

### `build_index.py`

CLI commands for building and querying indexes.

## Documentation

- **indexing_guide.md** — Comprehensive guide with examples, API docs, and troubleshooting
- **examples.py** — Practical code examples for common workflows

## Python API Example

```python
from pathlib import Path
from scripts.data_loaders import FileLoader
from scripts.index_builder import IndexBuilder, IndexConfig, DummyEmbedder

# Load documents
docs = list(FileLoader.from_directory(Path("./data")))

# Build index
config = IndexConfig(dim=384, bit_width=4, use_id_map=True)
embedder = DummyEmbedder(dim=384)
builder = IndexBuilder(config, embedder)
builder.add_documents(docs)
index = builder.build_index()

# Save
index.write("my_index.tq")
builder.save_metadata(Path("./output"))
```

## Installation

Requires Python 3.8+ and TurboVec:

```bash
pip install turbovec
```

For full functionality with embedding providers, also install:

```bash
pip install openai  # For OpenAI embeddings
pip install langchain  # For LangChain integration
```

## See Also

- TurboVec docs: https://github.com/RyanCodrai/turbovec
- Skill guide: `../references/indexing_guide.md`
- Examples: `../references/examples.py`

