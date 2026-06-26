"""CLI for building TurboVec indexes from common data sources."""

from __future__ import annotations

from pathlib import Path

from .data_loaders import FileLoader, URLLoader
from .index_builder import (
    DummyEmbedder,
    IndexBuilder,
    IndexConfig,
    IndexSearcher,
)


def _parse_extensions(extensions: str | None) -> set[str] | None:
    if extensions is None:
        return None
    normalized = {ext.strip().lower() for ext in extensions.split(",") if ext.strip()}
    return normalized or None


def _build_and_save(documents: list, output: Path, dim: int, bit_width: int) -> None:
    print("Building index...")
    config = IndexConfig(dim=dim, bit_width=bit_width, use_id_map=True)
    builder = IndexBuilder(config, DummyEmbedder(dim=dim))
    builder.add_documents(documents)

    try:
        index = builder.build_index()
    except ImportError as e:
        print(f"Error: {e}")
        return

    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "index.tq"
    index.write(str(index_path))
    print(f"Index saved to {index_path}")

    builder.save_metadata(output)
    print(f"Metadata saved to {output}")


def run_build_from_directory(
    dirpath: str,
    output: str,
    dim: int,
    bit_width: int,
    recursive: bool,
    extensions: str | None,
) -> None:
    """Build index from directory."""
    dirpath = Path(dirpath)
    output = Path(output)

    print(f"Loading documents from {dirpath}...")
    ext_set = _parse_extensions(extensions)
    documents = list(FileLoader.from_directory(dirpath, recursive=recursive, extensions=ext_set))
    print(f"Loaded {len(documents)} documents")
    _build_and_save(documents, output, dim, bit_width)


def run_build_from_file(
    filepath: str,
    output: str,
    dim: int,
    bit_width: int,
) -> None:
    """Build index from single file."""
    filepath = Path(filepath)
    output = Path(output)

    if filepath.suffix == ".jsonl":
        print(f"Loading JSONL from {filepath}...")
        documents = list(FileLoader.from_jsonl(filepath))
    else:
        print(f"Loading file {filepath}...")
        documents = [FileLoader.from_file(filepath)]

    print(f"Loaded {len(documents)} documents")
    _build_and_save(documents, output, dim, bit_width)


def run_build_from_urls(
    urls: str,
    output: str,
    dim: int,
    bit_width: int,
) -> None:
    """Build index from URLs."""
    url_list = urls.split(",")
    output = Path(output)

    print(f"Fetching {len(url_list)} URLs...")
    documents = list(URLLoader.from_urls(url_list))
    print(f"Fetched {len(documents)} documents")

    if not documents:
        print("Error: No documents fetched")
        return
    _build_and_save(documents, output, dim, bit_width)


def run_search(
    index_dir: str,
    query: str,
    k: int,
) -> None:
    """Search an index."""
    index_dir = Path(index_dir)

    print(f"Loading index from {index_dir}...")
    try:
        from turbovec import IdMapIndex
    except ImportError:
        print("Error: turbovec not installed")
        return

    try:
        config, documents = IndexBuilder.load_metadata(index_dir)
        index = IdMapIndex.load(str(index_dir / "index.tq"))
        embedder = DummyEmbedder(dim=config.dim)
        searcher = IndexSearcher(index, documents, embedder)
    except Exception as e:
        print(f"Error loading index: {e}")
        return

    print(f"Searching for: {query}")
    results = searcher.search(query, k=k)

    if not results:
        print("No results found")
        return

    print(f"\nTop {len(results)} results:")
    for result in results:
        doc = result.document
        preview = (doc.content[:80] + "...") if len(doc.content) > 80 else doc.content
        print(f"\n  [{result.rank + 1}] Score: {result.score:.3f}")
        print(f"      ID: {doc.id}")
        print(f"      Path: {doc.path}")
        print(f"      Preview: {preview}")



def run(command: str, **kwargs) -> None:
    """Execute command."""
    if command == "build-from-directory":
        run_build_from_directory(
            kwargs["dirpath"],
            kwargs["output"],
            kwargs["dim"],
            kwargs["bit_width"],
            recursive=not kwargs.get("no_recursive", False),
            extensions=kwargs.get("extensions"),
        )
    elif command == "build-from-file":
        run_build_from_file(
            kwargs["filepath"],
            kwargs["output"],
            kwargs["dim"],
            kwargs["bit_width"],
        )
    elif command == "build-from-urls":
        run_build_from_urls(
            kwargs["urls"],
            kwargs["output"],
            kwargs["dim"],
            kwargs["bit_width"],
        )
    elif command == "search":
        run_search(
            kwargs["index_dir"],
            kwargs["query"],
            kwargs["k"],
        )
    else:
        raise SystemExit(f"unknown command: {command}")

