"""TurboVec indexing examples (compact version)."""

from __future__ import annotations

from pathlib import Path

from scripts.data_loaders import FileLoader, URLLoader
from scripts.index_builder import DummyEmbedder, IndexBuilder, IndexConfig, IndexSearcher


def example_build_from_directory() -> None:
    """Build index from a local directory."""
    documents = list(FileLoader.from_directory(Path("./docs")))

    config = IndexConfig(dim=384, bit_width=4, use_id_map=True)
    builder = IndexBuilder(config, DummyEmbedder(dim=384))
    builder.add_documents(documents)

    index = builder.build_index()
    output_dir = Path("./out/dir_index")
    output_dir.mkdir(parents=True, exist_ok=True)
    index.write(str(output_dir / "index.tq"))
    builder.save_metadata(output_dir)

    print(f"Indexed {len(documents)} documents from ./docs")


def example_build_from_jsonl() -> None:
    """Build index from JSONL dataset."""
    documents = list(FileLoader.from_jsonl(Path("./documents.jsonl")))

    config = IndexConfig(dim=384, bit_width=4, use_id_map=True)
    builder = IndexBuilder(config, DummyEmbedder(dim=384))
    builder.add_documents(documents)

    index = builder.build_index()
    output_dir = Path("./out/jsonl_index")
    output_dir.mkdir(parents=True, exist_ok=True)
    index.write(str(output_dir / "index.tq"))
    builder.save_metadata(output_dir)

    print(f"Indexed {len(documents)} JSONL records")


def example_build_from_urls() -> None:
    """Build index from web URLs."""
    urls = [
        "https://example.com/a",
        "https://example.com/b",
    ]
    documents = list(URLLoader.from_urls(urls))

    config = IndexConfig(dim=384, bit_width=4, use_id_map=True)
    builder = IndexBuilder(config, DummyEmbedder(dim=384))
    builder.add_documents(documents)

    index = builder.build_index()
    output_dir = Path("./out/url_index")
    output_dir.mkdir(parents=True, exist_ok=True)
    index.write(str(output_dir / "index.tq"))
    builder.save_metadata(output_dir)

    print(f"Indexed {len(documents)} URL documents")


def example_search() -> None:
    """Search a previously saved index."""
    try:
        from turbovec import IdMapIndex
    except ImportError:
        print("Install turbovec to run search example: pip install turbovec")
        return

    index_dir = Path("./out/dir_index")
    config, documents = IndexBuilder.load_metadata(index_dir)
    index = IdMapIndex.load(str(index_dir / "index.tq"))

    searcher = IndexSearcher(index, documents, DummyEmbedder(dim=config.dim))
    results = searcher.search("how to build an index", k=5)

    print("Top results:")
    for result in results:
        preview = result.document.content[:60].replace("\n", " ")
        print(
            f"- rank={result.rank + 1} score={result.score:.3f} id={result.document.id} text={preview}"
        )


if __name__ == "__main__":
    # Uncomment the examples you want to run.
    # example_build_from_directory()
    # example_build_from_jsonl()
    # example_build_from_urls()
    # example_search()
    pass
