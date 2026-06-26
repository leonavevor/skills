"""Data loaders for TurboVec indexing.

Load vectors and documents from filesystem (files, directories),
URLs, and streaming sources.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, NamedTuple


class Document(NamedTuple):
    """A document with stable ID and optional metadata."""
    id: int
    content: str
    metadata: dict | None = None
    path: str | None = None


def _positive_stable_id(value: str) -> int:
    return hash(value) & 0x7FFFFFFF


class FileLoader:
    """Load documents from filesystem."""

    @staticmethod
    def from_file(path: Path, encoding: str = "utf-8") -> Document:
        """Load a single file as a document.

        Args:
            path: File path
            encoding: Text encoding (default: utf-8)

        Returns:
            Document with stable ID based on file path hash
        """
        content = path.read_text(encoding=encoding)
        doc_id = _positive_stable_id(str(path.resolve()))
        return Document(
            id=doc_id,
            content=content,
            path=str(path),
            metadata={"filename": path.name, "size_bytes": len(content)},
        )

    @staticmethod
    def supported_extensions() -> set[str]:
        """File extensions this loader supports."""
        return {".txt", ".md", ".json", ".py", ".js", ".yaml", ".yml", ".html"}

    @staticmethod
    def from_directory(
        dirpath: Path,
        recursive: bool = True,
        extensions: set[str] | None = None,
    ) -> Iterator[Document]:
        """Load all documents from a directory.

        Args:
            dirpath: Directory path
            recursive: Traverse subdirectories (default: True)
            extensions: File extensions to load (default: supported_extensions)

        Yields:
            Document for each file
        """
        if extensions is None:
            extensions = FileLoader.supported_extensions()

        pattern = "**/*" if recursive else "*"
        for filepath in dirpath.glob(pattern):
            if not filepath.is_file():
                continue
            if filepath.suffix.lower() not in extensions:
                continue
            try:
                yield FileLoader.from_file(filepath)
            except Exception as e:
                print(f"Warning: Failed to load {filepath}: {e}")

    @staticmethod
    def from_jsonl(path: Path) -> Iterator[Document]:
        """Load documents from JSONL file (one JSON object per line).

        Each line should be a JSON object with at least 'id' and 'content' fields.

        Args:
            path: Path to JSONL file

        Yields:
            Document for each line
        """
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    doc_id = obj.get("id", line_num)
                    content = obj.get("content", "")
                    metadata = {k: v for k, v in obj.items() if k not in ("id", "content")}
                    yield Document(id=doc_id, content=content, metadata=metadata or None)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSONL line {line_num}: {e}")


class URLLoader:
    """Load documents from URLs."""

    @staticmethod
    def from_url(url: str) -> Document | None:
        """Fetch text content from a URL.

        Requires urllib (standard library).
        Returns None on failure.

        Args:
            url: URL to fetch

        Returns:
            Document or None if fetch fails
        """
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=10) as response:
                content = response.read().decode("utf-8", errors="replace")
                doc_id = _positive_stable_id(url)
                return Document(
                    id=doc_id,
                    content=content,
                    path=url,
                    metadata={"url": url, "size_bytes": len(content)},
                )
        except Exception as e:
            print(f"Warning: Failed to fetch {url}: {e}")
            return None

    @staticmethod
    def from_urls(urls: list[str]) -> Iterator[Document]:
        """Fetch text from multiple URLs.

        Args:
            urls: List of URLs

        Yields:
            Document for each successfully fetched URL
        """
        for url in urls:
            doc = URLLoader.from_url(url)
            if doc:
                yield doc


class TextLoader:
    """Load documents from plain text strings."""

    @staticmethod
    def from_text(text: str, doc_id: int | None = None) -> Document:
        """Create a document from a plain text string.

        Args:
            text: Text content
            doc_id: Optional stable ID (default: hash of text)

        Returns:
            Document
        """
        if doc_id is None:
            doc_id = _positive_stable_id(text)
        return Document(id=doc_id, content=text, metadata={"size_bytes": len(text)})

    @staticmethod
    def from_texts(texts: list[str]) -> Iterator[Document]:
        """Create documents from a list of text strings.

        Args:
            texts: List of text strings

        Yields:
            Document for each string
        """
        for i, text in enumerate(texts):
            yield TextLoader.from_text(text, doc_id=i)

