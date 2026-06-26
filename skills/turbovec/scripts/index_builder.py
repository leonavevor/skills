"""Index builder utilities for TurboVec.

Build and manage TurboVec indexes from documents and embeddings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data_loaders import Document


class EmbeddingProvider:
    """Abstract base for embedding providers."""

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Vector embedding (list of floats)
        """
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts.

        Args:
            texts: List of text strings

        Returns:
            List of vector embeddings
        """
        raise NotImplementedError


class DummyEmbedder(EmbeddingProvider):
    """Dummy embedder for testing (returns deterministic hashes)."""

    def __init__(self, dim: int = 384):
        """Initialize with embedding dimension."""
        self.dim = dim

    def embed_text(self, text: str) -> list[float]:
        """Create a deterministic embedding from text hash."""
        h = hash(text)
        import random
        random.seed(abs(h))
        return [random.gauss(0, 1) for _ in range(self.dim)]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        return [self.embed_text(t) for t in texts]


class IndexConfig:
    """Configuration for TurboVec index building."""

    def __init__(
        self,
        dim: int = 1536,
        bit_width: int = 4,
        use_id_map: bool = True,
    ):
        """Initialize index config.

        Args:
            dim: Vector dimensionality (default: 1536, typical for OpenAI embeddings)
            bit_width: Quantization bits (2 or 4, default: 4)
            use_id_map: Use IdMapIndex for stable ID tracking (default: True)
        """
        if bit_width not in (2, 4):
            raise ValueError(f"bit_width must be 2 or 4, got {bit_width}")
        if dim < 32:
            raise ValueError(f"dim must be >= 32, got {dim}")

        self.dim = dim
        self.bit_width = bit_width
        self.use_id_map = use_id_map

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "dim": self.dim,
            "bit_width": self.bit_width,
            "use_id_map": self.use_id_map,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndexConfig:
        """Deserialize from dict."""
        return cls(**data)


class IndexBuilder:
    """Build TurboVec indexes from documents and embeddings."""

    def __init__(self, config: IndexConfig, embedder: EmbeddingProvider):
        """Initialize index builder.

        Args:
            config: Index configuration
            embedder: Embedding provider
        """
        self.config = config
        self.embedder = embedder
        self.documents: list[Document] = []
        self.embeddings: list[list[float]] = []
        self.id_map: dict[int, int] = {}  # doc_id -> vector_index

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the index.

        Args:
            documents: Documents to add
        """
        # Embed documents
        texts = [doc.content for doc in documents]
        embeddings = self.embedder.embed_batch(texts)

        if len(embeddings) != len(documents):
            raise ValueError("Embedder returned wrong number of embeddings")

        # Track documents and embeddings
        for doc, emb in zip(documents, embeddings):
            vec_idx = len(self.embeddings)
            self.documents.append(doc)
            self.embeddings.append(emb)
            self.id_map[doc.id] = vec_idx

    def build_index(self) -> Any:
        """Build and return a TurboVec index.

        Requires: turbovec package

        Returns:
            TurboQuantIndex or IdMapIndex instance
        """
        try:
            import numpy as np
            from turbovec import TurboQuantIndex, IdMapIndex
        except ImportError:
            raise ImportError(
                "turbovec not installed. Install with: pip install turbovec"
            )

        # Convert to numpy array
        vectors = np.array(self.embeddings, dtype=np.float32)

        if not self.config.use_id_map:
            # Simple index
            index = TurboQuantIndex(dim=self.config.dim, bit_width=self.config.bit_width)
            index.add(vectors)
            return index
        else:
            # Index with stable IDs
            index = IdMapIndex(dim=self.config.dim, bit_width=self.config.bit_width)
            ids = np.array([doc.id for doc in self.documents], dtype=np.uint64)
            index.add_with_ids(vectors, ids)
            return index

    def save_metadata(self, dirpath: Path) -> None:
        """Save index metadata (documents, config) to directory.

        Args:
            dirpath: Directory path
        """
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)

        # Save config
        config_file = dirpath / "config.json"
        config_file.write_text(json.dumps(self.config.to_dict(), indent=2))

        # Save documents as JSONL
        docs_file = dirpath / "documents.jsonl"
        with docs_file.open("w") as f:
            for doc in self.documents:
                obj = {
                    "id": doc.id,
                    "content": doc.content,
                    "path": doc.path,
                    "metadata": doc.metadata,
                }
                f.write(json.dumps(obj) + "\n")

        # Save ID map
        idmap_file = dirpath / "id_map.json"
        idmap_file.write_text(json.dumps(self.id_map, indent=2))

    @staticmethod
    def load_metadata(dirpath: Path) -> tuple[IndexConfig, list[Document]]:
        """Load index metadata from directory.

        Args:
            dirpath: Directory path

        Returns:
            (IndexConfig, list of Documents)
        """
        dirpath = Path(dirpath)

        # Load config
        config_file = dirpath / "config.json"
        config_data = json.loads(config_file.read_text())
        config = IndexConfig.from_dict(config_data)

        # Load documents
        docs_file = dirpath / "documents.jsonl"
        documents = []
        with docs_file.open("r") as f:
            for line in f:
                obj = json.loads(line)
                doc = Document(
                    id=obj["id"],
                    content=obj["content"],
                    path=obj.get("path"),
                    metadata=obj.get("metadata"),
                )
                documents.append(doc)

        return config, documents


class SearchResult:
    """Result of a search query."""

    def __init__(
        self,
        document: Document,
        score: float,
        rank: int,
    ):
        """Initialize search result.

        Args:
            document: The matched document
            score: Similarity score
            rank: Rank in result set (0-indexed)
        """
        self.document = document
        self.score = score
        self.rank = rank

    def __repr__(self) -> str:
        return f"SearchResult(id={self.document.id}, score={self.score:.3f}, rank={self.rank})"


class IndexSearcher:
    """Search a built TurboVec index."""

    def __init__(
        self,
        index: Any,
        documents: list[Document],
        embedder: EmbeddingProvider,
    ):
        """Initialize searcher.

        Args:
            index: Built TurboVec index (TurboQuantIndex or IdMapIndex)
            documents: List of indexed documents
            embedder: Embedding provider
        """
        self.index = index
        self.documents = {doc.id: doc for doc in documents}
        self.embedder = embedder

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Search the index.

        Args:
            query: Query text
            k: Number of results (default: 10)

        Returns:
            List of SearchResult objects
        """
        # Embed query
        query_embedding = self.embedder.embed_text(query)

        # Search index
        import numpy as np
        query_vec = np.array([query_embedding], dtype=np.float32)
        scores, indices_or_ids = self.index.search(query_vec, k=k)

        # Unpack results
        results = []
        scores_flat = scores[0] if getattr(scores, "ndim", 1) > 1 else scores
        indices_flat = indices_or_ids[0] if getattr(indices_or_ids, "ndim", 1) > 1 else indices_or_ids

        for rank, (score, idx_or_id) in enumerate(zip(scores_flat, indices_flat)):
            # Find document by ID
            doc = self.documents.get(int(idx_or_id))
            if doc:
                results.append(SearchResult(document=doc, score=float(score), rank=rank))

        return results

