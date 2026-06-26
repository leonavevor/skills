"""Generate framework integration scaffolds."""

from __future__ import annotations

from pathlib import Path


LANGCHAIN_SCAFFOLD = '''"""TurboVec + LangChain integration example."""

from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import TurboVecStore
from langchain.schema import Document

# Load documents
documents = [
    Document(page_content="Your document 1", metadata={"source": "doc1"}),
    Document(page_content="Your document 2", metadata={"source": "doc2"}),
]

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vector store
vectorstore = TurboVecStore.from_documents(
    documents=documents,
    embedding=embeddings,
)

# Search
results = vectorstore.similarity_search("query text", k=5)

for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
'''

LLAMAINDEX_SCAFFOLD = '''"""TurboVec + LlamaIndex integration example."""

from pathlib import Path
from llama_index.vector_stores.turbovec import TurboVecStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding

# Load documents
documents = SimpleDirectoryReader(input_dir="./data").load_data()

# Create embeddings
embed_model = OpenAIEmbedding()

# Create vector store and index
vector_store = TurboVecStore()
index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store,
    embed_model=embed_model,
)

# Create retriever and search
retriever = index.as_retriever(similarity_top_k=5)
results = retriever.retrieve("query text")

for node in results:
    print(f"Score: {node.score}")
    print(f"Content: {node.get_content()}")
'''

HAYSTACK_SCAFFOLD = '''"""TurboVec + Haystack integration example."""

from pathlib import Path
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever

# Create document store
docstore = InMemoryDocumentStore()

# Add documents
documents = [
    Document(content="Your document 1"),
    Document(content="Your document 2"),
]
docstore.write_documents(documents)

# Create retriever
retriever = InMemoryBM25Retriever(docstore)

# Search
results = retriever.run(query="query text", top_k=5)

for doc in results["documents"]:
    print(f"Content: {doc.content}")
'''

AGNO_SCAFFOLD = '''"""TurboVec + Agno integration example."""

from pathlib import Path
from agno.vectordb.lancedb import LanceDb
from agno.agent import Agent

# Create vector database
vector_db = LanceDb(table_name="documents")

# Add documents with embeddings
documents = [
    {"id": 1, "content": "Your document 1"},
    {"id": 2, "content": "Your document 2"},
]

# Create agent with vector DB
agent = Agent(
    agent_id="turbovec-agent",
    name="TurboVec RAG Agent",
    model="gpt-4",
    vectordb=vector_db,
)

# Search and retrieve
results = agent.retrieve("query text", top_k=5)
for result in results:
    print(f"Score: {result.score}")
    print(f"Content: {result.content}")
'''

SCAFFOLDS = {
    "langchain": LANGCHAIN_SCAFFOLD,
    "llamaindex": LLAMAINDEX_SCAFFOLD,
    "haystack": HAYSTACK_SCAFFOLD,
    "agno": AGNO_SCAFFOLD,
}


def run(framework: str, output_path: Path) -> None:
    """Generate integration scaffold for a framework."""
    if framework not in SCAFFOLDS:
        raise ValueError(f"Unknown framework: {framework}")

    code = SCAFFOLDS[framework]
    output_path.write_text(code)
    print(f"Integration scaffold for {framework} written to {output_path}")

