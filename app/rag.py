"""Retrieval over HR policy documents stored in Qdrant, embedded with an Ollama embedding model."""
from functools import lru_cache

import ollama
from qdrant_client import QdrantClient

from . import config

# Some embedding models are trained with task prefixes; using them noticeably improves retrieval.
TASK_PREFIXES = {
    "nomic-embed-text": {"document": "search_document: ", "query": "search_query: "},
    "snowflake-arctic-embed": {"document": "", "query": "Represent this sentence for searching relevant passages: "},
    "mxbai-embed-large": {"document": "", "query": "Represent this sentence for searching relevant passages: "},
}


@lru_cache(maxsize=1)
def ollama_client() -> ollama.Client:
    return ollama.Client(host=config.OLLAMA_HOST)


@lru_cache(maxsize=1)
def qdrant_client() -> QdrantClient:
    if config.QDRANT_URL == ":memory:":
        return QdrantClient(location=":memory:")
    return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)


def embed(texts: list[str], kind: str, model: str | None = None) -> list[list[float]]:
    """Embed texts. `kind` is 'document' or 'query'."""
    model = model or config.EMBED_MODEL
    prefix = next((p[kind] for name, p in TASK_PREFIXES.items() if model.startswith(name)), "")
    return ollama_client().embed(model=model, input=[prefix + t for t in texts]).embeddings


def search_policies(query: str, top_k: int = config.RAG_TOP_K) -> list[dict]:
    vector = embed([query], "query")[0]
    points = qdrant_client().query_points(
        collection_name=config.QDRANT_COLLECTION, query=vector, limit=top_k, with_payload=True,
        score_threshold=config.RAG_MIN_SCORE,
    ).points
    return [{
        "document": p.payload["document"],
        "section": p.payload["section"],
        "text": p.payload["text"],
        "score": round(p.score, 3),
    } for p in points]
