"""Vector search / hybrid retrieval for RAG."""


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Return top-k chunks for a query (stub)."""
    raise NotImplementedError("Implement with Vector Search or SQL similarity.")
