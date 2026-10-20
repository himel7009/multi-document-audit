"""
Phase 3 retrieval: hybrid search (dense embeddings + BM25 keyword search),
fused with Reciprocal Rank Fusion, then re-ranked with a cross-encoder.

Why hybrid: dense embeddings are great at matching *meaning* but can miss
exact terms — a specific product name, a dollar figure, an acronym — because
those don't always carry much weight in an embedding. BM25 is the opposite:
it's just weighted keyword overlap, so it catches exact matches dense search
misses, but it has no notion of meaning or paraphrase. Combining both and
re-ranking the combined candidates gets the benefits of each.
"""

import re

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60  # standard smoothing constant for reciprocal rank fusion


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def dense_search(collection, query_embedding, n: int, where: dict | None = None) -> list[str]:
    """Return chunk ids ranked by embedding similarity."""
    results = collection.query(query_embeddings=[query_embedding], n_results=n, where=where)
    return results["ids"][0]


def bm25_search(ids: list[str], documents: list[str], query: str, n: int) -> list[str]:
    """Return chunk ids ranked by BM25 keyword score."""
    tokenized_corpus = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(zip(ids, scores), key=lambda pair: pair[1], reverse=True)
    return [chunk_id for chunk_id, _ in ranked[:n]]


def reciprocal_rank_fusion(*ranked_id_lists: list[str]) -> list[str]:
    """Merge multiple ranked lists into one, favoring ids that rank highly in
    more than one list. Score = sum of 1/(k + rank) across all lists."""
    scores: dict[str, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, chunk_id in enumerate(ranked_ids):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores, key=scores.get, reverse=True)


_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def rerank(query: str, candidates: list[tuple[str, str, dict]], top_k: int):
    """candidates: list of (id, document_text, metadata). Returns the top_k
    re-sorted by actual cross-encoder relevance to the query."""
    pairs = [(query, doc) for _, doc, _ in candidates]
    scores = get_reranker().predict(pairs)
    reranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [candidate for candidate, _ in reranked[:top_k]]


def hybrid_retrieve(
    collection,
    query: str,
    query_embedding,
    top_k: int,
    candidate_pool: int = 10,
    sources: list[str] | None = None,
):
    """Full Phase 3+5 pipeline: dense + BM25 -> RRF fusion -> cross-encoder rerank.

    `sources`, if given, restricts retrieval to chunks from those document
    filenames only — this is the metadata filtering Phase 5 needs for
    document-scoped queries (e.g. "what does contract A say about X").
    Without it, retrieval naturally searches across every ingested document,
    which is what makes cross-document questions ("compare A and B") work.

    Returns a list of (id, document_text, metadata) tuples, length top_k.
    """
    where = {"source": {"$in": sources}} if sources else None

    all_data = collection.get(where=where)
    all_ids, all_docs, all_metas = all_data["ids"], all_data["documents"], all_data["metadatas"]

    if not all_ids:
        return []

    dense_ids = dense_search(collection, query_embedding, candidate_pool, where=where)
    bm25_ids = bm25_search(all_ids, all_docs, query, candidate_pool)
    fused_ids = reciprocal_rank_fusion(dense_ids, bm25_ids)[:candidate_pool]

    id_to_doc = dict(zip(all_ids, all_docs))
    id_to_meta = dict(zip(all_ids, all_metas))
    candidates = [(cid, id_to_doc[cid], id_to_meta[cid]) for cid in fused_ids]

    return rerank(query, candidates, top_k)
