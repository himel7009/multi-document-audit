"""
Phase 3 comparison: run the same question through naive dense-only retrieval
and through hybrid (dense + BM25 + rerank) retrieval, side by side, so the
difference is visible rather than theoretical.

Usage:
    python src/compare_retrieval.py "your question"
"""

import sys

import chromadb
from sentence_transformers import SentenceTransformer

from retrieve import hybrid_retrieve

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"
TOP_K = 3


def naive_dense_retrieve(collection, query_embedding, top_k):
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return list(zip(results["ids"][0], results["documents"][0], results["metadatas"][0]))


def print_results(label, results):
    print(f"\n--- {label} ---")
    for i, (chunk_id, chunk, meta) in enumerate(results, 1):
        preview = chunk.replace("\n", " ")[:150]
        print(f"[{i}] ({meta['source']}, page {meta['page']}) {preview}...")


def main():
    if len(sys.argv) != 2:
        print('Usage: python src/compare_retrieval.py "your question"')
        sys.exit(1)

    question = sys.argv[1]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([question]).tolist()[0]

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    if collection.count() == 0:
        print("No documents found. Run ingest.py first.")
        sys.exit(1)

    naive_results = naive_dense_retrieve(collection, query_embedding, TOP_K)
    hybrid_results = hybrid_retrieve(collection, question, query_embedding, top_k=TOP_K)

    print(f"Question: {question}")
    print_results("Naive (dense embeddings only)", naive_results)
    print_results("Hybrid (dense + BM25, RRF-fused, cross-encoder re-ranked)", hybrid_results)


if __name__ == "__main__":
    main()
