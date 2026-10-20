"""
Phase 5 utility: list every distinct document currently ingested in ChromaDB,
with its chunk count and page range. Useful before writing a document-scoped
or cross-document query.

Usage:
    python src/list_docs.py
"""

import chromadb

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"


def main():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    if collection.count() == 0:
        print("No documents ingested yet. Run ingest.py first.")
        return

    data = collection.get()
    by_source: dict[str, list[int]] = {}
    for meta in data["metadatas"]:
        by_source.setdefault(meta["source"], []).append(meta["page"])

    print(f"{len(by_source)} document(s) ingested:\n")
    for source, pages in sorted(by_source.items()):
        print(f"  {source} — {len(pages)} chunk(s), pages {min(pages)}-{max(pages)}")


if __name__ == "__main__":
    main()
