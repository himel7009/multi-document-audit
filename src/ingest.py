"""
Phase 2 ingestion: load a .txt or .pdf document, chunk it paragraph-by-paragraph
(never splitting a paragraph mid-sentence unless it's longer than one chunk on
its own), embed the chunks, and store them in a local ChromaDB collection.

Each chunk keeps track of which page it came from — this is what Phase 4
(paragraph-level citations) will build on.

Usage:
    python src/ingest.py data/sample.txt
    python src/ingest.py ai_project_ideas_clean.pdf
"""

import sys
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"
CHUNK_SIZE = 500  # target characters per chunk


def load_pages(doc_path: Path) -> list[tuple[int, str]]:
    """Return a list of (page_number, page_text). Page numbers start at 1."""
    if doc_path.suffix.lower() == ".pdf":
        reader = PdfReader(doc_path)
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    # Plain text has no real page concept, so treat the whole file as page 1.
    return [(1, doc_path.read_text(encoding="utf-8"))]


def split_into_paragraphs(page_text: str) -> list[str]:
    # Blank lines are the cheapest real structural signal we have without a
    # layout-aware parser — this is what keeps "Section 3: Expense..." together
    # with its own paragraph instead of fusing it onto the previous one.
    raw_paragraphs = page_text.split("\n\n")
    return [p.strip() for p in raw_paragraphs if p.strip()]


def pack_paragraphs_into_chunks(paragraphs: list[str], chunk_size: int) -> list[str]:
    """Greedily fill chunks with whole paragraphs. A paragraph is only split if
    it's longer than chunk_size on its own — chunking never cuts a normal
    paragraph in half."""
    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(paragraph), chunk_size):
                chunks.append(paragraph[start : start + chunk_size])
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) > chunk_size:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def ingest_document(doc_path: Path, embedding_model: SentenceTransformer | None = None) -> int:
    """Ingest one document into the shared ChromaDB collection. Returns the
    number of chunks stored. Shared by the CLI and the Streamlit UI so both
    go through the exact same chunking/embedding logic."""
    pages = load_pages(doc_path)

    chunks: list[str] = []
    page_numbers: list[int] = []
    for page_number, page_text in pages:
        paragraphs = split_into_paragraphs(page_text)
        page_chunks = pack_paragraphs_into_chunks(paragraphs, CHUNK_SIZE)
        chunks.extend(page_chunks)
        page_numbers.extend([page_number] * len(page_chunks))

    model = embedding_model or SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks).tolist()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    ids = [f"{doc_path.stem}-{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": doc_path.name, "chunk_index": i, "page": page_numbers[i]}
        for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(chunks)


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/ingest.py <path-to-document>")
        sys.exit(1)

    doc_path = Path(sys.argv[1])
    pages = load_pages(doc_path)
    print(f"Splitting '{doc_path.name}' ({len(pages)} page(s))...")

    num_chunks = ingest_document(doc_path)
    print(f"Stored {num_chunks} chunks in ChromaDB collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
