"""
Shared pipeline core: the embed -> retrieve -> generate -> evaluate -> log
sequence, as reusable functions. Both the CLI (ask.py) and the Streamlit UI
(app.py) call into this instead of duplicating the logic twice.

Phase 6 (evaluation) and Phase 7 (observability) both live here: every call
to answer_question() returns a faithfulness score alongside the answer, and
logs structured timing/token data for the query.
"""

import json
import os
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

from retrieve import hybrid_retrieve

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"
LOG_PATH = Path("logs/queries.jsonl")
GENERATION_MODEL = "gemini-flash-latest"

load_dotenv()

SYSTEM_PROMPT = """You answer questions using ONLY the provided numbered context chunks.
After every factual claim, cite the chunk number(s) it came from in square brackets, e.g. [1] or [1][2].
If the answer isn't in the context, say so explicitly instead of guessing. Do not use outside knowledge."""

FAITHFULNESS_PROMPT = """You are a strict fact-checker. Given CONTEXT and an ANSWER that claims to be
grounded in it, determine whether every claim in the ANSWER is actually supported by the CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}

Respond with a JSON object: {{"faithfulness_score": <float 0.0 to 1.0>, "unsupported_claims": [<strings>]}}
1.0 means every claim is fully supported by the context. Lower it for any claim that goes beyond, or
contradicts, what the context actually says. unsupported_claims should be empty if the score is 1.0."""


_embedding_model: SentenceTransformer | None = None
_genai_client: genai.Client | None = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def get_genai_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _genai_client


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(COLLECTION_NAME)


def build_numbered_context(chunks: list[str], metadatas: list[dict]) -> str:
    # Including the source filename inline (not just in the post-hoc Sources
    # list) matters for cross-document questions: without it, the model has
    # to guess which chunk belongs to which document from content alone, and
    # can mislabel which entity a fact belongs to even while citing the
    # technically-correct chunk number. Found this the hard way comparing two
    # contracts that don't repeat their own company name in every paragraph.
    return "\n\n".join(
        f"[{i}] (from {meta['source']}) {chunk}" for i, (chunk, meta) in enumerate(zip(chunks, metadatas), 1)
    )


def score_faithfulness(context: str, answer: str) -> dict:
    """LLM-as-judge faithfulness check. A second, independent Gemini call asks
    'does the answer actually say only what the context supports?' — this is
    the same idea Ragas/TruLens formalize, done by hand so the mechanism is
    visible instead of hidden inside a framework."""
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    response = get_genai_client().models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    try:
        return json.loads(response.text)
    except (json.JSONDecodeError, TypeError):
        return {"faithfulness_score": None, "unsupported_claims": [], "error": "could not parse judge response"}


def log_query(record: dict) -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def answer_question(question: str, top_k: int = 3, sources: list[str] | None = None) -> dict:
    """Runs the full pipeline for one question and returns a result dict with
    the answer, per-chunk sources, faithfulness score, and timing/token stats.
    Also appends a log line to logs/queries.jsonl."""
    collection = get_collection()
    if collection.count() == 0:
        raise RuntimeError("No documents ingested yet. Run ingest.py first.")

    t0 = time.perf_counter()
    query_embedding = get_embedding_model().encode([question]).tolist()[0]
    embed_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    results = hybrid_retrieve(collection, question, query_embedding, top_k=top_k, sources=sources)
    retrieve_ms = (time.perf_counter() - t0) * 1000

    chunks = [doc for _, doc, _ in results]
    metadatas = [meta for _, _, meta in results]
    context = build_numbered_context(chunks, metadatas)

    t0 = time.perf_counter()
    response = get_genai_client().models.generate_content(
        model=GENERATION_MODEL,
        contents=f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}",
    )
    generate_ms = (time.perf_counter() - t0) * 1000
    answer = response.text

    usage = getattr(response, "usage_metadata", None)
    input_tokens = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(usage, "candidates_token_count", None)

    t0 = time.perf_counter()
    judgement = score_faithfulness(context, answer)
    evaluate_ms = (time.perf_counter() - t0) * 1000

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": question,
        "sources_filter": sources,
        "num_chunks_retrieved": len(chunks),
        "retrieved_sources": [{"source": m["source"], "page": m["page"]} for m in metadatas],
        "faithfulness_score": judgement.get("faithfulness_score"),
        "unsupported_claims": judgement.get("unsupported_claims", []),
        "timing_ms": {
            "embed": round(embed_ms, 1),
            "retrieve": round(retrieve_ms, 1),
            "generate": round(generate_ms, 1),
            "evaluate": round(evaluate_ms, 1),
            "total": round(embed_ms + retrieve_ms + generate_ms + evaluate_ms, 1),
        },
        "tokens": {"input": input_tokens, "output": output_tokens},
    }
    log_query(record)

    return {
        "answer": answer,
        "metadatas": metadatas,
        "faithfulness_score": judgement.get("faithfulness_score"),
        "unsupported_claims": judgement.get("unsupported_claims", []),
        "timing_ms": record["timing_ms"],
        "tokens": record["tokens"],
    }
