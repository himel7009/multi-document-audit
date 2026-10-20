# Multi-Document Audit & Grounded RAG

I'm building a production-style Retrieval-Augmented Generation (RAG) system for auditing long documents
(think: 10-Ks, policy documents, vendor contracts) — not another "upload a PDF, ask a question" demo.

Most RAG demos trust whatever text the vector database hands back and never check if the answer is actually
grounded in the source. This project is my attempt to build the version that doesn't: paragraph-level
citations, hybrid retrieval, and an evaluation layer that scores every answer for faithfulness before showing
it to a user.

## Why I'm building this

I'm learning AI/ML engineering by building real systems, not just going through tutorials. I know Python and
the theory behind embeddings, vector search, and deep learning, but I haven't shipped something like this
before. So this repo is being built incrementally and in the open — the code, the commit history, and the
roadmap below are the actual learning log, not a polished final product pretending to have appeared fully formed.

If you're reading this on GitHub: this is a work in progress, built one phase at a time. Check
[`ROADMAP.md`](./ROADMAP.md) to see exactly where it stands and what's coming next.

## What this system does

- Ingests multiple long documents (`.txt`/`.pdf`) with paragraph-aware chunking.
- Retrieves relevant passages using **hybrid search** (dense embeddings + BM25 keyword search + cross-encoder
  re-ranking), optionally scoped to specific documents.
- Answers questions with **citations down to the paragraph level**, correctly attributed even across multiple
  documents.
- Scores every answer for **faithfulness** using a second, independent LLM call before showing it to a user.
- Logs latency (per pipeline stage) and token spend for every query.
- Ships with both a CLI and a **Streamlit UI** for uploading documents and asking questions.

## Current status: all 9 planned phases complete

Every phase in [`ROADMAP.md`](./ROADMAP.md) is done — that file is the detailed log of what was built in each
phase, what broke along the way, and what I learned fixing it. This README covers how to run it and the
overall architecture; the roadmap covers the *process*.

### Project structure

```
Multi_Document_Audit/
├── app.py                   Streamlit UI — upload docs, ask questions, see citations + faithfulness + timing
├── data/                    sample documents used for local testing
├── logs/                    per-query JSONL logs (gitignored — local only)
├── src/
│   ├── ingest.py            load a .txt/.pdf, chunk by paragraph, embed, store in ChromaDB
│   ├── retrieve.py          hybrid retrieval: dense + BM25 -> RRF fusion -> cross-encoder rerank
│   ├── pipeline.py          shared core: embed -> retrieve -> generate -> evaluate -> log (used by CLI + UI)
│   ├── ask.py               CLI: ask a question (optionally scoped to specific docs with --doc)
│   ├── list_docs.py         list what's currently ingested
│   └── compare_retrieval.py naive dense-only vs hybrid+rerank, side by side
├── requirements.txt
├── .env.example
├── ROADMAP.md               the phased build plan — the real log of what was built, broke, and was learned
├── STUDY_GUIDE_PHASE0.md    concepts + official docs to read before/alongside building
└── README.md
```

### Setup

I'm using my existing `ai_env_v2` conda environment for this (already has `torch` set up) rather than a new
venv, to avoid redownloading the GPU-enabled torch build.

```bash
conda activate ai_env_v2
pip install -r requirements.txt

cp .env.example .env
# then add your GEMINI_API_KEY to .env (free tier, from aistudio.google.com)
```

If you're following along without that environment, any Python 3.10+ venv works the same way:
`python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

### Running it

```bash
# 1. ingest documents into the local vector store (.txt or .pdf)
python src/ingest.py data/sample.txt
python src/ingest.py data/vendor_contract_a.txt
python src/ingest.py data/vendor_contract_b.txt

# 2. see what's ingested
python src/list_docs.py

# 3. ask a question (searches across all ingested documents)
python src/ask.py "What does this document say about onboarding?"

# ...or scope it to specific document(s)
python src/ask.py "What's the cure period?" --doc vendor_contract_a.txt
```

Every answer prints its citations, a faithfulness score (0.0-1.0, from a second LLM call checking the first
one's work), and per-stage timing/token counts. Every query is also logged to `logs/queries.jsonl`.

Or use the UI instead of the CLI:

```bash
streamlit run app.py
```

Upload documents in the sidebar, optionally scope a question to specific ones, and ask — the UI calls the
exact same `src/pipeline.py` functions as the CLI, so there's no second implementation to keep in sync.

### Stack so far

- **Ingestion:** `pypdf` for PDF text extraction, plus a hand-written paragraph-aware chunker (see
  `src/ingest.py`) — packs whole paragraphs into chunks instead of cutting at a fixed character count.
- **Embeddings:** `sentence-transformers` (local, free, no API key needed) — I wanted the embedding step to
  work without depending on an external API while I'm learning the mechanics.
- **Vector store:** ChromaDB, running locally on disk. Every chunk is stored with its source document and
  page number as metadata.
- **Retrieval:** hybrid — dense embedding search + BM25 keyword search (`rank-bm25`), merged with Reciprocal
  Rank Fusion, then re-ranked with a local cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Can be
  scoped to specific documents via a Chroma metadata filter. See `src/retrieve.py`.
- **LLM:** Google Gemini (`gemini-flash-latest`), for both answer generation and faithfulness evaluation —
  using the Google AI Studio free tier rather than a paid API while I'm learning and testing.
- **Evaluation:** a second, independent Gemini call acts as a fact-checker on the first call's answer,
  returning a faithfulness score and any unsupported claims as structured JSON. See
  `src/pipeline.py: score_faithfulness`.
- **Observability:** every query logs embed/retrieve/generate/evaluate timing and token counts to
  `logs/queries.jsonl` (local only, gitignored).

This will evolve — see the roadmap for what's being swapped in and why.

## Architecture & tradeoffs

**Why hybrid retrieval instead of dense-only.** Dense embeddings match on meaning, which is exactly what
makes them miss exact terms — a specific dollar figure, a clause number, a company name — because those don't
always dominate an embedding vector. BM25 catches what dense search misses, at the cost of missing paraphrases
dense search would catch. Fusing both with Reciprocal Rank Fusion and re-ranking the result gets most of the
benefit of each without manual weight-tuning between two differently-scaled scoring systems.

**Why local models over paid APIs where possible.** Embeddings (`sentence-transformers`) and re-ranking
(`cross-encoder/ms-marco-MiniLM-L-6-v2`) both run locally and free. Only generation and the faithfulness judge
call out to Gemini's free tier. This kept iteration fast (no per-call cost while debugging) and is also just a
reasonable default for a learning project with no budget — see `STUDY_GUIDE_PHASE0.md` for why these
specific tools.

**Why citations needed the source filename in the generation prompt, not just afterward.** Found this the
hard way in Phase 5: showing the model numbered chunks without saying which document each one came from let
it cite the technically-correct chunk while mislabeling which real-world entity (which contract) a fact
belonged to. The fix — `(from <filename>)` inline in every chunk shown to the model — is a small change with
an outsized effect on multi-document correctness.

**Why a hand-rolled faithfulness judge instead of Ragas/TruLens.** Both frameworks wrap this same
LLM-as-judge pattern (plus more metrics: context precision, context recall) behind real framework surface
area. Building the simplest version by hand first — one prompt, one JSON response — made the actual mechanism
visible rather than opaque, which mattered more for learning than having every metric from day one. This is
the single clearest thing I'd point to if asked "prove you understand what's happening, not just how to call
a library."

**What I'd do differently with more time:** add context precision/recall (not just faithfulness) to catch
retrieval-quality issues the current judge doesn't check for; add a real 10-K or long contract to the test set
to see hybrid retrieval's advantage show up more clearly than it does on the current small corpus; swap the
hand-written chunker for a table-aware parser (Unstructured/LlamaIndex) once a document with actual tables is
in the test set, since the current paragraph-based chunker has no concept of tabular structure.

## Notes for anyone reading this

This is a personal learning project, built in public. Code quality and design choices reflect where I am in
the learning process at each phase — later phases should look noticeably more deliberate than earlier ones,
and that's intentional: I'd rather show the real progression than fake a finished-looking v1.
