# Roadmap

This is the phased plan I'm following to build this project. Each phase ends with something runnable, so
progress is always demonstrable, not just theoretical. I'm checking phases off as I finish them.

## Phase 0 — Setup & mental model
- [x] Environment set up (Python, venv, Git).
- [x] Understand the basic RAG loop: embed → store → retrieve → generate.

## Phase 1 — Naive RAG baseline (done)
- [x] Ingest a single document, chunk it with fixed-size chunking.
- [x] Embed chunks locally with `sentence-transformers`, store in ChromaDB.
- [x] Retrieve top-k chunks by cosine similarity, stuff into an LLM prompt, return an answer.
- [x] Verified end to end: ingested `data/sample.txt`, asked a question about remote work policy, got a
      correct, grounded answer from Gemini using only the retrieved chunks.
- [ ] Try it on a few more question types and note where naive retrieval clearly fails (this will motivate Phase 3).

**Concepts I needed to learn here:** what an embedding actually is, cosine similarity vs dot product, why
chunk size and overlap change retrieval quality, basic prompt construction for grounded answers.

## Phase 2 — Real document ingestion & chunking (done)
- [x] Added PDF support (`pypdf`) alongside plain text.
- [x] Replaced fixed-size chunking with paragraph-aware chunking: chunks are packed from whole paragraphs
      (split on blank lines) and only split mid-paragraph if a single paragraph exceeds the chunk size.
      Deliberately skipped the `Unstructured` library here — it needs system-level deps (poppler/tesseract)
      that aren't worth the install weight yet, and writing the paragraph packer by hand made the "why does
      naive chunking break things" lesson concrete instead of hidden inside a library call.
- [x] Every chunk now carries its source page number in metadata — sets up Phase 4 citations.
- [x] Tested on a real two-page PDF (`ai_project_ideas_clean.pdf`) with two distinct sections — confirmed
      retrieval correctly pulled only page-2 chunks for a page-2-specific question.
- [ ] Revisit `Unstructured`/table-aware parsing once we test against a document with actual tables (a 10-K will need this).

**Concepts learned here:** why cutting a document into fixed-size windows can slice a sentence or section in
half, and why "chunk by paragraph, fall back to splitting only when necessary" is a cheap but real improvement
over naive chunking. Also hit a real-world lesson outside the plan: LLM provider model names/aliases drift
over time — `gemini-2.5-flash` was deprecated mid-project, and even its suggested replacement
(`gemini-3.8-flash`) returned 503s under load. Settled on `gemini-flash-latest`, a stable alias Google
maintains to always point at a current, available flash-tier model — more resilient than hardcoding a
specific version.

## Phase 3 — Hybrid retrieval + re-ranking (done)
- [x] Added BM25 sparse keyword search (`rank-bm25`) alongside dense embedding search.
- [x] Combined dense + BM25 results with Reciprocal Rank Fusion (RRF), then re-ranked the fused candidates
      with a local cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Used a local reranker instead of
      Cohere Rerank to stay consistent with the no-extra-API-key approach from earlier phases.
- [x] Built `src/compare_retrieval.py` — runs the same question through naive dense-only and hybrid+rerank
      side by side.
- [x] Ran the comparison on the current (small, ~10-chunk) corpus: both pipelines found the right answer at
      rank 1, with only minor reshuffling further down. Honest finding: hybrid's advantage is subtle on a
      small/clean corpus — expect a bigger, more obvious gap once Phase 5 adds a larger, messier document
      where dense search is more likely to miss exact terms.

**Concepts learned here:** why dense-only retrieval can miss exact keyword/number matches (BM25 catches what
pure meaning-matching doesn't), what Reciprocal Rank Fusion does and why it needs no manual weight tuning
between two very differently-scaled ranking signals, and what a cross-encoder reranker is doing differently
from the embedding similarity used for initial retrieval (it scores the actual query+passage pair jointly,
instead of comparing two independently-computed vectors — slower, but much more precise).

## Phase 4 — Paragraph-level citations (done)
- [x] Chunk metadata (document, page) was already tracked through ingestion and retrieval since Phase 2.
- [x] Numbered the retrieved chunks in the prompt (`[1]`, `[2]`, ...) and instructed Gemini to tag every claim
      with the chunk number(s) it came from, then printed a "Sources" legend mapping each number back to its
      document + page.
- [x] Caught a real example of this working as intended: a test question asked about remote work policy, but
      the only retrieved chunk touching that section was just its heading (no body text) — the model cited
      `[3]` and explicitly said it didn't have the detail, instead of inventing the policy from general
      knowledge. Citations made the "why didn't it answer fully" question immediately traceable rather than a
      mystery.

## Phase 5 — Multi-document support (done)
- [x] Document-level metadata (`source`, `page`) was already stored since Phase 2.
- [x] Added `--doc file1.txt,file2.pdf` filtering to `ask.py` (via a Chroma `where` metadata filter in
      `retrieve.hybrid_retrieve`), for scoping a question to specific documents.
- [x] Added `src/list_docs.py` to see what's currently ingested (source, chunk count, page range) before
      writing a query.
- [x] Added two realistic sample documents (`data/vendor_contract_a.txt`, `data/vendor_contract_b.txt`) with
      differing termination/payment/liability terms, to properly exercise cross-document comparison — closer
      to the "legal/finance" use case this project is actually about than the earlier HR-handbook sample.
- [x] Found and fixed a real cross-document bug: asking the model to compare two contracts, it cited the
      correct chunks but mislabeled *which contract* a fact belonged to ("First Contract" vs "Second
      Contract" didn't match the actual source files). Root cause: the numbered context sent to the model
      during generation didn't include the source filename per chunk — only the post-hoc "Sources" list did,
      so the model had to guess document identity from content alone. Fix: include `(from <filename>)` inline
      in every numbered context chunk. Verified the same query again afterward — correctly attributed.

**Concepts learned here:** metadata filtering in vector databases (Chroma's `where` clause), and a subtler
lesson not in the original plan — citation *correctness* (pointing to the right chunk) and attribution
*correctness* (describing the right entity) are different failure modes, and fixing one doesn't fix the other.
The model needs the identifying information in-context at generation time, not just displayed afterward.

## Phase 6 — Evaluation & hallucination guardrails (done)
- [x] Implemented faithfulness scoring as a second, independent Gemini call acting as a judge: given the
      retrieved context and the generated answer, it returns a `faithfulness_score` (0.0-1.0) and a list of
      any unsupported claims, as structured JSON (`src/pipeline.py: score_faithfulness`).
- [x] Deliberately skipped Ragas/TruLens here — same reasoning as skipping Unstructured in Phase 2: those
      frameworks wrap this same LLM-as-judge idea (plus additional metrics like context precision/recall)
      behind a fair amount of framework surface area, and hand-rolling the simplest version first made the
      actual mechanism visible instead of a black box. Revisit Ragas/TruLens later if more metrics are
      needed (context precision/recall specifically).
- [x] `ask.py` now prints the faithfulness score and any unsupported claims under every answer.

**Concepts learned here:** "LLM-as-judge" as a pattern — using a second model call, with a narrow and
specific prompt, to evaluate the first call's output. Also: the faithfulness score caught content-level
hallucination (did the answer say something the context doesn't support) but did NOT catch the entity
mislabeling bug found in Phase 5 — that bug produced a technically-supported claim attached to the wrong
label, which this particular check wasn't designed to catch. Good reminder that one metric doesn't catch
every failure mode.

## Phase 7 — Observability (done)
- [x] Every query now logs structured JSON to `logs/queries.jsonl` (gitignored — local only): timestamp,
      question, which documents/chunks were retrieved, faithfulness score, per-stage timing (embed/retrieve/
      generate/evaluate, in ms), and token counts (input/output) from Gemini's `usage_metadata`.
- [x] `ask.py` prints a timing + token summary line after every answer.
- [x] This surfaced a real finding immediately: the free-tier Gemini calls (generate + evaluate, two separate
      calls per question) dominate total latency — tens of seconds each under load — while local
      embedding/retrieval is consistently fast (single-digit ms to low seconds). The logging didn't just
      satisfy a checklist item, it answered "where does the time actually go" with real numbers instead of a
      guess.

**Concepts learned here:** where latency and cost actually concentrate in a RAG pipeline in practice (network
calls to the LLM provider dominate; local compute is comparatively cheap), and why running an evaluator model
call roughly doubles your LLM cost per query — a real tradeoff to be aware of before enabling it by default
in anything latency-sensitive.

## Phase 6 — Evaluation & hallucination guardrails
- [ ] Integrate Ragas or TruLens: context precision, context recall, faithfulness scoring.
- [ ] Add a confidence threshold — low-faithfulness answers get flagged or withheld instead of shown as fact.

**Concepts to learn:** what each RAG evaluation metric actually measures, how automated hallucination/faithfulness
scoring works under the hood.

## Phase 7 — Observability
- [ ] Log latency per pipeline stage, tokens in/out, estimated cost, and retrieval scores for every query.

**Concepts to learn:** where time and cost actually go in a RAG pipeline in practice.

## Phase 8 — UI (done)
- [x] Built `app.py`, a Streamlit front end: upload `.txt`/`.pdf` documents, see what's ingested, ask
      questions (optionally scoped to selected documents), and see the answer, citations, faithfulness score,
      and timing/token stats.
- [x] The UI calls the exact same `src/pipeline.py` and `src/ingest.py` functions the CLI uses — no logic is
      duplicated between the two front ends, which also means every improvement to retrieval/evaluation from
      earlier phases is automatically reflected here with zero extra work.
- [x] Smoke-tested: app boots cleanly, serves HTTP 200, health check passes.
- [ ] (Stretch, not done) Swap to a FastAPI backend + small frontend if I want it to look more production-grade.

## Phase 9 — Polish (done)
- [x] Wrote up the architecture and the tradeoffs hit along the way, as an "Architecture & tradeoffs" section
      in `README.md` — why hybrid retrieval, why local models where possible, the source-filename citation
      bug and fix, why a hand-rolled faithfulness judge instead of Ragas/TruLens, and honest "what I'd do
      differently with more time."
- [x] Final pass on README (project structure, running instructions for both CLI and UI, full stack list) and
      this roadmap.
- [x] Verified the whole repo end to end one more time: ingestion (txt + PDF, 4 documents), hybrid retrieval,
      citations with correct cross-document attribution, faithfulness scoring, logging, and the Streamlit UI
      all working together, not just individually.

---

All 9 planned phases are complete. This file stays as the real build log — what was built, what broke, what
got fixed and why — rather than being rewritten to look like the finished state was the plan from day one.
If I come back to extend this project, new work gets appended here rather than blended into the existing
phases, so the history stays honest.
