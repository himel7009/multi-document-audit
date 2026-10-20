# Phase 0 Study Guide — Concepts Before We Build

This is my reading list before writing more code. I'm not trying to become an expert in everything here —
just enough to understand *what the code I'm reading is actually doing* and *why it's built this way*, instead
of copy-pasting blindly. Each section below has: what it is, why it matters for this project, and an official
source to read.

---

## 1. Environment & tooling

### Virtual environments (conda / venv)
**What it is:** An isolated Python installation with its own set of installed packages, separate from your
system Python or other projects. Prevents one project's dependencies from breaking another's.
**Why it matters here:** We're using the `ai_env_v2` conda environment. Every time I open a new terminal to
work on this project, I need to run `conda activate ai_env_v2` first — otherwise Python won't find any of the
packages we installed (chromadb, sentence-transformers, etc.), because they're only installed *inside* that
environment, not system-wide.
**Read:** https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html

### Environment variables & `.env` files
**What it is:** A way to store secrets (like API keys) outside of your actual code, so they never get
committed to GitHub by accident. The `python-dotenv` package reads a local `.env` file and loads its values
so the Python script can use them via `os.environ`.
**Why it matters here:** My Gemini API key lives in `.env`, which is listed in `.gitignore` — meaning Git
will never track it or upload it to GitHub. This is a standard security practice, not specific to this
project.
**Read:** https://pypi.org/project/python-dotenv/

### pip and requirements.txt
**What it is:** `pip` installs Python packages. `requirements.txt` is a plain list of packages (and version
constraints) so anyone cloning the repo can run `pip install -r requirements.txt` and get the exact same
dependencies I'm using.
**Read:** https://pip.pypa.io/en/stable/user_guide/

---

## 2. What is an LLM API call, actually

**What it is:** When code "asks an LLM a question," it's sending an HTTP request to a provider's server
(Google, Anthropic, OpenAI, etc.) with: an API key (to identify/bill your account), a model name, and the
text you want it to respond to. The server runs the model and sends back generated text as a response.
**Why it matters here:** `src/ask.py` does exactly this — it builds a text prompt (context + question) and
sends it to Gemini via the `google-genai` Python library. Understanding this demystifies the "magic": it's a
network request/response, same shape as calling any other web API.
**Read (Gemini API quickstart):** https://ai.google.dev/gemini-api/docs/quickstart

### System prompts vs user prompts
**What it is:** Most LLM APIs let you separate "instructions for how to behave" (system prompt) from "the
actual question" (user prompt). In our code, the system prompt tells the model to only answer from the given
context and admit when it doesn't know — this is the first, crude version of a hallucination guardrail.
**Read:** https://ai.google.dev/gemini-api/docs/text-generation

---

## 3. Embeddings

**What it is:** A way of converting text into a list of numbers (a vector) such that texts with similar
*meaning* end up with similar vectors — even if they don't share the same words. This is what lets us search
by meaning instead of exact keyword matching.
**Why it matters here:** `src/ingest.py` and `src/ask.py` both use `sentence-transformers` to turn text
chunks and questions into vectors. The whole retrieval step depends on this.
**Read (conceptual intro):** https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
**Read (deeper, optional):** https://platform.openai.com/docs/guides/embeddings (OpenAI's explanation is
provider-agnostic and one of the clearest available, even though we're using Gemini/local embeddings)

---

## 4. Vector similarity (cosine similarity vs dot product)

**What it is:** Once text is turned into vectors, "how similar are these two pieces of text" becomes a math
question: how close are their vectors? Cosine similarity measures the *angle* between two vectors (ignoring
magnitude); dot product considers magnitude too. Different vector databases default to different metrics.
**Why it matters here:** This is literally how retrieval decides which chunks are "relevant" to your
question. It's also a favorite interview question for RAG systems — know the difference and why it matters
for normalized vs unnormalized embeddings.
**Read:** https://www.pinecone.io/learn/vector-similarity/ (good practical explanation, vendor-neutral enough)

---

## 5. Vector databases (ChromaDB)

**What it is:** A database purpose-built for storing vectors and doing fast "find the most similar vectors to
this one" queries at scale — something a normal SQL database isn't optimized for.
**Why it matters here:** We store every chunk's embedding in ChromaDB (`src/ingest.py`), and `src/ask.py`
queries it to retrieve the top-k most relevant chunks for a question.
**Read (official docs):** https://docs.trychroma.com/docs/overview/introduction

---

## 6. Retrieval-Augmented Generation (RAG) — the full picture

**What it is:** Instead of relying purely on what the LLM "remembers" from training, RAG retrieves relevant
real documents first, then feeds them to the LLM as context, so answers are grounded in your actual data.
**Why it matters here:** This is the entire project. Everything in Phases 1–9 of `ROADMAP.md` is refining one
part of this loop: ingest → chunk → embed → store → retrieve → generate → (eventually) evaluate.
**Read (good overview, provider-neutral):** https://www.promptingguide.ai/techniques/rag

---

## 7. Chunking

**What it is:** Splitting long documents into smaller pieces before embedding them, because (a) embedding
models have input size limits and (b) smaller, focused chunks retrieve more precisely than huge ones.
**Why it matters here:** Our current `chunk_text()` function in `src/ingest.py` is deliberately crude —
fixed-size character chunks with overlap. Phase 2 replaces this with structure-aware chunking. Understanding
*why* naive chunking is a problem (it can cut a sentence or table row in half) motivates that next phase.
**Read:** https://www.pinecone.io/learn/chunking-strategies/

---

## How to use this guide

I don't need to read all of this before touching any code — Phase 1 is already built and mostly working. The
plan is: read section 1–2 now (environment + API basics, needed to actually run what exists), then read one
new section per phase, right before we build that phase, so the concept is fresh when I'm looking at the
code that implements it.
