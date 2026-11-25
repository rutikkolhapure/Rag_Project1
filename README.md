# HR Policy RAG

Ask questions in plain English about your HR policy PDFs and get answers that cite the document,
page and section they came from.

```
PDFs in data/  ->  build_index.py  ->  index/          (no key needed)
your question  ->  retrieve best passages  ->  your LLM writes a cited answer   (LLM settings + key needed)
```

## Quick start (PowerShell, from this folder)

```powershell
# 1. Put your HR policy PDFs in the data\ folder (sub-folders are fine)

# 2. Ask questions (the index is built automatically the first time and whenever a PDF changes)
.venv\Scripts\python ask.py
.venv\Scripts\python ask.py "How many casual leave days do I get?"
```

The `.venv` folder already has everything installed. To set it up on another machine:
`py -m venv .venv` then `.venv\Scripts\python -m pip install -r requirements.txt`
(use `requirements-dev.txt` instead if you also want to run the tests).

Until your LLM is set up in `.env`, `ask.py` runs in **retrieval-only mode**: it prints the matching
passages with their source, so you can check what the search finds before connecting the LLM.

## Where to put your LLM details

You enter them in **one place only**, the [`.env`](.env) file (already created, all values blank):

| Line in `.env` | What to enter | Example |
|----------------|---------------|---------|
| `LLM_PROVIDER` | `openai` for any OpenAI-compatible service or local server (already set). `anthropic` for Claude. | `openai` |
| `LLM_BASE_URL` | The API address of your LLM service | `https://api.openai.com/v1` |
| `LLM_MODEL` | The model name your service expects | `gpt-4o-mini` |
| `LLM_API_KEY` | Your key. Leave blank for a local server (Ollama, LM Studio). | *(your key)* |

No quotes, no spaces around `=`. `LLM_BASE_URL` has no default on purpose, so your key can never be sent
to a service you did not choose. Ready-made addresses for OpenAI, Gemini, Mistral, Groq, DeepSeek,
OpenRouter, Ollama and LM Studio are listed in the comments inside `.env`.

The `openai` provider works with anything that offers the OpenAI "chat completions" API
(`POST {LLM_BASE_URL}/chat/completions`). It uses only Python's standard library, so there is nothing
extra to install. It does **not** cover services with their own format (Azure OpenAI deployment URLs,
AWS Bedrock, Gemini's native API); use their OpenAI-compatible endpoint if they have one.

To use Claude instead: set `LLM_PROVIDER=anthropic`, put your key on the `ANTHROPIC_API_KEY=` line,
optionally set `LLM_MODEL` (default `claude-opus-5-5`), and run
`.venv\Scripts\python -m pip install anthropic`.

Two places in the code read these settings. You do not edit them, they are listed so you know where the
key is used:

| File | Where | What happens there |
|------|-------|--------------------|
| [`hr_rag/config.py`](hr_rag/config.py) | `load_dotenv(...)`, `llm_api_key()` | Loads `.env` into the environment and reads the key. |
| [`hr_rag/llm.py`](hr_rag/llm.py) | `_generate_openai()` (or `_client()` for Claude) | Sends the key to your LLM service, and nowhere else. |

The LLM key is never needed for `build_index.py`.

## Connecting Chroma (optional)

Chroma adds **meaning-based search** next to the keyword search, so "can I telecommute?" finds a passage
that says "work from home". It is off until you set `VECTOR_STORE=chroma` in [`.env`](.env). Everything it
needs is in the "OPTIONAL: Chroma" section at the bottom of `.env` (all values blank):

| Block in `.env` | What to enter |
|-----------------|---------------|
| **Where Chroma runs** | Self-hosted: `CHROMA_HOST` (and `CHROMA_PORT`, default 8000; `CHROMA_SSL=true` for https). Chroma Cloud: leave `CHROMA_HOST` blank and fill `CHROMA_TENANT`, `CHROMA_DATABASE`, `CHROMA_API_KEY`. |
| **Embeddings service** | `EMBED_BASE_URL`, `EMBED_MODEL`, `EMBED_API_KEY`: any OpenAI-compatible `/embeddings` service (OpenAI, Gemini, Mistral, local Ollama...). It can be a different service from your LLM, which is useful because some LLM services (Anthropic, Groq, DeepSeek) have no embeddings. |
| **Collection** | `CHROMA_COLLECTION` (default `hr_policies`). This project **deletes and rebuilds** that collection on every re-index, so never point it at one that holds other data. |

Then install the client once and build:

```powershell
.venv\Scripts\python -m pip install chromadb-client
.venv\Scripts\python build_index.py
```

- **Why `chromadb-client` and not `chromadb`?** The full `chromadb` package cannot be installed on Python 3.15
  (it requires `onnxruntime`, which has no 3.15 build). The thin client talks to a **running Chroma server**
  over HTTP, so you need one: `chroma run --path <folder>` from a Python 3.12/3.13 install, Docker
  (`chromadb/chroma`), or Chroma Cloud. It cannot open a Chroma folder on disk by itself.
- **What happens:** `build_index.py` (and `ask.py`, automatically, when the PDFs or the settings changed) embeds
  every passage and stores it in the collection. Each question is embedded too; Chroma returns the nearest
  passages and they are merged with the keyword results (reciprocal rank fusion), so a passage both
  rankings like comes first. `--keyword-only` on `ask.py` skips Chroma.
- **If Chroma or the embeddings service is down,** `ask.py` says so and carries on with keyword search.
- **Off-topic questions** still return "not found": vector matches farther than `CHROMA_MAX_DISTANCE` (cosine
  distance, default 0.8) are ignored. Embedding models differ, so if relevant passages go missing raise it
  (for example 1.0); if unrelated passages show up lower it. `0` turns the cut-off off.
- **Changing `EMBED_MODEL`** (or the embeddings address) refills Chroma automatically, because vectors from
  different models cannot be compared.
- **Privacy:** passage text is sent to the embeddings service and stored in Chroma. Telemetry is switched off.

| File | Where | What happens there |
|------|-------|--------------------|
| [`hr_rag/embeddings.py`](hr_rag/embeddings.py) | `_embed_batch()` | Sends `EMBED_API_KEY` to the embeddings service, and nowhere else. |
| [`hr_rag/vectorstore.py`](hr_rag/vectorstore.py) | `_make_client()` | Sends `CHROMA_API_KEY` (if you set one) to your Chroma server. |

## How it works

1. **Ingest** ([`hr_rag/ingest.py`](hr_rag/ingest.py)): `pypdf` extracts the text; running headers,
   footers and page numbers are stripped; the text is split into ~250-word chunks that never straddle
   a section heading. Each chunk remembers its file, page range and section heading.
2. **Retrieve** ([`hr_rag/retriever.py`](hr_rag/retriever.py)): BM25 keyword ranking over stemmed
   words. A small HR synonym table lets "days off" find "leave", and words that appear nowhere in
   your documents are matched to their closest spelling ("harrasment" finds "harassment").
   Questions that match nothing return nothing instead of random text. With Chroma connected (see above),
   [`hr_rag/vectorstore.py`](hr_rag/vectorstore.py) adds meaning-based search and merges both rankings.
3. **Generate** ([`hr_rag/llm.py`](hr_rag/llm.py)): the top passages go to your LLM with strict
   instructions: answer only from the excerpts, cite them as `[1]`, flag conflicting versions (for
   example a 2025 and a 2026 policy), and say so when the documents don't cover the question.
   Smaller or local models follow these rules less reliably than large ones, so check their citations.
4. **Chat memory**: the last 3 exchanges are remembered, and short follow-ups such as "what about
   sick leave?" borrow the previous question for searching. `/clear` forgets them.

## Files

```
data/               your PDFs go here
index/              generated by build_index.py (safe to delete)
.env                your LLM settings and key (git-ignored)
ask.py              ask questions (chat or one-shot)
build_index.py      rebuild the index by hand (ask.py also does it when PDFs change)
hr_rag/             config.py, ingest.py, retriever.py, index.py, llm.py, rag.py,
                    embeddings.py, vectorstore.py (Chroma), http_json.py (shared HTTP helper)
tests/              run:  .venv\Scripts\python -m unittest discover -s tests -v
requirements.txt    libraries needed to run   (requirements-dev.txt adds what the tests need)
```

Tweak chunk size, number of passages (`TOP_K`) and more in [`hr_rag/config.py`](hr_rag/config.py);
edit the synonym table at the top of [`hr_rag/retriever.py`](hr_rag/retriever.py) to match your
company's vocabulary (for example add your own names for leave types).

## Things to know

- **Privacy:** when an LLM is on, the retrieved passages and your question are sent to the service at
  `LLM_BASE_URL` (nothing leaves your machine if that is a local server). Anyone who can run this can
  query every indexed PDF, so don't index documents (for example salary bands) that the people using
  it shouldn't see.
- **Scanned PDFs** (pictures of pages) have no text to read. They are skipped with a message and need
  OCR first. **Tables** may come out as run-together text because PDFs don't store table structure.
- **Keyword search is the default.** This was built on Python 3.15, for which PyTorch and ONNX have no
  installable packages, so there are no local embedding models. Keyword search works well for policy
  text, but a question that shares no words with the policy and isn't in the synonym table can be
  missed (the LLM only sees the passages the search returns). Connecting Chroma with an embeddings
  service (see above) closes that gap.
- **Answers are only as good as the documents.** For decisions that affect a person, check with HR.
