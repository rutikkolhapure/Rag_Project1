"""Chroma vector database: store passage embeddings, search them, and merge with keyword search.

>>> CHROMA_API_KEY (only if you use one) is read in config.py and sent only to your Chroma server. <<<

How the pieces fit:
    index build / start-up   passages -> embeddings service -> Chroma collection        (sync_vector_store)
    a question               question -> embedding -> Chroma nearest passages  \\
                             question -> BM25 keyword search                    -> merged ranking (HybridRetriever)
If Chroma or the embeddings service is down, the keyword search still answers (with a visible note).

Uses the lightweight `chromadb-client` package (it talks to a running Chroma server over HTTP).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence

from . import config
from .embeddings import embed_texts
from .http_json import ServiceError
from .index import MANIFEST_FILE, write_json_atomic
from .ingest import Chunk
from .retriever import Hit, Retriever

EMBED_FORMAT = 1       # bump if embedding_text() changes, so Chroma gets refilled
RRF_K = 60             # reciprocal-rank-fusion constant (standard value)
CANDIDATE_FACTOR = 3   # each ranking contributes top_k * this many candidates to the merge
ADD_BATCH = 100        # passages per add() call to Chroma


class VectorStoreError(ServiceError):
    """A problem talking to Chroma, phrased so it can be shown to the user as is."""


# ---- talking to Chroma ------------------------------------------------------------

def _target() -> str:
    if config.chroma_is_cloud():
        return f"Chroma Cloud ({config.CHROMA_TENANT}/{config.CHROMA_DATABASE})"
    return f"{config.CHROMA_HOST}:{config.CHROMA_PORT}"


def _is_not_found(exc: Exception) -> bool:
    text = str(exc).lower()
    return type(exc).__name__ == "NotFoundError" or ("collection" in text and
                                                     ("does not exist" in text or "not found" in text))


def _translate(exc: Exception) -> VectorStoreError:
    """Turn whatever the Chroma client raised into one readable message."""
    if isinstance(exc, VectorStoreError):
        return exc
    name, text = type(exc).__name__, str(exc).strip()
    lower = text.lower()
    if "connect" in lower or isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return VectorStoreError(f"Could not connect to Chroma at {_target()}. Check the CHROMA_ settings in your "
                                f".env file and that the server is running. ({text[:150]})")
    if name in {"AuthorizationError", "ChromaAuthError"} or any(w in lower for w in ("unauthorized", "forbidden")):
        return VectorStoreError(f"Chroma rejected the credentials. Check CHROMA_API_KEY in your .env file. "
                                f"({text[:150]})")
    return VectorStoreError(f"Chroma error ({name}): {text[:300]}")


def _make_client():
    try:
        import chromadb
    except ImportError as exc:
        raise VectorStoreError("VECTOR_STORE=chroma needs the 'chromadb-client' package. Install it with:  "
                               ".venv\\Scripts\\python -m pip install chromadb-client") from exc
    # Telemetry is switched off: this project handles HR documents.
    settings = chromadb.Settings(anonymized_telemetry=False)
    # >>> CHROMA KEY: the key (if any) is USED here <<<
    # Read from the environment, which hr_rag/config.py fills from your .env file. Do not write it here.
    key = config.chroma_api_key()
    try:
        if config.chroma_is_cloud():
            return chromadb.CloudClient(tenant=config.CHROMA_TENANT, database=config.CHROMA_DATABASE,
                                        api_key=key, settings=settings)
        headers = {"Authorization": f"Bearer {key}"} if key else None
        return chromadb.HttpClient(host=config.CHROMA_HOST, port=int(config.CHROMA_PORT), ssl=config.CHROMA_SSL,
                                   headers=headers, settings=settings)
    except Exception as exc:   # the client raises plain ValueError for "cannot connect"
        raise _translate(exc) from exc


def embedding_text(chunk: Chunk) -> str:
    """What gets embedded: the passage with its section heading, which helps short questions match."""
    return f"{chunk.section}\n{chunk.text}" if chunk.section else chunk.text


def _metadata(chunk: Chunk) -> dict:
    return {"doc": chunk.doc, "title": chunk.title, "section": chunk.section,
            "page_start": chunk.page_start, "page_end": chunk.page_end}


def _chunks_from(result) -> list[tuple[Chunk, float]]:
    """Chroma query results are nested per query; we always send one query, so take element 0."""
    def first(key: str) -> list:
        value = result.get(key)
        return value[0] if value else []

    ids, documents, metadatas, distances = first("ids"), first("documents"), first("metadatas"), first("distances")
    found = []
    for i, chunk_id in enumerate(ids):
        try:
            meta, text = metadatas[i], documents[i]
            chunk = Chunk(id=chunk_id, doc=meta["doc"], title=meta["title"], section=meta["section"],
                          page_start=int(meta["page_start"]), page_end=int(meta["page_end"]), text=text)
            found.append((chunk, float(distances[i])))
        except (IndexError, KeyError, TypeError, ValueError):
            continue   # a record without our metadata was not written by this project
    return found


class ChromaStore:
    """The one Chroma collection this project owns."""

    def __init__(self, client, name: str):
        self.client = client
        self.name = name

    @classmethod
    def connect(cls) -> "ChromaStore":
        return cls(_make_client(), config.CHROMA_COLLECTION)

    def _collection(self):
        # embedding_function=None: we always pass our own embeddings (the client's built-in default
        # embedder needs onnxruntime, which has no build for Python 3.15).
        return self.client.get_collection(name=self.name, embedding_function=None)

    def count(self) -> int | None:
        """Number of passages stored, or None when the collection does not exist yet."""
        try:
            return self._collection().count()
        except Exception as exc:
            if _is_not_found(exc):
                return None
            raise _translate(exc) from exc

    def replace_all(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        """Delete the collection (if it exists) and fill a fresh one. Cosine distance (0 = identical)."""
        try:
            try:
                self.client.delete_collection(self.name)
            except Exception as exc:
                if not _is_not_found(exc):
                    raise
            collection = self.client.create_collection(
                name=self.name, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None,
                metadata={"description": "HR policy passages written by the hr-rag project"})
            for start in range(0, len(chunks), ADD_BATCH):
                part = chunks[start:start + ADD_BATCH]
                collection.add(ids=[c.id for c in part], embeddings=[list(v) for v in vectors[start:start + ADD_BATCH]],
                               documents=[c.text for c in part], metadatas=[_metadata(c) for c in part])
        except Exception as exc:
            raise _translate(exc) from exc

    def search(self, vector: Sequence[float], n_results: int) -> list[tuple[Chunk, float]]:
        """The `n_results` nearest passages as (chunk, cosine distance), nearest first."""
        try:
            result = self._collection().query(query_embeddings=[list(vector)], n_results=n_results,
                                              include=["documents", "metadatas", "distances"])
        except Exception as exc:
            if _is_not_found(exc):
                raise VectorStoreError(f"The Chroma collection '{self.name}' does not exist yet. "
                                       "Run  py build_index.py  to fill it.") from exc
            raise _translate(exc) from exc
        return _chunks_from(result)


# ---- keeping Chroma in step with the index ----------------------------------------

def desired_state(chunks: Sequence[Chunk]) -> dict:
    """What Chroma should contain. Saved in the manifest after a sync, to spot when it is out of date."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(f"{chunk.id}\0{chunk.text}\n".encode("utf-8"))
    return {"type": "chroma", "target": _target(), "collection": config.CHROMA_COLLECTION,
            "embedding_service": config.EMBED_BASE_URL, "embedding_model": config.EMBED_MODEL,
            "embed_format": EMBED_FORMAT, "chunks": len(chunks), "chunks_sha256": digest.hexdigest()}


def sync_vector_store(index_dir: Path, chunks: Sequence[Chunk], store: ChromaStore,
                      log: Callable[[str], None] = print, force: bool = False) -> bool:
    """Fill Chroma from `chunks` unless it already matches. Returns True when it was (re)filled.

    The whole collection is rebuilt (policy sets are small); nothing is merged. Raises ServiceError.
    """
    state = desired_state(chunks)
    manifest_path = index_dir / MANIFEST_FILE
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    if not force and manifest.get("vector_store") == state and store.count() == len(chunks):
        return False

    log(f"Filling Chroma collection '{store.name}' with {len(chunks)} passages "
        f"(embeddings by {config.EMBED_MODEL}) ...")
    vectors = embed_texts([embedding_text(c) for c in chunks],
                          progress=lambda done, total: log(f"  embedded {done}/{total}"))
    store.replace_all(chunks, vectors)
    manifest["vector_store"] = state
    write_json_atomic(manifest_path, manifest)
    log("  Chroma is up to date.")
    return True


# ---- searching ------------------------------------------------------------------

def fuse(lexical: Sequence[Hit], nearby: Sequence[tuple[Chunk, float]]) -> list[Hit]:
    """Merge two rankings with reciprocal rank fusion: passages ranked well by both rise to the top."""
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    first_seen: dict[str, int] = {}
    rankings = [[hit.chunk for hit in lexical], [chunk for chunk, _ in nearby]]
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (RRF_K + rank)
            chunks.setdefault(chunk.id, chunk)
            first_seen.setdefault(chunk.id, len(first_seen))
    ranked = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], first_seen[chunk_id]))
    return [Hit(chunks[chunk_id], scores[chunk_id]) for chunk_id in ranked]


class HybridRetriever:
    """Keyword (BM25) search plus Chroma meaning-based search, merged. Same interface as Retriever."""

    def __init__(self, keyword: Retriever, store: ChromaStore, max_distance: float | None):
        self.keyword = keyword
        self.chunks = keyword.chunks
        self.store = store
        self.max_distance = max_distance
        self.last_warning = ""   # set when the last search had to fall back to keywords only

    def search(self, query: str, top_k: int = config.TOP_K) -> list[Hit]:
        self.last_warning = ""
        wide = top_k * CANDIDATE_FACTOR
        lexical = self.keyword.search(query, wide)
        try:
            vector = embed_texts([query])[0]
            nearby = [(chunk, distance) for chunk, distance in self.store.search(vector, wide)
                      if self.max_distance is None or distance <= self.max_distance]
        except ServiceError as exc:
            self.last_warning = f"Meaning-based search is unavailable ({exc}) - used keyword search only."
            return lexical[:top_k]
        return fuse(lexical, nearby)[:top_k]


def prepare_search(index_dir: Path, retriever: Retriever, log: Callable[[str], None] = print,
                   force_sync: bool = False):
    """The keyword `retriever`, upgraded to hybrid search when Chroma is configured and reachable.

    Never raises for Chroma or embeddings problems: it explains them through `log` and carries on
    with keyword search, so a Chroma outage cannot take the whole assistant down.
    """
    if not config.VECTOR_STORE:
        return retriever
    problem = config.vector_store_problem()
    if problem:
        log(f"Chroma is not set up ({problem}), so using keyword search only.\n")
        return retriever
    try:
        store = ChromaStore.connect()
        sync_vector_store(index_dir, retriever.chunks, store, log, force=force_sync)
    except ServiceError as exc:
        log(f"Chroma is unavailable ({exc}), so using keyword search only.\n")
        return retriever
    return HybridRetriever(retriever, store, config.chroma_max_distance())
