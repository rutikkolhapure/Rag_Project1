"""Turn text into embeddings (lists of numbers) with an OpenAI-compatible /embeddings service.

>>> This module sends your EMBED_API_KEY (from .env) to the service at EMBED_BASE_URL, and nowhere else. <<<
Works with anything that offers `POST {base}/embeddings` the OpenAI way: OpenAI, Gemini, Mistral, Together,
or a local Ollama / LM Studio / vLLM server. Standard library only.
"""
from __future__ import annotations

from typing import Callable, Sequence

from . import config
from .http_json import Service, ServiceError, error_text, post_json

_SERVICE = Service("embeddings service", "EMBED_API_KEY", "EMBED_BASE_URL", "EMBED_MODEL")


class EmbeddingError(ServiceError):
    """A problem getting embeddings, phrased so it can be shown to the user as is."""


def _embeddings_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    return url if url.endswith("/embeddings") else url + "/embeddings"


def _parse(data, expected: int) -> list[list[float]]:
    if isinstance(data, dict) and data.get("error") and not data.get("data"):
        raise EmbeddingError(f"The embeddings service returned an error: {error_text(data['error'])[:300]}")
    try:
        rows = sorted(enumerate(data["data"]), key=lambda pair: pair[1].get("index", pair[0]))
        vectors = [row["embedding"] for _, row in rows]
    except (KeyError, TypeError, AttributeError) as exc:
        raise EmbeddingError("The embeddings service sent a reply in an unexpected format. Check that "
                             "EMBED_BASE_URL points at an OpenAI-compatible API (usually ending in /v1).") from exc
    if len(vectors) != expected:
        raise EmbeddingError(f"The embeddings service returned {len(vectors)} embeddings for {expected} texts.")
    for vector in vectors:
        if (not isinstance(vector, list) or not vector
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in vector)):
            raise EmbeddingError("The embeddings service returned something that is not a list of numbers. "
                                 "Is EMBED_MODEL an embedding model (not a chat model)?")
    if len({len(v) for v in vectors}) != 1:
        raise EmbeddingError("The embeddings service returned vectors of different lengths.")
    return [[float(x) for x in v] for v in vectors]


def _embed_batch(batch: Sequence[str]) -> list[list[float]]:
    headers = {"Content-Type": "application/json", "User-Agent": "hr-rag/1.0"}
    # >>> EMBEDDINGS KEY: the key is USED here <<<
    # Read from the environment, which hr_rag/config.py fills from your .env file. Keep the key in .env;
    # do not write it here. Local servers (Ollama ...) need no key.
    key = config.embed_api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = {"model": config.EMBED_MODEL, "input": list(batch)}
    data = post_json(_embeddings_url(config.EMBED_BASE_URL), headers, body, _SERVICE,
                     config.EMBED_TIMEOUT_SECONDS, EmbeddingError)
    return _parse(data, len(batch))


def embed_texts(texts: Sequence[str], progress: Callable[[int, int], None] | None = None) -> list[list[float]]:
    """One embedding per text, in order. Raises EmbeddingError on failure."""
    texts = [t if t.strip() else "." for t in texts]   # services reject empty strings
    vectors: list[list[float]] = []
    size = max(1, config.EMBED_BATCH_SIZE)
    for start in range(0, len(texts), size):
        vectors.extend(_embed_batch(texts[start:start + size]))
        if progress:
            progress(min(start + size, len(texts)), len(texts))
    if len({len(v) for v in vectors}) > 1:
        raise EmbeddingError("The embeddings service returned vectors of different lengths.")
    return vectors
