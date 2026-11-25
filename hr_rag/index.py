"""Build, save, load and freshness-check the on-disk index.

Layout of the index folder:
    chunks.json    every chunk of every PDF (text, page numbers, section heading)
    manifest.json  which PDFs (by SHA-256) and which chunk settings produced chunks.json

The search structures themselves are rebuilt in memory on load (about a second for
thousands of chunks), so there is nothing fragile to unpickle.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .config import CHUNK_OVERLAP_WORDS, CHUNK_WORDS, MIN_CHUNK_WORDS
from .ingest import Chunk, ingest_pdf
from .retriever import Retriever

INDEX_VERSION = 2   # bump whenever the logic in ingest.py changes, so old indexes get rebuilt
CHUNKS_FILE = "chunks.json"
MANIFEST_FILE = "manifest.json"


def find_pdfs(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _doc_name(pdf: Path, data_dir: Path) -> str:
    return pdf.relative_to(data_dir).as_posix()


def _settings() -> dict:
    return {"chunk_words": CHUNK_WORDS, "overlap_words": CHUNK_OVERLAP_WORDS,
            "min_chunk_words": MIN_CHUNK_WORDS}


def write_json_atomic(path: Path, payload: object) -> None:
    """Write via a temp file so a crash never leaves a half-written index behind."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def build_index(data_dir: Path, index_dir: Path, log: Callable[[str], None] = print) -> dict:
    """Read every PDF under data_dir and (re)write the index. Returns the manifest."""
    pdfs = find_pdfs(data_dir)
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {data_dir}. Copy your HR policy PDFs there.")

    chunks: list[Chunk] = []
    files: dict[str, str] = {}
    skipped: dict[str, str] = {}
    for pdf in pdfs:
        name = _doc_name(pdf, data_dir)
        files[name] = _sha256(pdf)
        try:
            report = ingest_pdf(pdf, name, CHUNK_WORDS, CHUNK_OVERLAP_WORDS, MIN_CHUNK_WORDS)
        except ValueError as exc:
            skipped[name] = str(exc)
            log(f"  SKIPPED  {name}: {exc}")
            continue
        chunks.extend(report.chunks)
        log(f"  {name}: {report.pages} page(s) -> {len(report.chunks)} chunk(s)")
        for warning in report.warnings:
            log(f"    warning: {warning}")

    if not chunks:
        raise RuntimeError("None of the PDFs contained readable text, so no index was written.")

    index_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": INDEX_VERSION,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "settings": _settings(),
        "files": files,
        "skipped": skipped,
        "chunk_count": len(chunks),
    }
    write_json_atomic(index_dir / CHUNKS_FILE, [asdict(c) for c in chunks])
    write_json_atomic(index_dir / MANIFEST_FILE, manifest)
    return manifest


def index_status(data_dir: Path, index_dir: Path) -> tuple[str, str]:
    """('ok' | 'missing' | 'stale', reason). Cheap enough to run on every start-up."""
    manifest_path = index_dir / MANIFEST_FILE
    if not manifest_path.exists() or not (index_dir / CHUNKS_FILE).exists():
        return "missing", "no index has been built yet"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "stale", "the index manifest is unreadable"
    if manifest.get("version") != INDEX_VERSION or manifest.get("settings") != _settings():
        return "stale", "chunking settings changed"

    current = {_doc_name(p, data_dir): _sha256(p) for p in find_pdfs(data_dir)}
    known = manifest.get("files", {})
    added = sorted(set(current) - set(known))
    removed = sorted(set(known) - set(current))
    changed = sorted(n for n in set(current) & set(known) if current[n] != known[n])
    if added or removed or changed:
        parts = [f"{label}: {', '.join(names)}" for label, names in
                 (("new", added), ("removed", removed), ("changed", changed)) if names]
        return "stale", "; ".join(parts)
    return "ok", ""


def load_chunks(index_dir: Path) -> list[Chunk]:
    raw = json.loads((index_dir / CHUNKS_FILE).read_text(encoding="utf-8"))
    return [Chunk(**item) for item in raw]


def load_retriever(index_dir: Path) -> Retriever:
    return Retriever(load_chunks(index_dir))
