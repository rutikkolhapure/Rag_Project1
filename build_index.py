"""Read the HR policy PDFs in data/ and build the search index.

    py build_index.py
    py build_index.py --data-dir "D:\\HR\\policies"

Run it again whenever you add, change or remove a PDF. (ask.py also does this
automatically when it notices the PDFs changed.) The keyword index needs no key. If VECTOR_STORE=chroma
is set in .env, the passages are also embedded and stored in Chroma (needs the EMBED_ and CHROMA_ settings).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hr_rag.config import DATA_DIR, INDEX_DIR, VECTOR_STORE, vector_store_problem
from hr_rag.http_json import ServiceError
from hr_rag.index import build_index, load_chunks
from hr_rag.vectorstore import ChromaStore, sync_vector_store


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Build the search index from HR policy PDFs.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help=f"folder with PDFs (default: {DATA_DIR})")
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR, help=f"where to write the index (default: {INDEX_DIR})")
    args = parser.parse_args()

    print(f"Reading PDFs from {args.data_dir} ...")
    try:
        manifest = build_index(args.data_dir, args.index_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"\nError: {exc}")
        return 1

    skipped = manifest["skipped"]
    print(f"\nDone: {manifest['chunk_count']} passages from "
          f"{len(manifest['files']) - len(skipped)} PDF(s) -> {args.index_dir}")
    if skipped:
        print(f"{len(skipped)} file(s) were skipped (see the SKIPPED lines above).")

    if VECTOR_STORE:
        print()
        problem = vector_store_problem()
        if problem:
            print(f"Error: the keyword index is ready, but Chroma is not set up: {problem}")
            return 1
        try:
            sync_vector_store(args.index_dir, load_chunks(args.index_dir), ChromaStore.connect(), force=True)
        except ServiceError as exc:
            print(f"Error: the keyword index is ready, but filling Chroma failed: {exc}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
