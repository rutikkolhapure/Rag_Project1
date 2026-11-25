"""Everything the web app needs to start: make sure the index is current, pick the search, check the LLM.

(ask.py does the same for the command line, with prints. This returns the results as data instead.)
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .index import MANIFEST_FILE, build_index, index_status, load_retriever
from .ingest import Chunk
from .rag import HRAssistant
from .vectorstore import HybridRetriever, prepare_search


class SetupError(RuntimeError):
    """The assistant cannot start (for example there are no PDFs yet); the text says what to do."""


@dataclass
class Backend:
    assistant: HRAssistant
    chunks: list[Chunk]
    llm_problem: str | None                       # None when the LLM is ready to write answers
    hybrid: bool                                  # True when Chroma meaning-based search is active
    notes: list[str] = field(default_factory=list)     # setup messages worth showing to the user
    skipped: dict[str, str] = field(default_factory=dict)   # PDFs that could not be read, with the reason

    @property
    def documents(self) -> list[tuple[str, int]]:
        """(file name, number of passages) for every indexed PDF."""
        return sorted(Counter(chunk.doc for chunk in self.chunks).items())


def _skipped(index_dir: Path) -> dict[str, str]:
    try:
        return dict(json.loads((index_dir / MANIFEST_FILE).read_text(encoding="utf-8")).get("skipped", {}))
    except (OSError, ValueError):
        return {}


def start(data_dir: Path | None = None, index_dir: Path | None = None) -> Backend:
    """Bring the index up to date (building it when PDFs were added, changed or removed) and load it.

    Raises SetupError when there is nothing to search. Chroma or LLM problems never raise: they are
    reported in `notes` / `llm_problem`, and the assistant falls back to what does work.
    """
    data_dir = Path(data_dir or config.DATA_DIR)
    index_dir = Path(index_dir or config.INDEX_DIR)

    if index_status(data_dir, index_dir)[0] != "ok":
        try:
            build_index(data_dir, index_dir, log=lambda _: None)
        except (FileNotFoundError, RuntimeError) as exc:
            raise SetupError(str(exc)) from exc

    log: list[str] = []
    retriever = load_retriever(index_dir)
    search = prepare_search(index_dir, retriever, log=lambda message: log.append(message.strip()))
    hybrid = isinstance(search, HybridRetriever)
    # prepare_search also logs progress ("embedded 3/9"). Only its last line matters, and only when Chroma was
    # wanted but could not be used: that line is the reason.
    notes = log[-1:] if config.VECTOR_STORE and not hybrid else []
    problem = config.llm_setup_problem()
    return Backend(assistant=HRAssistant(search, use_llm=problem is None), chunks=retriever.chunks,
                   llm_problem=problem, hybrid=hybrid, notes=notes, skipped=_skipped(index_dir))
