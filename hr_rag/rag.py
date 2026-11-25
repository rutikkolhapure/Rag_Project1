"""Ties retrieval and generation together: question in, cited answer out."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .config import TOP_K
from .llm import cited_numbers, generate_answer
from .retriever import Hit, Retriever

NOT_FOUND = ("I couldn't find anything about that in the indexed HR documents. "
             "Try rephrasing the question, or check with HR.")

# A question this short after an earlier one is probably a follow-up ("what about sick leave?"),
# so the previous question is added to the search to give it context.
FOLLOW_UP_MAX_WORDS = 6


@dataclass
class Source:
    number: int   # the [n] used in the answer
    doc: str
    pages: str
    section: str

    def label(self) -> str:
        text = f"{self.doc}, {self.pages}"
        return f"{text} - {self.section}" if self.section else text


@dataclass
class Answer:
    text: str            # the LLM's answer; empty in retrieval-only mode
    hits: list[Hit]      # everything retrieved, in the order Claude saw it
    sources: list[Source]
    used_llm: bool
    notice: str = ""     # something worth telling the user, e.g. that vector search was unavailable


def _source(number: int, hit: Hit) -> Source:
    c = hit.chunk
    pages = f"p.{c.page_start}" if c.page_start == c.page_end else f"pp.{c.page_start}-{c.page_end}"
    return Source(number, c.doc, pages, c.section)


class HRAssistant:
    def __init__(self, retriever: Retriever, top_k: int = TOP_K, use_llm: bool = True):
        self.retriever = retriever
        self.top_k = top_k
        self.use_llm = use_llm

    def search_query(self, question: str, history: Sequence[tuple[str, str]]) -> str:
        if history and len(question.split()) <= FOLLOW_UP_MAX_WORDS:
            return f"{history[-1][0]} {question}"
        return question

    def ask(self, question: str, history: Sequence[tuple[str, str]] = ()) -> Answer:
        """Raises hr_rag.llm.LLMError if the LLM can't be reached (only when use_llm is on)."""
        hits = self.retriever.search(self.search_query(question, history), self.top_k)
        notice = getattr(self.retriever, "last_warning", "")   # a hybrid retriever explains fallbacks here
        if not hits:
            return Answer(NOT_FOUND, [], [], used_llm=False, notice=notice)
        if not self.use_llm:
            return Answer("", hits, [_source(n, h) for n, h in enumerate(hits, 1)], used_llm=False, notice=notice)

        text = generate_answer(question, hits, history)
        cited = [n for n in cited_numbers(text) if 1 <= n <= len(hits)]
        shown = cited or list(range(1, len(hits) + 1))   # no citations -> list everything consulted
        return Answer(text, hits, [_source(n, hits[n - 1]) for n in shown], used_llm=True, notice=notice)
