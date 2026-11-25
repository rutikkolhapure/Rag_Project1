"""Keyword retrieval tuned for policy documents: BM25 + stemming + HR synonyms + spelling repair.

Why not neural embeddings? They need PyTorch/ONNX wheels, which don't exist yet for the
Python version this project was built on, plus a model download. This retriever needs
neither, starts in about a second and works fully offline:

  * stemming            "employees" / "employee", "days" / "day"
  * synonym expansion   "days off" also searches for "leave", "vacation", "holiday" ...
  * spelling repair     "harrasment" is matched to "harassment" if the documents contain it
  * BM25 ranking        exact policy terms and numbers ("14 days", "probation") score highest

Claude then does the semantic reading of the top passages, so the retriever only has to get
the right passages into the top few.
If you later run on a Python with sentence-transformers available, a dense-vector ranking can
be added in Retriever.search() without touching anything else.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import get_close_matches
from functools import lru_cache

import numpy as np
import snowballstemmer
from rank_bm25 import BM25Plus

from .ingest import Chunk

# Deliberately small. (scikit-learn's list drops words that matter in HR text,
# such as "full" and "part" from "full-time" / "part-time", so it is not used.)
STOPWORDS = frozenset("""
a about all also an and any are as at be been but by can could did do does for from get got had
has have how i if in into is it its many me much my of on or our should so than that the their
them then there these they this to up was we were what when where which who will with would you your
""".split())

# If any phrase in a group appears in the question, the group's other phrases are added
# (at reduced weight) to the search. Edit freely to match your company's vocabulary.
SYNONYM_GROUPS: list[tuple[str, ...]] = [
    ("leave", "vacation", "holiday", "pto", "absence", "time off", "days off", "day off"),
    ("sick", "illness", "unwell", "medical leave", "sick leave"),
    ("salary", "pay", "wage", "compensation", "payroll", "ctc", "remuneration", "stipend"),
    ("remote", "wfh", "work from home", "telework", "hybrid"),
    ("resign", "resignation", "quit", "notice period", "termination", "offboarding", "separation"),
    ("maternity", "paternity", "parental", "childbirth", "adoption", "newborn"),
    ("bonus", "incentive", "variable pay"),
    ("overtime", "extra hours", "working hours", "shift", "timings"),
    ("harassment", "discrimination", "bullying", "grievance", "complaint", "misconduct", "posh"),
    ("insurance", "mediclaim", "health cover", "benefit", "benefits"),
    ("pf", "provident fund", "epf", "pension", "gratuity", "retirement"),
    ("reimbursement", "reimburse", "expense", "expenses", "claim", "allowance"),
    ("probation", "probationary", "confirmation"),
    ("appraisal", "performance review", "rating", "increment", "promotion", "hike"),
    ("dress code", "attire", "uniform", "clothing"),
    ("attendance", "lateness", "punctuality", "late coming"),
]

EXPANSION_WEIGHT = 0.35   # how much synonym matches count relative to the user's own words
SPELLING_CUTOFF = 0.8     # 0..1 similarity needed to treat an unknown word as a typo of a known one
MIN_SPELLING_LENGTH = 5   # shorter words are never "corrected"

_WORD = re.compile(r"[a-z0-9]+")
_stemmer = snowballstemmer.stemmer("english")


@lru_cache(maxsize=200_000)
def _stem(word: str) -> str:
    return _stemmer.stemWord(word)


def content_words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in content_words(text)]


def _prepare_groups() -> list[list[tuple[str, ...]]]:
    groups = []
    for group in SYNONYM_GROUPS:
        phrases = [tuple(tokenize(p)) for p in group]
        groups.append([p for p in phrases if p])
    return groups


_GROUPS = _prepare_groups()


def _contains(tokens: list[str], phrase: tuple[str, ...]) -> bool:
    n = len(phrase)
    return any(tuple(tokens[i:i + n]) == phrase for i in range(len(tokens) - n + 1))


def expand_query(tokens: list[str]) -> list[str]:
    """Synonym tokens to add for the given (stemmed) query tokens."""
    extra: list[str] = []
    for group in _GROUPS:
        if any(_contains(tokens, phrase) for phrase in group):
            for phrase in group:
                for tok in phrase:
                    if tok not in tokens and tok not in extra:
                        extra.append(tok)
    return extra


@dataclass
class Hit:
    chunk: Chunk
    score: float   # BM25 score incl. synonym matches; only meaningful for comparing hits to each other


def _searchable(chunk: Chunk) -> str:
    """Text that is indexed: the file title and section heading give every chunk its context."""
    return f"{chunk.title}. {chunk.section}. {chunk.text}"


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("cannot build a retriever from zero chunks")
        self.chunks = chunks
        words_per_chunk = [content_words(_searchable(c)) for c in chunks]
        # BM25Plus with delta=0 is plain BM25 whose word weights never go negative. (BM25Okapi
        # gives words found in over half the chunks a negative weight, so on a small index,
        # such as a single short PDF, every score collapses and nothing is ever found.)
        self._bm25 = BM25Plus([[_stem(w) for w in words] for words in words_per_chunk], delta=0)
        self._known_stems = {_stem(w) for words in words_per_chunk for w in words}
        self._vocabulary = sorted({w for words in words_per_chunk for w in words if w.isalpha()})

    def _repair_spelling(self, words: list[str]) -> list[str]:
        """Swap words that occur nowhere in the documents for their closest known spelling."""
        repaired = []
        for word in words:
            if _stem(word) in self._known_stems or len(word) < MIN_SPELLING_LENGTH or not word.isalpha():
                repaired.append(word)
                continue
            close = get_close_matches(word, self._vocabulary, n=1, cutoff=SPELLING_CUTOFF)
            repaired.append(close[0] if close else word)
        return repaired

    def search(self, query: str, top_k: int = 6) -> list[Hit]:
        """Best chunks for `query`, most relevant first. Empty list when nothing matches."""
        tokens = [_stem(w) for w in self._repair_spelling(content_words(query))]
        if not tokens:
            return []

        scores = np.asarray(self._bm25.get_scores(tokens), dtype=float)
        extra = expand_query(tokens)
        if extra:
            scores = scores + EXPANSION_WEIGHT * np.asarray(self._bm25.get_scores(extra), dtype=float)

        # Only chunks sharing at least one word with the question may be returned; this is
        # what lets an off-topic question come back empty instead of "the least bad" passages.
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]
