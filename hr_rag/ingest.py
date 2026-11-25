"""PDF -> cleaned lines -> heading-aware chunks.

Chunks remember the document, the page range and the section heading(s) they cover, so that
answers can cite "Leave_Policy.pdf, p.3, Casual Leave" and so that retrieval can use the
heading text (which is often the only place a policy name appears).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Chunk:
    id: str          # "<file name>#<n>"
    doc: str         # file name, e.g. "Leave_Policy_2026.pdf"
    title: str       # readable title derived from the file name
    section: str     # heading(s) the chunk covers, e.g. "Sick Leave / Maternity Leave" ("" if none)
    page_start: int  # 1-based
    page_end: int
    text: str


@dataclass
class PdfReport:
    pages: int
    chunks: list[Chunk]
    warnings: list[str]


# --------------------------------------------------------------------------- reading
def read_pdf(path: Path) -> tuple[list[str], list[str]]:
    """Return (text of each page, warnings). Raises ValueError if the file can't be opened."""
    warnings: list[str] = []
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF is password-protected")
        pages: list[str] = []
        for number, page in enumerate(reader.pages, start=1):
            try:
                pages.append(page.extract_text() or "")
            except Exception as exc:  # one bad page should not lose the whole document
                pages.append("")
                warnings.append(f"page {number}: text extraction failed ({type(exc).__name__})")
        return pages, warnings
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"could not read PDF ({type(exc).__name__}: {exc})") from exc


# --------------------------------------------------------------------------- cleaning
_PAGE_NUMBER = re.compile(r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$", re.IGNORECASE)


def _fingerprint(line: str) -> str:
    """Make 'Page 3 of 10' and 'Page 4 of 10' compare equal when spotting running headers."""
    return re.sub(r"\d+", "#", line.strip().lower())


def _edge_positions(count: int) -> set[int]:
    """Indexes of the first two and last two lines of a page, where headers and footers live."""
    return set(range(min(2, count))) | set(range(max(count - 2, 0), count))


def clean_pages(pages: list[str]) -> list[list[str]]:
    """Split pages into lines; drop blank lines, bare page numbers and running headers/footers.

    A running header/footer is a line at the top or bottom of a page that repeats (ignoring
    digits) on at least 60% of the pages. Repeated lines in the middle of a page are kept.
    """
    split = [[ln.strip() for ln in text.replace("\r", "").split("\n") if ln.strip()] for text in pages]

    repeated: set[str] = set()
    if len(split) >= 3:  # a header can only be recognised if it repeats
        counts = Counter(
            fp for lines in split for fp in {_fingerprint(lines[i]) for i in _edge_positions(len(lines))}
        )
        repeated = {fp for fp, n in counts.items() if n / len(split) >= 0.6}

    cleaned = []
    for lines in split:
        edges = _edge_positions(len(lines))
        cleaned.append([
            ln for i, ln in enumerate(lines)
            if not _PAGE_NUMBER.match(ln) and not (i in edges and _fingerprint(ln) in repeated)
        ])
    return cleaned


# --------------------------------------------------------------------------- headings
_NUMBERED = re.compile(r"^\d+(\.\d+){0,3}[.)]?\s+[A-Za-z]")
_KEYWORD = re.compile(r"^(section|article|clause|chapter|part|appendix|annexure|schedule)\b", re.IGNORECASE)
_SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with", "&"}
_BULLETS = "•-*–·▪○"


def looks_like_heading(line: str, next_line: str | None = None) -> bool:
    """Heuristic: short line, no sentence punctuation, numbered / UPPERCASE / Title Case."""
    words = line.split()
    if not words or len(words) > 12 or len(line) > 90:
        return False
    if line[0] in _BULLETS or line.endswith((".", ",", ";")):
        return False
    if sum(c.isalpha() for c in line) < 3:
        return False
    # A line that continues in lower case on the next line is a wrapped sentence, not a heading.
    if next_line and next_line[0].islower():
        return False
    if _NUMBERED.match(line) or _KEYWORD.match(line) or line.isupper():
        return True
    if len(words) <= 8:
        content = [w for w in words if w.lower() not in _SMALL_WORDS and w[0].isalpha()]
        return bool(content) and sum(w[0].isupper() for w in content) / len(content) >= 0.75
    return False


# --------------------------------------------------------------------------- chunking
Line = tuple[str, int, bool]  # (text, page number, is_heading)


def title_from_filename(name: str) -> str:
    return re.sub(r"[_\-\s]+", " ", Path(name).stem).strip()


def _word_count(lines: list[Line]) -> int:
    return sum(len(item[0].split()) for item in lines)


def _flatten(page_lines: list[list[str]], max_words: int) -> list[Line]:
    """Every line of the document with its page and heading flag; absurdly long lines are split."""
    flat: list[Line] = []
    for page_no, lines in enumerate(page_lines, start=1):
        for i, line in enumerate(lines):
            next_line = lines[i + 1] if i + 1 < len(lines) else None
            if looks_like_heading(line, next_line):
                flat.append((line, page_no, True))
                continue
            words = line.split()
            for start in range(0, len(words), max_words):
                flat.append((" ".join(words[start:start + max_words]), page_no, False))
    return flat


def _heading_before(lines: list[Line], index: int, inherited: str) -> str:
    """The heading in effect just before lines[index]."""
    for text, _, is_heading in reversed(lines[:index]):
        if is_heading:
            return text
    return inherited


def _label(lines: list[Line], inherited: str) -> str:
    """Sections a chunk covers: the heading it continues from, then every heading inside it."""
    names = [inherited] if inherited and not lines[0][2] else []
    names += [text for text, _, is_heading in lines if is_heading]
    unique = list(dict.fromkeys(names))
    return " / ".join(unique[:3]) + (" / ..." if len(unique) > 3 else "")


def chunk_document(doc: str, page_lines: list[list[str]], target_words: int,
                   overlap_words: int, min_words: int) -> list[Chunk]:
    title = title_from_filename(doc)
    chunks: list[Chunk] = []
    cur: list[Line] = []     # the chunk being built
    carried = 0              # leading lines of `cur` repeated from the previous chunk (overlap)
    section = ""             # latest heading seen
    inherited = ""           # heading in effect where `cur` begins (when it doesn't start with one)
    last_state: list[tuple[list[Line], str]] = []   # [(lines, start heading)] of the last emitted chunk

    def emit(lines: list[Line], start_heading: str) -> None:
        pages = [page for _, page, _ in lines]
        chunks.append(Chunk(
            id=f"{doc}#{len(chunks) + 1}", doc=doc, title=title,
            section=_label(lines, start_heading),
            page_start=min(pages), page_end=max(pages),
            text="\n".join(text for text, _, _ in lines),
        ))
        last_state[:] = [(list(lines), start_heading)]

    def overlap_tail(lines: list[Line]) -> list[Line]:
        tail: list[Line] = []
        for item in reversed(lines):
            if _word_count(tail) + len(item[0].split()) > overlap_words:
                break
            tail.insert(0, item)
        return tail

    for line, page, is_heading in _flatten(page_lines, target_words):
        if is_heading:
            if _word_count(cur[carried:]) >= min_words:   # previous section has enough text: close it
                emit(cur, inherited)
                cur, carried = [], 0
            cur.append((line, page, True))
            section = line
            continue
        if not cur:
            inherited = section
        cur.append((line, page, False))
        if _word_count(cur) >= target_words:
            full = cur
            emit(full, inherited)
            cur = overlap_tail(full)
            carried = len(cur)
            inherited = _heading_before(full, len(full) - carried, inherited)

    leftover = cur[carried:]
    if leftover:
        if _word_count(leftover) < min_words and last_state:
            previous_lines, previous_heading = last_state[0]   # too small to stand alone:
            chunks.pop()                                       # fold it into the previous chunk
            emit(previous_lines + leftover, previous_heading)
        else:
            emit(cur, inherited)
    return chunks


def ingest_pdf(path: Path, doc_name: str, target_words: int, overlap_words: int,
               min_words: int) -> PdfReport:
    """Read one PDF and turn it into chunks. Raises ValueError if it can't be read.

    `doc_name` is how the document is cited later (its path relative to the data folder).
    """
    pages, warnings = read_pdf(path)
    page_lines = clean_pages(pages)
    if not any(page_lines):
        raise ValueError("no extractable text (probably a scanned PDF - it needs OCR first)")
    empty = [n for n, lines in enumerate(page_lines, start=1) if not lines]
    if empty and len(empty) < len(page_lines):
        shown = ", ".join(map(str, empty[:8])) + ("..." if len(empty) > 8 else "")
        warnings.append(f"no text on page(s) {shown} (images or scans are not read)")
    chunks = chunk_document(doc_name, page_lines, target_words, overlap_words, min_words)
    return PdfReport(pages=len(pages), chunks=chunks, warnings=warnings)
