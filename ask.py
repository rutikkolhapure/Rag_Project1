"""Ask questions about your HR policy PDFs.

    py ask.py                                   chat mode
    py ask.py "How many casual leave days do I get?"
    py ask.py --retrieve-only "notice period"   show matching passages, no LLM call

Until your LLM is set up in .env it automatically runs in retrieval-only mode, so you can
check what the search finds before connecting the LLM. (Settings and key go in .env, see README.)
"""
from __future__ import annotations

import argparse
import shutil
import sys
import textwrap
from pathlib import Path

from hr_rag.config import DATA_DIR, HISTORY_TURNS, INDEX_DIR, LLM_MODEL, TOP_K, llm_setup_problem
from hr_rag.index import build_index, index_status, load_retriever
from hr_rag.llm import LLMError
from hr_rag.rag import Answer, HRAssistant
from hr_rag.vectorstore import HybridRetriever, prepare_search


def width() -> int:
    return max(60, min(100, shutil.get_terminal_size((100, 20)).columns - 2))


def ensure_index(data_dir: Path, index_dir: Path, allow_rebuild: bool) -> bool:
    status, reason = index_status(data_dir, index_dir)
    if status == "ok":
        return True
    if not allow_rebuild:
        print(f"Note: index is {status} ({reason}). Run  py build_index.py  to refresh it.")
        return status != "missing"
    print(f"Index is {status} ({reason}) - building it now ...")
    try:
        build_index(data_dir, index_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"\nError: {exc}")
        return False
    print()
    return True


def show(answer: Answer) -> None:
    w = width()
    if answer.notice:
        print(f"\nNote: {textwrap.fill(answer.notice, w, subsequent_indent='      ')}")
    if answer.used_llm:
        print()
        for paragraph in answer.text.split("\n"):
            print(textwrap.fill(paragraph, w, subsequent_indent="  " if paragraph.lstrip()[:1] in "-*•" else "")
                  if paragraph.strip() else "")
        print("\nSources:")
    elif not answer.hits:
        print(f"\n{answer.text}")
        return
    else:
        print("\nTop matching passages (retrieval-only mode):")

    by_number = {s.number: s for s in answer.sources}
    for number in sorted(by_number):
        print(f"  [{number}] {by_number[number].label()}")
        if not answer.used_llm:
            snippet = " ".join(answer.hits[number - 1].chunk.text.split())
            print(textwrap.fill(snippet[:450] + ("..." if len(snippet) > 450 else ""), w,
                                initial_indent="      ", subsequent_indent="      "))


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Ask questions about your HR policy PDFs.")
    parser.add_argument("question", nargs="*", help="a single question (omit for chat mode)")
    parser.add_argument("--retrieve-only", action="store_true", help="only show matching passages; no LLM call")
    parser.add_argument("--top-k", type=int, default=TOP_K, help=f"passages to retrieve (default {TOP_K})")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--no-rebuild", action="store_true", help="never rebuild the index automatically")
    parser.add_argument("--keyword-only", action="store_true",
                        help="skip Chroma (VECTOR_STORE) and use keyword search only")
    args = parser.parse_args()

    if not ensure_index(args.data_dir, args.index_dir, allow_rebuild=not args.no_rebuild):
        return 1

    use_llm = not args.retrieve_only
    problem = llm_setup_problem() if use_llm else None
    if problem:
        print(f"LLM not set up ({problem}) so running in retrieval-only mode (showing matching passages).\n"
              "To get written answers, fill in the LLM settings in the .env file (see README).\n")
        use_llm = False

    retriever = load_retriever(args.index_dir)
    search = retriever if args.keyword_only else prepare_search(args.index_dir, retriever)
    assistant = HRAssistant(search, top_k=args.top_k, use_llm=use_llm)

    if args.question:
        try:
            show(assistant.ask(" ".join(args.question)))
        except LLMError as exc:
            print(f"\nError: {exc}")
            return 1
        return 0

    docs = len({c.doc for c in retriever.chunks})
    mode = f"answers by {LLM_MODEL}" if use_llm else "retrieval-only"
    finder = "keyword + Chroma search" if isinstance(search, HybridRetriever) else "keyword search"
    print(f"HR Policy Assistant - {docs} document(s), {len(retriever.chunks)} passages, {finder}, {mode}.")
    print("Type a question. 'exit' quits, '/clear' forgets the conversation so far.")
    history: list[tuple[str, str]] = []
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question.lower() in {"exit", "quit", "/exit", "/quit"}:
            return 0
        if question == "/clear":
            history.clear()
            print("Conversation cleared.")
            continue
        try:
            answer = assistant.ask(question, history)
        except LLMError as exc:
            print(f"\nError: {exc}")
            continue
        show(answer)
        if answer.used_llm:
            history = (history + [(question, answer.text)])[-HISTORY_TURNS:]


if __name__ == "__main__":
    sys.exit(main())
