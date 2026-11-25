"""HR Policy AI Assistant: a chat window for your HR policy PDFs.

Start it (PowerShell, from this folder):
    .venv\\Scripts\\python -m streamlit run app.py

It uses the same search and LLM settings as ask.py (see .env and README). Until an LLM is set up in .env
it shows the matching policy passages instead of a written answer.
"""
from __future__ import annotations

import streamlit as st

from hr_rag import config, service
from hr_rag.llm import LLMError
from hr_rag.rag import Answer

TITLE = "HR Policy AI Assistant"
INTRO = "Ask me questions about leave, notice periods, remote work, conduct, expenses and other HR policies!"
ABOUT = ("This assistant answers questions about your company's HR policies. It searches the policy PDFs in "
         "the data folder and writes an answer from the matching passages, citing the document, page and "
         "section so you can check the source. If the documents don't cover a question, it says so instead of "
         "guessing. For decisions that affect you, confirm with HR.")
# Edit these to match the policies you have indexed. Clicking one asks it.
EXAMPLE_QUESTIONS = [
    "How many casual leave days do I get?",
    "What is the notice period for resignation?",
    "Can I work from home, and how often?",
    "What are the maternity and paternity leave rules?",
    "How do I report harassment?",
    "What is the dress code?",
    "How do I claim travel expenses?",
]
SNIPPET_CHARS = 600

st.set_page_config(page_title=TITLE, page_icon="📄")   # must be the first Streamlit call


def md(text: str) -> str:
    """Streamlit markdown treats $...$ as maths, and HR answers mention amounts, so escape the dollar signs."""
    return text.replace("$", "\\$")


@st.cache_resource(show_spinner="Reading the policy documents ...")
def get_backend() -> service.Backend:
    return service.start()


# ---- button callbacks (they run before the page is redrawn) -----------------------------------
def ask_example(question: str) -> None:
    st.session_state["pending_question"] = question


def reload_documents() -> None:
    get_backend.clear()


def clear_conversation() -> None:
    st.session_state["messages"] = []
    st.session_state["history"] = []


# ---- showing messages ------------------------------------------------------------------------
def to_message(answer: Answer) -> dict:
    """Plain data (so it survives Streamlit reruns) describing what to show for one answer."""
    mode = "answer" if answer.used_llm else ("passages" if answer.hits else "none")
    passages = []
    if mode == "passages":   # no LLM yet: show the matching text itself
        for source in answer.sources:
            text = " ".join(answer.hits[source.number - 1].chunk.text.split())
            passages.append((source.number, source.label(),
                             text[:SNIPPET_CHARS] + ("..." if len(text) > SNIPPET_CHARS else "")))
    return {"role": "assistant", "mode": mode, "content": answer.text, "notice": answer.notice,
            "sources": [(s.number, s.label()) for s in answer.sources], "passages": passages}


def render(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(md(message["content"]))
        elif message.get("error"):
            st.error(message["error"])
        else:
            if message["notice"]:
                st.warning(message["notice"])
            if message["mode"] == "passages":
                st.markdown("The LLM isn't connected yet, so here are the policy passages that best match "
                            "your question:")
                for number, label, text in message["passages"]:
                    with st.expander(f"[{number}] {label}"):
                        st.markdown(md(text))
            else:
                st.markdown(md(message["content"]))
                if message["sources"]:
                    with st.expander(f"Sources ({len(message['sources'])})"):
                        for number, label in message["sources"]:
                            st.markdown(f"- **[{number}]** {md(label)}")


# ---- page ------------------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(ABOUT)
    st.header("Example Questions")
    for number, example in enumerate(EXAMPLE_QUESTIONS):
        st.button(example, key=f"example_{number}", on_click=ask_example, args=(example,), use_container_width=True)

st.title(TITLE)
st.info(INTRO)

try:
    backend = get_backend()
except service.SetupError as exc:
    st.error(f"{exc}\n\nThen press **Reload documents** in the sidebar.")
    with st.sidebar:
        st.button("Reload documents", on_click=reload_documents, use_container_width=True)
    st.stop()

with st.sidebar:
    st.header("Status")
    documents = backend.documents
    st.caption(f"{len(documents)} document(s), {len(backend.chunks)} passages")
    with st.expander("Indexed documents"):
        for name, count in documents:
            st.markdown(f"- {md(name)} ({count} passages)")
    st.caption("Search: keyword + Chroma" if backend.hybrid else "Search: keyword")
    if backend.llm_problem:
        st.caption(f"Answers: LLM not connected ({backend.llm_problem})")
    else:
        st.caption(f"Answers: written by {config.LLM_MODEL}")
    for note in backend.notes:
        st.warning(note)
    for name, reason in backend.skipped.items():
        st.warning(f"Skipped {name}: {reason}")
    st.button("Reload documents", on_click=reload_documents, use_container_width=True)
    st.button("Clear conversation", on_click=clear_conversation, use_container_width=True)

messages = st.session_state.setdefault("messages", [])
history = st.session_state.setdefault("history", [])
for message in messages:
    render(message)

question = st.chat_input("What do you want to know?") or st.session_state.pop("pending_question", None)
if question:
    user_message = {"role": "user", "content": question}
    messages.append(user_message)
    render(user_message)
    with st.spinner("Searching the policies ..."):
        try:
            answer = backend.assistant.ask(question, history)
            reply = to_message(answer)
            if answer.used_llm:   # same as ask.py: only written answers are remembered for follow-ups
                history.append((question, answer.text))
                del history[:-config.HISTORY_TURNS]
        except LLMError as exc:
            reply = {"role": "assistant", "error": str(exc)}
    messages.append(reply)
    render(reply)
