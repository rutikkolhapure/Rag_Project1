"""Tests for the Streamlit chat page (app.py), run headlessly with Streamlit's own AppTest harness.

No browser, key or network: the LLM answer is faked, and the index is built from sample PDFs in a temp folder.
Run from the project root:   .venv\\Scripts\\python -m unittest discover -s tests -v
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_pipeline import DOCUMENTS, HAVE_REPORTLAB, make_pdf  # noqa: E402

HAVE_STREAMLIT = importlib.util.find_spec("streamlit") is not None
if HAVE_STREAMLIT:
    import streamlit as st
    from streamlit.testing.v1 import AppTest

from hr_rag import config, llm  # noqa: E402

APP = str(ROOT / "app.py")
logging.getLogger("streamlit").setLevel(logging.ERROR)   # "missing ScriptRunContext" is expected outside a server


@unittest.skipUnless(HAVE_STREAMLIT and HAVE_REPORTLAB, "needs streamlit and reportlab")
class ChatPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.data = cls.root / "data"
        cls.data.mkdir()
        for name in ("Leave_Policy_2026.pdf", "Remote_Work_Policy.pdf"):
            make_pdf(cls.data / name, DOCUMENTS[name])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        st.cache_resource.clear()                        # the page caches its loaded index between runs
        self.index = Path(tempfile.mkdtemp(dir=self._tmp.name)) / "index"
        self.settings = {"DATA_DIR": self.data, "INDEX_DIR": self.index, "VECTOR_STORE": "",
                         "LLM_PROVIDER": "openai", "LLM_BASE_URL": "", "LLM_MODEL": ""}   # no LLM by default
        for patcher in (mock.patch.multiple(config, **self.settings),
                        mock.patch.dict(os.environ, {"LLM_API_KEY": ""})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def open(self) -> "AppTest":
        return AppTest.from_file(APP, default_timeout=60).run()

    @staticmethod
    def chat(app) -> list[tuple[str, str]]:
        """(who, all the text in that message) for every chat message on the page."""
        return [(m.name, " ".join(str(el.value) for el in m.markdown)) for m in app.chat_message]

    def ask(self, app, question: str):
        app.chat_input[0].set_value(question).run()
        self.assertFalse(app.exception, [e.value for e in app.exception])
        return app

    # ---- the page itself ---------------------------------------------------------
    def test_page_matches_the_sample_layout(self):
        app = self.open()
        self.assertFalse(app.exception, [e.value for e in app.exception])
        self.assertEqual(app.title[0].value, "HR Policy AI Assistant")
        self.assertIn("Ask me questions about", app.info[0].value)
        self.assertEqual([h.value for h in app.sidebar.header][:2], ["About", "Example Questions"])
        self.assertEqual(app.chat_input[0].placeholder, "What do you want to know?")
        labels = [b.label for b in app.sidebar.button]
        for example in ("How many casual leave days do I get?", "What is the dress code?"):
            self.assertIn(example, labels)
        self.assertIn("Reload documents", labels)

    def test_sidebar_status_reports_documents_and_missing_llm(self):
        status = " ".join(c.value for c in self.open().sidebar.caption)
        self.assertIn("2 document(s)", status)
        self.assertIn("Search: keyword", status)
        self.assertIn("LLM not connected", status)

    # ---- asking ------------------------------------------------------------------------
    def test_without_an_llm_the_matching_passages_are_shown(self):
        app = self.ask(self.open(), "How many casual leave days?")
        (who_user, _), (who_bot, text) = self.chat(app)
        self.assertEqual((who_user, who_bot), ("user", "assistant"))
        self.assertIn("isn't connected yet", text)
        expanders = [e.label for e in app.expander if e.label.startswith("[1]")]
        self.assertTrue(expanders and "Leave_Policy_2026.pdf" in expanders[0], expanders)
        self.assertIn("14 days", " ".join(m.value for m in app.chat_message[1].expander[0].markdown))

    def test_with_an_llm_the_answer_and_its_sources_are_shown(self):
        seen = []

        def fake_answer(question, hits, history):
            seen.append((question, list(history)))
            return "You get 14 days of casual leave [1]. A penalty of $5 or $6 applies."

        with mock.patch.object(config, "llm_setup_problem", return_value=None), \
                mock.patch("hr_rag.rag.generate_answer", side_effect=fake_answer):
            app = self.ask(self.open(), "How many casual leave days?")
            text = self.chat(app)[1][1]
            self.assertIn("14 days of casual leave [1]", text)
            self.assertIn("\\$5 or \\$6", text)                      # dollar signs escaped so they are not maths
            self.assertTrue([e for e in app.expander if e.label == "Sources (1)"])
            sources = " ".join(m.value for e in app.expander if e.label == "Sources (1)" for m in e.markdown)
            self.assertIn("Leave_Policy_2026.pdf", sources)
            self.ask(app, "What about sick leave?")                   # a follow-up
        self.assertEqual(seen[0][1], [])                              # nothing remembered yet
        self.assertEqual(seen[1][1][0][0], "How many casual leave days?")   # first exchange is remembered

    def test_clicking_an_example_question_asks_it(self):
        app = self.open()
        example = next(b for b in app.sidebar.button if b.label == "How many casual leave days do I get?")
        example.click().run()
        self.assertFalse(app.exception, [e.value for e in app.exception])
        self.assertEqual(self.chat(app)[0], ("user", "How many casual leave days do I get?"))
        self.assertEqual(len(app.chat_message), 2)

    def test_off_topic_question_says_nothing_was_found(self):
        app = self.ask(self.open(), "quantum physics lecture")
        self.assertIn("couldn't find anything", self.chat(app)[1][1])

    def test_an_llm_failure_is_shown_as_an_error_not_a_crash(self):
        with mock.patch.object(config, "llm_setup_problem", return_value=None), \
                mock.patch("hr_rag.rag.generate_answer", side_effect=llm.LLMError("The LLM service rejected the API key.")):
            app = self.ask(self.open(), "How many casual leave days?")
        self.assertIn("rejected the API key", app.error[0].value)

    def test_clear_conversation_empties_the_chat(self):
        app = self.ask(self.open(), "How many casual leave days?")
        self.assertEqual(len(app.chat_message), 2)
        next(b for b in app.sidebar.button if b.label == "Clear conversation").click().run()
        self.assertEqual(len(app.chat_message), 0)

    # ---- start-up problems --------------------------------------------------------
    def test_no_pdfs_shows_a_setup_message_instead_of_the_chat(self):
        empty = Path(tempfile.mkdtemp(dir=self._tmp.name))
        with mock.patch.object(config, "DATA_DIR", empty):
            app = self.open()
        self.assertFalse(app.exception, [e.value for e in app.exception])
        self.assertIn("No PDF files found", app.error[0].value)
        self.assertEqual(len(app.chat_input), 0)
        self.assertEqual(app.title[0].value, "HR Policy AI Assistant")   # the page still has its heading

    def test_a_chroma_problem_is_a_warning_and_keyword_search_still_works(self):
        with mock.patch.multiple(config, VECTOR_STORE="chroma", CHROMA_HOST="", EMBED_BASE_URL="", EMBED_MODEL=""):
            app = self.ask(self.open(), "How many casual leave days?")
        self.assertTrue([w for w in app.sidebar.warning if "Chroma is not set up" in w.value])
        self.assertIn("Leave_Policy_2026.pdf", " ".join(e.label for e in app.expander))


if __name__ == "__main__":
    unittest.main()
