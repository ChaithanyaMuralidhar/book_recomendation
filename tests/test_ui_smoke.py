"""Runs the real Streamlit app headlessly (skipped automatically if streamlit is not installed)."""
import os
import tempfile
import unittest

os.environ.setdefault("BOOKREC_DB", os.path.join(tempfile.mkdtemp(), "ui_test.db"))

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover
    AppTest = None

PAGES = ["🏠 Home Dashboard", "📷 Scan Books", "🎥 Video / Webcam / VR", "📊 Excel Import & Export",
         "📖 Book Catalogue", "👤 Profile & Reading History", "✨ AI Recommendations",
         "🏛️ Scan a Library / Store", "📈 ML Analytics & Clustering", "⚙️ Settings & Help"]


@unittest.skipUnless(AppTest, "streamlit not installed")
class UISmokeTests(unittest.TestCase):
    def test_every_page_renders_empty_then_with_demo_data(self):
        at = AppTest.from_file("app.py", default_timeout=60).run()
        self.assertFalse(at.exception)
        for p in PAGES:
            at.session_state["page"] = p
            at.run()
            self.assertFalse(at.exception, f"{p}: {at.exception}")
        at.session_state["page"] = PAGES[0]
        at.run()
        next(b for b in at.button if "Load demo" in b.label).click().run()
        for p in PAGES:
            at.session_state["page"] = p
            at.run()
            self.assertFalse(at.exception, f"{p} (demo data): {at.exception}")


if __name__ == "__main__":
    unittest.main()
