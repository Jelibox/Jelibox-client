"""The Streamlit demo page renders (headless, via Streamlit's AppTest)."""
import os
import sys
import unittest

from tests.helpers import isolated_workspace, REPO

ctx = isolated_workspace()

try:
    from streamlit.testing.v1 import AppTest
    HAVE_STREAMLIT = True
except Exception:
    HAVE_STREAMLIT = False


@unittest.skipUnless(HAVE_STREAMLIT, "streamlit not installed")
class StreamPageTests(unittest.TestCase):
    SCRIPT = os.path.join(REPO, "utils", "stream_app.py")

    def run_page(self, model_path):
        old = sys.argv
        sys.argv = ["stream_app.py", "--", model_path]
        try:
            at = AppTest.from_file(self.SCRIPT, default_timeout=90)
            at.run()
            return at
        finally:
            sys.argv = old

    def page_html(self, at):
        return " ".join(m.value for m in at.markdown)

    def test_page_renders_with_theme_branding_and_logo(self):
        model = os.path.join(ctx.root, "dummy.pt")
        open(model, "wb").close()
        at = self.run_page(model)
        self.assertEqual([e.value for e in at.exception], [])
        html = self.page_html(at)
        self.assertIn("Jelibox", html)
        self.assertIn("bx-logo", html)
        self.assertIn("data:image/png;base64", html)
        self.assertNotIn("@ACCENT@", html, "all palette placeholders are substituted")
        from utils import theme
        self.assertIn(theme.C_ACCENT, html)

    def test_missing_model_shows_a_clear_error(self):
        at = self.run_page(os.path.join(ctx.root, "nope.pt"))
        self.assertTrue(any("Model not found" in e.value for e in at.error))


if __name__ == "__main__":
    unittest.main()
