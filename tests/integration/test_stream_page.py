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

    # ---- a YouTube link is downloaded first, then read from the file (OpenCV cannot open the address yt-dlp finds)
    URL = "https://www.youtube.com/watch?v=abc"

    def press_start(self, at):
        old = sys.argv
        sys.argv = ["stream_app.py", "--", self.model]                  # the page reads the model from argv on every run
        try:
            at.text_input(key="youtube_url").set_value(self.URL).run()
            next(b for b in at.button if "Start" in b.label).click().run()
        finally:
            sys.argv = old

    def page_with_model(self):
        self.model = os.path.join(ctx.root, "dummy.pt")
        open(self.model, "wb").close()
        return self.run_page(self.model)

    def test_a_refused_youtube_download_is_shown_and_nothing_starts(self):
        from unittest import mock
        from utils import stream_support as ss
        at = self.page_with_model()
        with mock.patch.object(ss, "download_youtube", side_effect=ss.StreamError("YouTube refused (403)")) as dl:
            self.press_start(at)
        self.assertEqual(dl.call_args.args[0], self.URL)
        self.assertEqual(dl.call_args.args[1], ss.stream_folder(os.path.join(REPO)))       # saved in temp/stream
        self.assertTrue(any("YouTube refused (403)" in e.value for e in at.error), [e.value for e in at.error])
        self.assertFalse(at.session_state["streaming"])

    def test_a_download_that_is_not_a_video_says_so_instead_of_spinning(self):
        from unittest import mock
        from utils import stream_support as ss
        junk = os.path.join(ctx.root, "not_a_video.mp4")
        with open(junk, "wb") as f:
            f.write(b"this is not a video")
        at = self.page_with_model()
        with mock.patch.object(ss, "download_youtube", return_value=junk):
            self.press_start(at)
        self.assertTrue(any("could not be opened" in e.value for e in at.error), [e.value for e in at.error])
        self.assertFalse(at.session_state["streaming"])

    def test_the_downloaded_file_is_what_gets_played(self):
        import cv2
        import numpy as np
        from unittest import mock
        from utils import stream_support as ss
        video = os.path.join(ctx.root, "downloaded.avi")
        out = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
        for _ in range(3):
            out.write(np.zeros((48, 64, 3), dtype=np.uint8))
        out.release()
        at = self.page_with_model()
        with mock.patch.object(ss, "download_youtube", return_value=video):
            self.press_start(at)                                    # then the page runs the model, which cannot load dummy.pt: fine here
        self.assertEqual(at.session_state["target_vid"], video)
        self.assertTrue(at.session_state["streaming"])


if __name__ == "__main__":
    unittest.main()
