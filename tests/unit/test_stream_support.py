"""utils/stream_support.py: video names, the YouTube download (a fake yt-dlp), and what Clear cache removes."""
import os
import shutil
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from utils import stream_support as ss


def touch(path, data=b"x", age=0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    if age:
        old = time.time() - age
        os.utime(path, (old, old))
    return path


class Base(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="jelibox_stream_")
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)
        self.folder = ss.stream_folder(self.base)
        self.temp = os.path.join(self.base, "temp")


class NameTests(Base):
    def test_the_folder_is_temp_stream(self):
        self.assertEqual(self.folder, os.path.join(self.base, "temp", "stream"))

    def test_an_uploaded_name_is_cleaned_and_kept_unique(self):
        os.makedirs(self.folder)
        self.assertEqual(ss.clean_video_name("Street cam.MP4", self.folder), "Street cam.mp4")
        touch(os.path.join(self.folder, "Street cam.mp4"))
        self.assertEqual(ss.clean_video_name("Street cam.mp4", self.folder), "Street cam_2.mp4")
        self.assertEqual(ss.clean_video_name("..\\..\\evil/../a b?*.avi", self.folder), "a b__.avi")   # only the last part, odd characters replaced
        self.assertEqual(ss.clean_video_name("CON.mp4", self.folder), "_CON.mp4")
        self.assertEqual(ss.clean_video_name("...x.mp4", self.folder), "x.mp4")                       # leading dots would hide the file
        self.assertLessEqual(len(ss.clean_video_name("x" * 400 + ".mkv", self.folder)), 110)

    def test_only_videos_are_taken(self):
        for bad in ("notes.txt", "a.exe", "movie", "", None, "a.mp4.zip", ".mp4"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                ss.clean_video_name(bad, self.folder)

    def test_a_video_is_found_by_its_bare_name_only(self):
        path = touch(os.path.join(self.folder, "a.mp4"))
        self.assertEqual(ss.video_path(self.base, "a.mp4"), path)
        for bad in ("../a.mp4", "sub/a.mp4", "", "ghost.mp4", "a.txt", ".hidden.mp4", None, "..\\a.mp4"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                ss.video_path(self.base, bad)

    def test_the_list_has_the_newest_first_and_skips_other_files(self):
        touch(os.path.join(self.folder, "old.mp4"), b"1234", age=500)
        touch(os.path.join(self.folder, "new.avi"), b"12")
        touch(os.path.join(self.folder, "x.mp4.part"))
        rows = ss.list_videos(self.base)
        self.assertEqual([(r["name"], r["bytes"]) for r in rows], [("new.avi", 2), ("old.mp4", 4)])
        self.assertEqual(ss.list_videos(os.path.join(self.base, "nowhere")), [])


class UrlTests(unittest.TestCase):
    def test_youtube_links_are_accepted(self):
        for url in ("https://www.youtube.com/watch?v=abc", "http://youtube.com/watch?v=abc", "https://youtu.be/abc",
                    "https://m.youtube.com/watch?v=abc", "  https://music.youtube.com/watch?v=abc  "):
            self.assertEqual(ss.check_youtube_url(url), url.strip())

    def test_everything_else_is_refused_so_the_server_cannot_be_pointed_at_the_network(self):
        for url in ("http://127.0.0.1:8420/api/system", "http://192.168.1.1/x.mp4", "file:///etc/passwd", "ftp://youtube.com/x",
                    "https://youtube.com.evil.com/watch", "https://evil.com/?u=youtube.com", "https://notyoutube.com/x",
                    "javascript:alert(1)", "youtube.com/watch?v=abc", "", None, "https://[bad"):
            with self.assertRaises(ValueError, msg=repr(url)):
                ss.check_youtube_url(url)


class ErrorTextTests(unittest.TestCase):
    def test_a_403_says_what_to_do(self):
        text = ss.friendly_error(Exception("ERROR: unable to download video data: HTTP Error 403: Forbidden"))
        self.assertIn("403", text)
        self.assertIn("yt-dlp", text)
        self.assertIn("upload", text.lower())

    def test_the_bot_check_says_upload(self):
        text = ss.friendly_error(Exception("ERROR: [youtube] x: Sign in to confirm you're not a bot"))
        self.assertIn("upload", text.lower())

    def test_an_unreachable_youtube_points_at_the_network(self):
        self.assertIn("DNS", ss.friendly_error(Exception("<urlopen error [Errno -2] Name or service not known>")))

    def test_other_errors_keep_the_last_line_without_colour_codes(self):
        text = ss.friendly_error(Exception("\x1b[0;31mERROR:\x1b[0m something odd\nERROR: the real reason"))
        self.assertTrue(text.endswith("the real reason"), text)
        self.assertNotIn("\x1b", text)
        self.assertIn("private", ss.friendly_error(Exception("ERROR: Private video. Sign in")).lower() + "private")


def fake_yt_dlp(write=True, error=None, calls=None, size=64, warn=()):
    """A stand-in for the yt_dlp module: YoutubeDL(options).extract_info writes the file the options point at."""
    class YoutubeDL:
        def __init__(self, options):
            self.options = options
            if calls is not None:
                calls.append(options)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download=True):
            folder = os.path.dirname(self.options["outtmpl"])
            partial = os.path.join(folder, "yt_abc.mp4.part")
            touch(partial)
            for text in warn:
                self.options["logger"].warning(text)
            for done in (size // 4, size // 2, size):
                for hook in self.options["progress_hooks"]:
                    hook({"status": "downloading", "downloaded_bytes": done, "total_bytes": size})
            if error:
                raise error
            os.remove(partial)
            path = os.path.join(folder, "yt_abc.mp4")
            if write:
                touch(path, b"v" * size)
            return {"id": "abc", "ext": "mp4", "requested_downloads": [{"filepath": path}]}

        def prepare_filename(self, info):
            return os.path.join(os.path.dirname(self.options["outtmpl"]), "yt_abc.mp4")

    module = types.ModuleType("yt_dlp")
    module.YoutubeDL = YoutubeDL
    return module


class DownloadTests(Base):
    URL = "https://www.youtube.com/watch?v=abc"

    def run_download(self, module, **kwargs):
        with mock.patch.dict(sys.modules, {"yt_dlp": module}):
            return ss.download_youtube(self.URL, self.folder, **kwargs)

    def test_the_video_is_downloaded_into_the_stream_folder_and_progress_is_reported(self):
        calls, seen = [], []
        path = self.run_download(fake_yt_dlp(calls=calls), progress=lambda pct, text: seen.append((round(pct), text)))
        self.assertEqual(path, os.path.join(self.folder, "yt_abc.mp4"))
        self.assertEqual(os.path.getsize(path), 64)
        self.assertEqual([p for p, _ in seen], [25, 50, 99])                    # never 100 before it is saved
        self.assertIn("MB", seen[0][1])
        options = calls[0]
        self.assertTrue(options["noplaylist"])
        self.assertIn("[ext=mp4]", options["format"])
        self.assertEqual(os.path.dirname(options["outtmpl"]), self.folder)

    def test_only_a_youtube_link_reaches_yt_dlp(self):
        calls = []
        with mock.patch.dict(sys.modules, {"yt_dlp": fake_yt_dlp(calls=calls)}):
            with self.assertRaises(ValueError):
                ss.download_youtube("http://127.0.0.1:8420/api/system", self.folder)
        self.assertEqual(calls, [])

    def test_a_failed_download_gives_a_readable_error_and_leaves_no_pieces(self):
        with self.assertRaises(ss.StreamError) as ctx:
            self.run_download(fake_yt_dlp(error=Exception("ERROR: unable to download video data: HTTP Error 403: Forbidden")))
        self.assertIn("403", str(ctx.exception))
        self.assertEqual(os.listdir(self.folder), [])

    def test_what_yt_dlp_says_is_missing_is_passed_on_with_a_failure(self):
        warning = "[youtube] No supported JavaScript runtime could be found. Only deno is enabled by default"
        with self.assertRaises(ss.StreamError) as ctx:
            self.run_download(fake_yt_dlp(error=Exception("HTTP Error 403: Forbidden"), warn=[warning]))
        self.assertIn("403", str(ctx.exception))
        self.assertIn("JavaScript runtime", str(ctx.exception))
        self.assertIn("deno", str(ctx.exception))
        with self.assertRaises(ss.StreamError) as ctx:                                 # no complaint, no hint
            self.run_download(fake_yt_dlp(error=Exception("HTTP Error 403: Forbidden"), warn=["something else"]))
        self.assertNotIn("JavaScript", str(ctx.exception))

    def test_yt_dlp_output_goes_to_our_logger_not_the_console(self):
        calls = []
        self.run_download(fake_yt_dlp(calls=calls))
        self.assertTrue(hasattr(calls[0]["logger"], "warning"))
        self.assertNotIn("quiet", calls[0])

    def test_stopping_raises_cancelled_and_removes_the_unfinished_file(self):
        with self.assertRaises(ss.Cancelled):
            self.run_download(fake_yt_dlp(), cancelled=lambda: True)
        self.assertEqual(os.listdir(self.folder), [])

    def test_a_download_that_saved_nothing_is_an_error(self):
        with self.assertRaises(ss.StreamError) as ctx:
            self.run_download(fake_yt_dlp(write=False))
        self.assertIn("not saved", str(ctx.exception))

    def test_without_yt_dlp_it_says_how_to_install_it(self):
        with mock.patch.dict(sys.modules, {"yt_dlp": None}):
            with self.assertRaises(ss.StreamError) as ctx:
                ss.download_youtube(self.URL, self.folder)
        self.assertIn("pip install yt-dlp", str(ctx.exception))

    def test_the_files_already_there_are_not_touched_by_a_failed_download(self):
        keep = touch(os.path.join(self.folder, "mine.mp4"))
        with self.assertRaises(ss.StreamError):
            self.run_download(fake_yt_dlp(error=Exception("boom")))
        self.assertTrue(os.path.exists(keep))


class CacheTests(Base):
    def fill(self):
        self.video = touch(os.path.join(self.folder, "a.mp4"), b"a" * 100)
        self.other = touch(os.path.join(self.folder, "b.mkv"), b"b" * 50)
        self.part = touch(os.path.join(self.folder, "c.mp4.part"), b"p" * 10)
        self.old_upload = touch(os.path.join(self.temp, "uploads", "0123456789abcdef", "x.jpg"), b"u" * 20)
        self.old_export = touch(os.path.join(self.temp, "downloads", "abc.zip"), b"z" * 30)
        for path, age in ((self.old_upload, 25 * 3600), (self.old_export, 7 * 3600)):
            os.utime(os.path.dirname(path) if "uploads" in path else path, (time.time() - age,) * 2)
        self.fresh_export = touch(os.path.join(self.temp, "downloads", "fresh.zip"))
        self.fresh_upload = touch(os.path.join(self.temp, "uploads", "fedcba9876543210", "y.jpg"))
        self.log_old = touch(os.path.join(self.temp, "server.log.old"), b"l" * 5)
        self.log = touch(os.path.join(self.temp, "server.log"))
        self.lock = touch(os.path.join(self.temp, "training.lock"))

    def test_the_summary_counts_videos_and_junk_separately(self):
        self.fill()
        s = ss.cache_summary(self.base)
        self.assertEqual((s["videos"], s["video_bytes"]), (2, 150))
        self.assertEqual(s["junk"], 4)                                           # .part, old upload folder, old export, server.log.old
        self.assertEqual(s["junk_bytes"], 10 + 20 + 30 + 5)

    def test_clearing_removes_the_videos_and_the_junk_and_nothing_else(self):
        self.fill()
        r = ss.clear_cache(self.base)
        self.assertEqual((r["removed"], r["bytes"], r["failed"], r["kept"]), (6, 150 + 65, 0, []))
        for gone in (self.video, self.other, self.part, os.path.dirname(self.old_upload), self.old_export, self.log_old):
            self.assertFalse(os.path.exists(gone), gone)
        for stays in (self.fresh_export, os.path.dirname(self.fresh_upload), self.log, self.lock):
            self.assertTrue(os.path.exists(stays), stays)                         # in use, or not ours
        self.assertTrue(os.path.isdir(self.folder))                              # the folder itself stays
        self.assertEqual(ss.cache_summary(self.base)["videos"], 0)

    def test_the_video_that_plays_is_kept(self):
        self.fill()
        r = ss.clear_cache(self.base, keep={"a.mp4"})
        self.assertEqual(r["kept"], ["a.mp4"])
        self.assertTrue(os.path.exists(self.video))
        self.assertFalse(os.path.exists(self.other))
        self.assertEqual(ss.cache_summary(self.base, {"a.mp4"})["videos"], 0)

    def test_an_empty_or_missing_folder_is_fine(self):
        self.assertEqual(ss.clear_cache(self.base), {"removed": 0, "bytes": 0, "failed": 0, "kept": []})
        self.assertEqual(ss.cache_summary(self.base)["videos"], 0)

    def test_nothing_outside_temp_is_ever_deleted(self):
        outside = touch(os.path.join(self.base, "datasetsInput", "pets-1", "a.mp4"))
        model = touch(os.path.join(self.base, "models", "pets", "modelAssistant.pt"))
        self.fill()
        ss.clear_cache(self.base)
        self.assertTrue(os.path.exists(outside) and os.path.exists(model))

    def test_a_link_in_the_folder_is_removed_without_following_it(self):
        target = touch(os.path.join(self.base, "precious.mp4"))
        os.makedirs(self.folder)
        link = os.path.join(self.folder, "link.mp4")
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError):
            self.skipTest("symbolic links are not available here")
        ss.clear_cache(self.base)
        self.assertTrue(os.path.exists(target))


if __name__ == "__main__":
    unittest.main()
