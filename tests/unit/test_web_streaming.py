"""The web Stream: videos (upload, YouTube), a running stream with a fake model, the feed, the idle stop and Clear cache."""
import os
import sys
import time
import unittest
from unittest import mock

from tests.unit.test_stream_support import fake_yt_dlp, touch
from tests.unit.test_web_app import H
from tests.unit.test_web_editor import INSTANCE, EditorBase

try:
    import cv2
    import numpy as np
except Exception:                                                    # the web tests also run where cv2 is missing
    cv2 = None


def make_video(path, frames=8, fps=20, size=(64, 48)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(frames):
        out.write(np.full((size[1], size[0], 3), (20 * i) % 256, dtype=np.uint8))
    out.release()
    return path


class FakeBoxes:
    def __init__(self, cls):
        self.cls = cls

    def __len__(self):
        return len(self.cls)


class FakeResult:
    names = {0: "cat", 1: "dog"}

    def __init__(self, frame, cls):
        self.frame, self.boxes = frame, FakeBoxes(cls)

    def plot(self):
        return self.frame.copy()


def fake_loader(path):
    def model(frame, conf, verbose):
        model.confs.append(conf)
        return [FakeResult(frame, [0, 0, 1])]
    model.confs = []
    fake_loader.last = model
    return model


class StreamBase(EditorBase):
    def setUp(self):
        super().setUp()
        self.streams = self.app.state.streams
        self.streams.load_model = fake_loader
        self.folder = self.streams.folder
        self.url = f"{self.base}/stream"
        self.addCleanup(self.stop_everything)

    def stop_everything(self):
        session = self.streams.current()
        if session:
            session.stop()
            self.wait_for(lambda: not session.alive)

    def give_model(self):
        touch(os.path.join(self.tmp, "models", "pets", "modelAssistant.pt"), b"weights")

    def wait_for(self, check, seconds=10):
        end = time.time() + seconds
        while time.time() < end:
            if check():
                return True
            time.sleep(0.02)
        self.fail("did not happen in time")

    def video(self, name="clip.avi", **kwargs):
        return make_video(os.path.join(self.folder, name), **kwargs)

    def start(self, **body):
        return self.client.post(f"{self.url}/start", json={"video": "clip.avi", "conf": 0.4, **body}, headers=H)


class InfoTests(StreamBase):
    def test_what_the_window_shows(self):
        data = self.client.get(self.url).json()
        self.assertEqual((data["model"], data["videos"], data["session"]), (None, [], None))
        self.assertEqual(data["cache"]["videos"], 0)
        self.assertIsInstance(data["youtube"], bool)
        self.give_model()
        touch(os.path.join(self.folder, "a.mp4"), b"1234")
        data = self.client.get(self.url).json()
        self.assertEqual(data["model"], "modelAssistant.pt")
        self.assertEqual([(v["name"], v["bytes"]) for v in data["videos"]], [("a.mp4", 4)])
        self.assertEqual((data["cache"]["videos"], data["cache"]["video_bytes"]), (1, 4))

    def test_everything_needs_a_session_and_the_header(self):
        self.client.post("/api/logout", json={}, headers=H)
        self.assertEqual(self.client.get(self.url).status_code, 401)
        self.assertEqual(self.client.put("/api/stream/upload?name=a.mp4", content=b"x", headers=H).status_code, 401)
        self.assertEqual(self.client.post("/api/stream/clear-cache", json={}, headers=H).status_code, 401)
        self.assertEqual(self.client.get("/api/stream/0123456789abcdef/feed").status_code, 401)
        self.signed_in_again()

    def signed_in_again(self):
        self.login(password="adminpass1")
        self.assertEqual(self.client.post("/api/stream/clear-cache", json={}).status_code, 403)     # no X-Jelibox header
        self.assertEqual(self.client.put("/api/stream/upload?name=a.mp4", content=b"x").status_code, 403)

    def test_an_unknown_dataset_is_a_404(self):
        self.assertEqual(self.client.get("/api/ws/ghost-1/stream").status_code, 404)


class UploadTests(StreamBase):
    def put(self, name, data=b"video bytes"):
        return self.client.put(f"/api/stream/upload?name={name}", content=data, headers={**H, "Content-Type": "application/octet-stream"})

    def test_a_video_lands_in_temp_stream(self):
        res = self.put("street cam.mp4", b"x" * 5000)
        self.assertEqual((res.status_code, res.json()), (201, {"name": "street cam.mp4", "bytes": 5000}))
        self.assertEqual(os.listdir(self.folder), ["street cam.mp4"])
        self.assertEqual(os.path.getsize(os.path.join(self.folder, "street cam.mp4")), 5000)
        self.assertEqual(os.path.dirname(self.folder), os.path.join(self.tmp, "temp"))

    def test_the_same_name_twice_keeps_both(self):
        self.put("a.mp4", b"1")
        self.assertEqual(self.put("a.mp4", b"22").json()["name"], "a_2.mp4")
        self.assertEqual(sorted(os.listdir(self.folder)), ["a.mp4", "a_2.mp4"])

    def test_names_cannot_leave_the_folder(self):
        res = self.put("..%2F..%2Fevil.mp4")
        self.assertEqual((res.status_code, res.json()["name"]), (201, "evil.mp4"))
        self.assertEqual(os.listdir(self.folder), ["evil.mp4"])
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "evil.mp4")))

    def test_only_videos_and_not_empty_ones(self):
        for name in ("notes.txt", "a.exe", "a"):
            self.assertEqual(self.put(name).status_code, 400, name)
        res = self.put("a.mp4", b"")
        self.assertEqual(res.status_code, 400)
        self.assertEqual(os.listdir(self.folder) if os.path.isdir(self.folder) else [], [])        # no .part left behind

    def test_a_full_disk_is_refused_without_leaving_a_file(self):
        with mock.patch("utils.web.importing.free_bytes", return_value=10):
            res = self.put("a.mp4", b"x" * 100)
        self.assertEqual(res.status_code, 507)
        self.assertEqual(os.listdir(self.folder), [])


class YoutubeTests(StreamBase):
    URL = "https://www.youtube.com/watch?v=abc"

    def job(self, module, url=None):
        with mock.patch.dict(sys.modules, {"yt_dlp": module}):
            res = self.client.post("/api/stream/youtube", json={"url": url or self.URL}, headers=H)
            self.assertEqual(res.status_code, 202, res.text)
            job = res.json()
            for _ in range(300):
                job = self.client.get(f"/api/jobs/{job['id']}").json()
                if job["state"] != "running":
                    return job
                time.sleep(0.02)
        self.fail("the job did not end")

    def test_the_video_is_downloaded_into_temp_stream(self):
        job = self.job(fake_yt_dlp())
        self.assertEqual((job["state"], job["result"]), ("done", {"name": "yt_abc.mp4", "bytes": 64}))
        self.assertTrue(os.path.isfile(os.path.join(self.folder, "yt_abc.mp4")))
        self.assertEqual([v["name"] for v in self.client.get(self.url).json()["videos"]], ["yt_abc.mp4"])

    def test_a_link_that_is_not_youtube_is_refused_before_a_job_starts(self):
        for url in ("http://127.0.0.1:8420/api/system", "file:///etc/passwd", "", "https://evil.com/watch?v=abc"):
            res = self.client.post("/api/stream/youtube", json={"url": url}, headers=H)
            self.assertEqual(res.status_code, 400, url)
            self.assertIn("YouTube", res.json()["error"])

    def test_a_refused_download_says_what_to_do(self):
        job = self.job(fake_yt_dlp(error=Exception("ERROR: unable to download video data: HTTP Error 403: Forbidden")))
        self.assertEqual(job["state"], "error")
        self.assertIn("403", job["error"])
        self.assertIn("upload", job["error"].lower())
        self.assertEqual(os.listdir(self.folder), [])

    def test_stopping_the_download_ends_the_job_as_cancelled(self):
        module = fake_yt_dlp()
        real = module.YoutubeDL.extract_info

        def slow(self, url, download=True):
            time.sleep(0.4)                                          # the test cancels meanwhile
            return real(self, url, download)
        module.YoutubeDL.extract_info = slow
        with mock.patch.dict(sys.modules, {"yt_dlp": module}):
            job = self.client.post("/api/stream/youtube", json={"url": self.URL}, headers=H).json()
            self.assertEqual(self.client.post(f"/api/jobs/{job['id']}/cancel", json={}, headers=H).status_code, 200)
            for _ in range(300):
                job = self.client.get(f"/api/jobs/{job['id']}").json()
                if job["state"] != "running":
                    break
                time.sleep(0.02)
        self.assertEqual(job["state"], "cancelled")
        self.assertEqual(os.listdir(self.folder), [])

    def test_the_cache_is_not_cleared_while_a_download_runs(self):
        with self.streams.transfer():
            res = self.client.post("/api/stream/clear-cache", json={}, headers=H)
        self.assertEqual(res.status_code, 409)
        self.assertIn("running", res.json()["error"])


@unittest.skipIf(cv2 is None, "OpenCV is not installed")
class StreamTests(StreamBase):
    def test_start_needs_a_model_and_a_video(self):
        self.video()
        res = self.start()
        self.assertEqual(res.status_code, 400)
        self.assertIn("no trained model", res.json()["error"])
        self.give_model()
        for bad in ("ghost.avi", "", "../clip.avi", "clip.txt"):
            self.assertEqual(self.start(video=bad).status_code, 400, bad)
        for bad in (0, 1, "high"):
            self.assertEqual(self.start(conf=bad).status_code, 400, bad)
        self.assertEqual(self.start(conf=0.95).status_code, 201)

    def test_a_stream_runs_the_model_and_reports_numbers(self):
        self.video(frames=6)
        self.give_model()
        res = self.start()
        self.assertEqual(res.status_code, 201, res.text)
        sid = res.json()["id"]
        self.assertEqual((res.json()["video"], res.json()["workspace"], res.json()["conf"]), ("clip.avi", "pets", 0.4))
        self.wait_for(lambda: self.client.get(f"/api/stream/{sid}").json()["frames"] >= 8)           # 6 frames: it loops
        data = self.client.get(f"/api/stream/{sid}").json()
        self.assertEqual(data["state"], "running")
        self.assertGreaterEqual(data["loops"], 1)
        self.assertEqual(data["counts"], {"cat": 2, "dog": 1})
        self.assertEqual(data["detections"], data["frames"] * 3)
        self.assertEqual((data["width"], data["height"]), (64, 48))
        self.assertGreater(data["fps"], 0)
        self.assertEqual(self.client.get(self.url).json()["session"]["id"], sid)                     # the window finds it again

    def test_the_confidence_can_change_while_it_runs(self):
        self.video(frames=40)
        self.give_model()
        sid = self.start(conf=0.4).json()["id"]
        self.assertEqual(self.client.post(f"/api/stream/{sid}", json={"conf": 0.7}, headers=H).json()["conf"], 0.7)
        self.wait_for(lambda: 0.7 in fake_loader.last.confs)
        self.assertEqual(self.client.post(f"/api/stream/{sid}", json={"conf": 5}, headers=H).status_code, 400)
        self.assertEqual(self.client.post(f"/api/stream/{sid}", json={"conf": "x"}, headers=H).status_code, 400)

    def test_stop_ends_it_and_frees_the_model_for_the_next_job(self):
        from utils.web import assistant
        self.video()
        self.give_model()
        sid = self.start().json()["id"]
        self.assertEqual(self.client.post(f"/api/stream/{sid}/stop", json={}, headers=H).status_code, 200)
        self.wait_for(lambda: self.client.get(f"/api/stream/{sid}").json()["state"] == "ended")
        assistant.acquire("someone", "Infer")                         # the slot is free again
        assistant.release()
        self.assertIsNone(self.client.get(self.url).json()["session"])

    def test_one_model_job_at_a_time(self):
        from utils.web import assistant
        self.video()
        self.give_model()
        assistant.acquire("bob", "Auto-annotate all")
        try:
            res = self.start()
        finally:
            assistant.release()
        self.assertEqual(res.status_code, 409)
        self.assertTrue(res.json()["busy"])
        self.assertIn("bob", res.json()["error"])
        self.assertEqual(self.start().status_code, 201)
        second = self.start()                                           # a stream holds the slot too
        self.assertEqual(second.status_code, 409)
        self.assertIn("Stream", second.json()["error"])

    def test_a_video_that_cannot_be_opened_ends_with_a_message(self):
        touch(os.path.join(self.folder, "clip.avi"), b"this is not a video")
        self.give_model()
        sid = self.start().json()["id"]
        self.wait_for(lambda: self.client.get(f"/api/stream/{sid}").json()["state"] == "error")
        self.assertIn("could not be opened", self.client.get(f"/api/stream/{sid}").json()["error"])
        self.assertIsNone(self.client.get(self.url).json()["session"])
        self.assertEqual(self.start().status_code, 201)                 # and the slot was released

    def test_a_model_that_fails_ends_with_its_message(self):
        self.video()
        self.give_model()

        def broken(path):
            raise RuntimeError("CUDA error: out of memory")
        self.streams.load_model = broken
        sid = self.start().json()["id"]
        self.wait_for(lambda: self.client.get(f"/api/stream/{sid}").json()["state"] == "error")
        self.assertIn("out of memory", self.client.get(f"/api/stream/{sid}").json()["error"])

    def test_nobody_watching_stops_it(self):
        from utils.web import streaming
        self.video(frames=50, fps=60)
        self.give_model()
        with mock.patch.object(streaming, "IDLE_SECONDS", 0.3):
            sid = self.start().json()["id"]
            session = self.streams.get(sid)                             # not through the API: that would count as watching
            self.wait_for(lambda: not session.alive)
        self.assertEqual(session.state, "ended")
        self.assertIn("Nobody was watching", session.note)
        self.assertIsNone(self.streams.current())

    def test_the_feed_is_motion_jpeg(self):
        from utils.web import streaming
        session = streaming.Session("admin", "pets", "clip.avi", "x", "m", 0.4)
        session.jpeg, session.seq, session.state = b"\xff\xd8JPEGDATA\xff\xd9", 1, "ended"
        self.streams._sessions[session.id] = session                    # an ended stream with its last picture: the feed sends it and closes
        res = self.client.get(f"/api/stream/{session.id}/feed")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers["content-type"], "multipart/x-mixed-replace; boundary=frame")
        self.assertEqual(res.headers["cache-control"], "no-store")
        self.assertTrue(res.content.startswith(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 12\r\n\r\n\xff\xd8"))
        self.assertTrue(res.content.endswith(b"\xff\xd9\r\n"))
        self.assertEqual(self.client.get("/api/stream/0123456789abcdef/feed").status_code, 404)

    def test_a_real_run_makes_real_jpegs(self):
        self.video(frames=4)
        self.give_model()
        sid = self.start().json()["id"]
        session = self.streams.get(sid)
        self.wait_for(lambda: session.jpeg is not None)
        picture = cv2.imdecode(np.frombuffer(session.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(picture.shape[:2], (48, 64))


@unittest.skipIf(cv2 is None, "OpenCV is not installed")
class WideVideoTests(StreamBase):
    def test_wide_pictures_are_shrunk_before_they_are_sent(self):
        self.video(frames=3, size=(1920, 1080))
        self.give_model()
        session = self.streams.get(self.start().json()["id"])
        self.wait_for(lambda: session.jpeg is not None)
        picture = cv2.imdecode(np.frombuffer(session.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(picture.shape[:2], (720, 1280))
        self.assertEqual((session.public()["width"], session.public()["height"]), (1920, 1080))


class ClearCacheTests(StreamBase):
    def fill(self):
        touch(os.path.join(self.folder, "a.avi"), b"a" * 100)
        touch(os.path.join(self.folder, "b.mp4"), b"b" * 50)
        touch(os.path.join(self.folder, "c.mp4.part"), b"p" * 10)
        old = touch(os.path.join(self.tmp, "downloads_old.zip"))
        self.lock = touch(os.path.join(self.tmp, "temp", "training.lock"))
        stale = touch(os.path.join(self.tmp, "temp", "downloads", "old.zip"), b"z" * 30)
        os.utime(stale, (time.time() - 7 * 3600,) * 2)
        self.stale = stale
        self.outside = old

    def test_it_removes_the_videos_and_the_junk(self):
        self.fill()
        info = self.client.get(self.url).json()["cache"]
        self.assertEqual((info["videos"], info["junk"]), (2, 2))
        res = self.client.post("/api/stream/clear-cache", json={}, headers=H)
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(res.json(), {"removed": 4, "bytes": 190, "failed": 0, "kept": []})
        self.assertEqual(os.listdir(self.folder), [])
        self.assertFalse(os.path.exists(self.stale))
        self.assertTrue(os.path.exists(self.lock))                       # not ours to delete
        self.assertTrue(os.path.exists(self.outside))
        self.assertEqual(self.client.get(self.url).json()["cache"]["videos"], 0)

    @unittest.skipIf(cv2 is None, "OpenCV is not installed")
    def test_the_video_that_plays_is_kept(self):
        self.fill()
        self.video(frames=40, fps=30)
        self.give_model()
        self.start()
        res = self.client.post("/api/stream/clear-cache", json={}, headers=H).json()
        self.assertEqual(res["kept"], ["clip.avi"])
        self.assertTrue(os.path.isfile(os.path.join(self.folder, "clip.avi")))
        self.assertFalse(os.path.exists(os.path.join(self.folder, "a.avi")))
        self.assertEqual(self.client.get(self.url).json()["cache"]["videos"], 0)       # what is left is in use


if __name__ == "__main__":
    unittest.main()
