"""
Long work (importing a dataset, deleting a workspace, later auto-annotate and export) runs as a job on its own thread.
A request only starts it and answers at once; the page asks `GET /api/jobs/<id>` every half second for the progress.

A job function is called as fn(job): it reports with job.progress(done, total, text), looks at job.cancelled() between
steps, and returns the result (anything JSON-friendly). An exception ends the job with its message.
"""
import secrets
import threading
import time

KEEP_SECONDS = 3600                 # a finished job can still be read for an hour


class Job:
    def __init__(self, kind, title, user):
        self.id = secrets.token_hex(8)
        self.kind, self.title, self.user = kind, title, user
        self.state = "running"      # running | done | error | cancelled
        self.done = self.total = 0
        self.text = "Starting ..."
        self.result = None
        self.error = None
        self.started = time.time()
        self.finished = None
        self._cancel = threading.Event()

    def progress(self, done=None, total=None, text=None):
        if done is not None:
            self.done = done
        if total is not None:
            self.total = total
        if text is not None:
            self.text = text

    def cancel(self):
        self._cancel.set()

    def cancelled(self):
        return self._cancel.is_set()

    def public(self):
        return {"id": self.id, "kind": self.kind, "title": self.title, "state": self.state, "done": self.done,
                "total": self.total, "text": self.text, "result": self.result, "error": self.error,
                "seconds": round((self.finished or time.time()) - self.started, 1)}


class JobManager:
    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()

    def start(self, kind, title, user, fn):
        job = Job(kind, title, user)
        with self._lock:
            self._prune()
            self._jobs[job.id] = job

        def run():
            try:
                job.result = fn(job)
                job.state = "cancelled" if job.cancelled() else "done"
            except Exception as exc:            # the message goes to the page; the type does not matter there
                job.error = str(exc) or exc.__class__.__name__
                job.state = "error"
            job.finished = time.time()

        threading.Thread(target=run, name=f"job-{kind}", daemon=True).start()
        return job

    def get(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def running(self, kind=None):
        with self._lock:
            return [j for j in self._jobs.values() if j.state == "running" and (kind is None or j.kind == kind)]

    def _prune(self):
        now = time.time()
        for job_id in [i for i, j in self._jobs.items() if j.finished and now - j.finished > KEEP_SECONDS]:
            del self._jobs[job_id]
