"""
The Stream routes of the web API.

  /api/ws/<dataset>/stream            GET what the window shows (model, videos on the server, cache size, a running stream)
  /api/ws/<dataset>/stream/start      POST {video, conf} start the model on a video
  /api/stream/upload?name=            PUT the bytes of a video (kept in temp/stream)
  /api/stream/youtube                 POST {url} download a YouTube video into temp/stream (a job)
  /api/stream/clear-cache             POST delete the videos and the junk in temp/ (what it deletes: utils/stream_support.py)
  /api/stream/<id>                    GET the numbers; POST {conf} changes the confidence while it runs
  /api/stream/<id>/stop               POST
  /api/stream/<id>/feed               GET the annotated pictures as motion JPEG (an <img> shows it)
"""
import asyncio
import os

from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from .. import stream_support as ss
from . import assistant, importing
from .editing import EditError, Workspace
from .streaming import CONF_RANGE


def fail(message, status=400, **extra):
    return JSONResponse({"error": message, **extra}, status_code=status)


async def body_of(request):
    try:
        data = await request.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


def routes(guard, jobs, base_dir, streams):
    async def call(request, work):
        user, error = guard(request)
        if error:
            return error
        try:
            ws = await run_in_threadpool(Workspace, base_dir, request.path_params["instance"])
            result = await run_in_threadpool(work, ws, user)
        except assistant.Busy as exc:
            return fail(str(exc), 409, busy=True)
        except EditError as exc:
            return fail(str(exc), 404 if "does not exist" in str(exc) else 400)
        return result if isinstance(result, JSONResponse) else JSONResponse(result)

    async def info(request):
        return await call(request, lambda ws, user: streams.info(ws))

    async def start(request):
        data = await body_of(request)

        def work(ws, user):
            session = streams.start(user["name"], ws, data.get("video"), data.get("conf", 0.4))
            return JSONResponse(session.public(), status_code=201)
        return await call(request, work)

    # ---------------------------------------------------------------- videos
    async def upload(request):
        user, error = guard(request)
        if error:
            return error
        os.makedirs(streams.folder, exist_ok=True)
        try:
            name = ss.clean_video_name(request.query_params.get("name"), streams.folder)
        except ValueError as exc:
            return fail(str(exc))
        room = importing.free_bytes(streams.folder)
        if int(request.headers.get("content-length") or 0) > room:
            return fail("Not enough free disk space on the server.", 507)
        path = os.path.join(streams.folder, name)
        part, received = path + ".part", 0
        with streams.transfer():
            try:
                with open(part, "wb") as out:
                    async for chunk in request.stream():
                        received += len(chunk)
                        if received > room:
                            raise ValueError("Not enough free disk space on the server.")
                        await run_in_threadpool(out.write, chunk)
                if not received:
                    raise ValueError("The file is empty.")
                os.replace(part, path)
            except ValueError as exc:
                _remove(part)
                return fail(str(exc), 507 if "disk space" in str(exc) else 400)
            except Exception:
                _remove(part)
                raise
        return JSONResponse({"name": name, "bytes": received}, status_code=201)

    async def youtube(request):
        user, error = guard(request)
        if error:
            return error
        try:
            url = ss.check_youtube_url((await body_of(request)).get("url"))
        except ValueError as exc:
            return fail(str(exc))
        job = jobs.start("stream", "Download from YouTube", user["name"], lambda job: streams.download(job, url))
        return JSONResponse(job.public(), status_code=202)

    async def clear_cache(request):
        user, error = guard(request)
        if error:
            return error
        try:
            return JSONResponse(await run_in_threadpool(streams.clear_cache))
        except EditError as exc:
            return fail(str(exc), 409)

    # ---------------------------------------------------------------- a running stream
    def session_of(request, user):
        session = streams.get(request.path_params["sid"])
        session.touch()
        return session

    async def item(request):
        user, error = guard(request)
        if error:
            return error
        try:
            session = session_of(request, user)
        except EditError as exc:
            return fail(str(exc), 404)
        if request.method == "POST":
            try:
                conf = round(float((await body_of(request)).get("conf")), 2)
            except (TypeError, ValueError):
                return fail("The confidence must be a number.")
            if not CONF_RANGE[0] <= conf <= CONF_RANGE[1]:
                return fail(f"The confidence must be between {CONF_RANGE[0]} and {CONF_RANGE[1]}.")
            session.conf = conf
        return JSONResponse(session.public())

    async def stop(request):
        user, error = guard(request)
        if error:
            return error
        try:
            session = session_of(request, user)
        except EditError as exc:
            return fail(str(exc), 404)
        session.stop()
        return JSONResponse(session.public())

    async def feed(request):
        user, error = guard(request)
        if error:
            return error
        try:
            session = session_of(request, user)
        except EditError as exc:
            return fail(str(exc), 404)

        async def frames():
            last = -1
            while True:
                data, seq = session.jpeg, session.seq
                if data and seq != last:
                    last = seq
                    session.touch()
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(data)) + data + b"\r\n"
                elif not session.alive:
                    return
                else:
                    await asyncio.sleep(0.02)
        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store"})

    base = "/api/ws/{instance}/stream"
    return [
        Route(base, info),
        Route(base + "/start", start, methods=["POST"]),
        Route("/api/stream/upload", upload, methods=["PUT"]),
        Route("/api/stream/youtube", youtube, methods=["POST"]),
        Route("/api/stream/clear-cache", clear_cache, methods=["POST"]),
        Route("/api/stream/{sid}", item, methods=["GET", "POST"]),
        Route("/api/stream/{sid}/stop", stop, methods=["POST"]),
        Route("/api/stream/{sid}/feed", feed),
    ]
