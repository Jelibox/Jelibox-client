"""
Export and training routes of the web API.

  /api/ws/<dataset>/export/dataset   GET info, POST start (a job; the result is a download, nothing stays saved on the server)
  /api/ws/<dataset>/export/model     GET info, POST start (a job)
  /api/ws/<dataset>/augment/...      the augmentation picker: catalog, tile previews, grid thumbnails, preview picture
  /api/downloads/<token>             a finished export as a file (a folder is zipped first)
  /api/ws/<dataset>/train            GET what the setup shows, POST start
  /api/ws/<dataset>/train/recount    count the pictures again (a job)
  /api/train                         the training that runs now, or the last one; .../finish-early .../stop .../dismiss
"""
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from .. import app_settings, workspace_counts
from . import exporting
from .editing import EditError, Workspace
from .training import Busy, NeedsConfirm


def fail(message, status=400, **extra):
    return JSONResponse({"error": message, **extra}, status_code=status)


async def body_of(request):
    try:
        data = await request.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def routes(guard, jobs, base_dir, downloads, trainer):
    async def call(request, work):
        user, error = guard(request)
        if error:
            return error
        try:
            ws = await run_in_threadpool(Workspace, base_dir, request.path_params["instance"])
            result = await run_in_threadpool(work, ws, user)
        except NeedsConfirm as exc:
            return fail(str(exc), 409, needs_confirm={"title": exc.title, "message": str(exc)})
        except Busy as exc:
            return fail(str(exc), 409, busy=True)
        except EditError as exc:
            return fail(str(exc), 404 if "does not exist" in str(exc) else 400)
        return result if isinstance(result, (JSONResponse, FileResponse)) else JSONResponse(result)

    # ---------------------------------------------------------------- Export Dataset / Model
    async def dataset(request):
        if request.method == "GET":
            return await call(request, lambda ws, user: exporting.dataset_info(ws))
        data = await body_of(request)

        def work(ws, user):
            parsed = exporting.check_dataset_request(ws, data)
            job = jobs.start("export", "Export Dataset", user["name"],
                             lambda job: exporting.run_dataset_export(ws, job, data, parsed, downloads, user["name"]))
            return JSONResponse(job.public(), status_code=202)
        return await call(request, work)

    async def model(request):
        if request.method == "GET":
            return await call(request, lambda ws, user: exporting.model_info(ws))
        data = await body_of(request)

        def work(ws, user):
            parsed = exporting.check_model_request(ws, data)
            job = jobs.start("export", "Export Model", user["name"],
                             lambda job: exporting.run_model_export(ws, job, data, parsed, downloads, user["name"]))
            return JSONResponse(job.public(), status_code=202)
        return await call(request, work)

    # ---------------------------------------------------------------- the augmentation picker
    async def catalog(request):
        training = request.query_params.get("training") in ("1", "true")
        return await call(request, lambda ws, user: exporting.catalog(training))

    async def previews(request):
        data = await body_of(request)
        return await call(request, lambda ws, user: exporting.previews(ws, str(data.get("tile") or ""), data.get("ops")))

    async def thumbnails(request):
        data = await body_of(request)
        return await call(request, lambda ws, user: exporting.thumbnails(ws, data.get("ops")))

    async def sample(request):
        if request.method == "GET":
            return await call(request, lambda ws, user: exporting.sample_info(ws))
        data = await body_of(request)
        return await call(request, lambda ws, user: exporting.choose_sample(ws, str(data.get("name") or "")))

    async def download(request):
        user, error = guard(request)
        if error:
            return error
        try:
            path, name = await run_in_threadpool(downloads.prepare, request.path_params["token"], user["name"])
        except EditError as exc:
            return fail(str(exc), 404)
        return FileResponse(path, filename=name)

    # ---------------------------------------------------------------- training
    async def train(request):
        if request.method == "GET":
            polygon = request.query_params.get("polygon") in ("1", "true")
            return await call(request, lambda ws, user: {**trainer.info(ws, polygon), **trainer.state()})
        data = await body_of(request)
        return await call(request, lambda ws, user: trainer.start(ws, data, user["name"], bool(data.get("confirm"))))

    async def recount(request):
        def work(ws, user):
            job = jobs.start("count", "Counting the pictures", user["name"],
                             lambda job: workspace_counts.recount(ws.name))
            return JSONResponse(job.public(), status_code=202)
        return await call(request, work)

    async def train_state(request):
        user, error = guard(request)
        if error:
            return error
        try:
            rows = int(request.query_params.get("rows", 0))
        except ValueError:
            rows = 0
        return JSONResponse(trainer.state(rows))

    def control(action):
        async def handler(request):
            user, error = guard(request)
            if error:
                return error
            try:
                result = await run_in_threadpool(getattr(trainer, action))
            except EditError as exc:
                return fail(str(exc))
            return JSONResponse(result if isinstance(result, dict) else {"ok": True})
        return handler

    base = "/api/ws/{instance}"
    return [
        Route(base + "/export/dataset", dataset, methods=["GET", "POST"]),
        Route(base + "/export/model", model, methods=["GET", "POST"]),
        Route(base + "/augment/catalog", catalog),
        Route(base + "/augment/previews", previews, methods=["POST"]),
        Route(base + "/augment/thumbnails", thumbnails, methods=["POST"]),
        Route(base + "/augment/sample", sample, methods=["GET", "PUT"]),
        Route(base + "/train", train, methods=["GET", "POST"]),
        Route(base + "/train/recount", recount, methods=["POST"]),
        Route("/api/downloads/{token}", download),
        Route("/api/train", train_state),
        Route("/api/train/finish-early", control("finish_early"), methods=["POST"]),
        Route("/api/train/stop", control("stop"), methods=["POST"]),
        Route("/api/train/dismiss", control("dismiss"), methods=["POST"]),
    ]
