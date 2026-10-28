"""
The Label Assistant routes: /api/ws/<dataset>/assistant (settings), /infer and /autoannotate (jobs).

Infer and Auto-annotate answer 202 with a job (poll /api/jobs/<id>, cancel with /api/jobs/<id>/cancel). Before a model
is downloaded they answer 409 with `needs_confirm` {title, message}; the page asks and sends the call again with
`confirm_download: true`. If another model job is running they answer 409 with `busy`.
"""
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from starlette.routing import Route

from ..assistant_providers import AssistantError
from . import assistant
from .editing import EditError, Workspace


def fail(message, status=400, **extra):
    return JSONResponse({"error": message, **extra}, status_code=status)


async def body_of(request):
    try:
        data = await request.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def routes(guard, jobs, base_dir):
    async def call(request, work):
        """Check the call, build the workspace and run work(workspace, user) on a thread."""
        user, error = guard(request)
        if error:
            return error
        try:
            ws = await run_in_threadpool(Workspace, base_dir, request.path_params["instance"])
            result = await run_in_threadpool(work, ws, user)
        except assistant.Busy as exc:
            return fail(str(exc), 409, busy=True)
        except (EditError, AssistantError) as exc:
            return fail(str(exc), 404 if "does not exist" in str(exc) else 400)
        return result if isinstance(result, JSONResponse) else JSONResponse(result)

    def start(ws, user, kind, title, mode_check, fn, confirm):
        """Take the model, ask before a download, and run fn(ctx, job) as a job."""
        ctx = assistant.Ctx(ws)
        pending = assistant.needs_download(ctx)
        if pending and not confirm:
            return JSONResponse({"error": pending["message"], "needs_confirm": pending}, status_code=409)
        assistant.acquire(user["name"], title)
        try:
            def work(job):
                try:
                    return fn(ctx, job)
                finally:
                    assistant.release()
            job = jobs.start(kind, title, user["name"], work)
        except Exception:
            assistant.release()
            raise
        return JSONResponse(job.public(), status_code=202)

    async def settings(request):
        if request.method == "GET":
            return await call(request, lambda ws, user: assistant.settings_payload(ws))
        data = await body_of(request)

        def work(ws, user):
            warnings = assistant.save_settings(ws, data)
            return {"warnings": warnings, "settings": assistant.settings_payload(ws)}
        return await call(request, work)

    async def inspect(request):
        data = await body_of(request)
        return await call(request, lambda ws, user: assistant.inspect_file(ws, data.get("kind"), data.get("path")))

    async def references(request):
        data = await body_of(request)

        def work(ws, user):
            added = 0
            if data.get("action") == "load":
                added = assistant.load_labeled(ws)
            elif data.get("action") == "forget":
                assistant.forget_references(ws.name)
            else:
                raise EditError("Unknown action.")
            return {"added": added, "references": len(assistant.sam_store(ws.name))}
        return await call(request, work)

    async def infer(request):
        data = await body_of(request)
        name = str(data.get("name") or "")

        def work(ws, user):
            ws.path(name)                                           # a picture of this dataset, or an error now

            def run(ctx, job):
                return assistant.infer_one(ctx, name, data.get("bboxes") or [], data.get("polygons") or [],
                                           bool(data.get("polygon")), lambda text: job.progress(text=text))
            return start(ws, user, "assist", "Label Assistant", None, run, bool(data.get("confirm_download")))
        return await call(request, work)

    async def autoannotate(request):
        data = await body_of(request)

        def work(ws, user):
            only = data.get("only_unlabeled")
            only = True if only is None else bool(only)
            return start(ws, user, "assist", "Auto-annotate all", None,
                         lambda ctx, job: assistant.autoannotate(ctx, job, only, bool(data.get("polygon"))),
                         bool(data.get("confirm_download")))
        return await call(request, work)

    base = "/api/ws/{instance}"
    return [
        Route(base + "/assistant", settings, methods=["GET", "PUT"]),
        Route(base + "/assistant/inspect", inspect, methods=["POST"]),
        Route(base + "/assistant/references", references, methods=["POST"]),
        Route(base + "/infer", infer, methods=["POST"]),
        Route(base + "/autoannotate", autoannotate, methods=["POST"]),
    ]
