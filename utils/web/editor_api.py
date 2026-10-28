"""
The annotation editor routes: /api/ws/<dataset>/... (pictures, annotations, classes).

Every call builds a `Workspace` (editing.py) for the dataset in the URL; a name that is not a dataset of this
installation is a 404. `guard(request)` is the sign-in / header check of app.py.
"""
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from . import assistant
from .editing import Conflict, EditError, Workspace


def fail(message, status=400, **extra):
    return JSONResponse({"error": message, **extra}, status_code=status)


async def body_of(request):
    try:
        data = await request.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def routes(guard, base_dir):
    async def run(request, work, *, not_found=404):
        """Check the call, build the workspace and run work(workspace) on a thread. EditError -> a JSON error."""
        user, error = guard(request)
        if error:
            return error
        try:
            workspace = await run_in_threadpool(Workspace, base_dir, request.path_params["instance"])
            result = await run_in_threadpool(work, workspace)
        except Conflict as exc:
            return fail(str(exc), 409, conflict=True)
        except EditError as exc:
            missing = "does not exist" in str(exc)
            return fail(str(exc), not_found if missing else 400)
        return result if isinstance(result, (JSONResponse, FileResponse)) else JSONResponse(result)

    async def opened(request):
        return await run(request, lambda ws: ws.open_info())

    async def image(request):
        name = request.path_params["name"]
        return await run(request, lambda ws: FileResponse(ws.path(name), headers={"Cache-Control": "private, no-cache"}))

    async def annotations(request):
        name = request.path_params["name"]
        if request.method == "GET":
            return await run(request, lambda ws: ws.load(name))
        data = await body_of(request)
        reviewed = data.get("reviewed", True) is not False            # annotations the assistant made and nobody looked at
                                                                       # are not something SAM 2 should learn from
        return await run(request, lambda ws: ws.save(
            name, data.get("bboxes", []), data.get("polygons", []), data.get("masks"), data.get("version"),
            bool(data.get("force")),
            on_saved=(lambda path, boxes, polys: assistant.sam_store(ws.name).remember(path, boxes, polys))
            if reviewed else None))

    async def delete_image(request):
        name = request.path_params["name"]

        def work(ws):
            path = ws.path(name)
            ws.delete_image(name)
            assistant.forget_references(ws.name, path)
            return {"ok": True, "images": ws.images(), "annotated": ws.annotated()}
        return await run(request, work)

    async def classes(request):
        if request.method == "GET":
            return await run(request, lambda ws: {"classes": ws.class_info()})
        data = await body_of(request)

        def work(ws):
            message = ws.add_class(data.get("name"))
            return {"message": message, "classes": ws.class_info()}
        return await run(request, work)

    async def class_item(request):
        name = request.path_params["class_name"]

        def work(ws):
            message = ws.delete_class(name)
            return {"message": message, "classes": ws.class_info(), "annotated": ws.annotated()}
        return await run(request, work)

    base = "/api/ws/{instance}"
    return [
        Route(base, opened),
        Route(base + "/image/{name}", image),
        Route(base + "/annotations/{name}", annotations, methods=["GET", "PUT"]),
        Route(base + "/images/{name}", delete_image, methods=["DELETE"]),
        Route(base + "/classes", classes, methods=["GET", "POST"]),
        Route(base + "/classes/{class_name}", class_item, methods=["DELETE"]),
    ]
