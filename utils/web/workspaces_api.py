"""
The workspace routes of the web API (Add Workspace, delete, jobs). `routes(guard, jobs, uploads, base_dir)` returns
the Starlette routes; `guard(request)` is the sign-in / header check of app.py.

The file system is only touched through utils/workspace_manager.py, utils/dataset_import.py and importing.py, the same
code the desktop window uses. Imports and deletes take one lock: two of them at once could pick the same instance name.
"""
import os
import threading

from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from starlette.routing import Route

from .. import dataset_import as di
from .. import process_guard, workspace_counts
from .. import workspace_manager as wm
from . import importing

change_lock = threading.Lock()


def fail(message, status=400, **extra):
    return JSONResponse({"error": message, **extra}, status_code=status)


async def body_of(request):
    try:
        data = await request.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def routes(guard, jobs, uploads, base_dir):
    def need(request):
        return guard(request)

    # ---------------------------------------------------------------- jobs
    async def job_item(request):
        user, error = need(request)
        if error:
            return error
        job = jobs.get(request.path_params["job_id"])
        if job is None:
            return fail("Unknown job.", 404)
        return JSONResponse(job.public())

    async def job_cancel(request):
        user, error = need(request)
        if error:
            return error
        job = jobs.get(request.path_params["job_id"])
        if job is None:
            return fail("Unknown job.", 404)
        job.cancel()
        return JSONResponse(job.public())

    # ---------------------------------------------------------------- folders and uploads
    async def folders(request):
        user, error = need(request)
        if error:
            return error
        try:
            return JSONResponse(await run_in_threadpool(importing.list_folder, request.query_params.get("path", ""),
                                                        request.query_params.get("ext", "")))
        except ValueError as exc:
            return fail(str(exc))

    async def upload_new(request):
        user, error = need(request)
        if error:
            return error
        return JSONResponse({"id": await run_in_threadpool(uploads.create, user["name"])}, status_code=201)

    async def upload_file(request):
        user, error = need(request)
        if error:
            return error
        try:
            path = uploads.target(request.path_params["upload_id"], user["name"], request.query_params.get("path", ""))
        except ValueError as exc:
            return fail(str(exc))
        room = importing.free_bytes(uploads.root)
        if int(request.headers.get("content-length") or 0) > room:
            return fail("Not enough free disk space on the server.", 507)
        part, received = path + ".part", 0
        try:
            with open(part, "wb") as out:
                async for chunk in request.stream():
                    received += len(chunk)
                    if received > room:
                        raise ValueError("Not enough free disk space on the server.")
                    await run_in_threadpool(out.write, chunk)
            os.replace(part, path)
        except ValueError as exc:
            _remove(part)
            return fail(str(exc), 507)
        except Exception:
            _remove(part)
            raise
        return JSONResponse({"ok": True, "bytes": received})

    async def upload_remove(request):
        user, error = need(request)
        if error:
            return error
        try:
            uploads.check(request.path_params["upload_id"], user["name"])
        except ValueError as exc:
            return fail(str(exc), 404)
        await run_in_threadpool(uploads.remove, request.path_params["upload_id"])
        return JSONResponse({"ok": True})

    def source_folder(data, user):
        """The folder to import from: an upload's staging folder (zip files unpacked) or a folder on the server."""
        if data.get("upload_id"):
            folder = uploads.check(str(data["upload_id"]), user["name"])
            uploads.unpack_zips(folder)
            return folder
        folder = str(data.get("folder") or "").strip()
        if not folder:
            raise ValueError("Choose a dataset folder first.")
        return os.path.abspath(folder)

    async def scan(request):
        user, error = need(request)
        if error:
            return error
        data = await body_of(request)
        try:
            folder = await run_in_threadpool(source_folder, data, user)
            return JSONResponse(await run_in_threadpool(importing.summarize, folder, data.get("workspace") or None))
        except ValueError as exc:
            return fail(str(exc))

    # ---------------------------------------------------------------- import
    def checked_import(data, user):
        """Validate everything the dialog sent; returns the arguments of import_dataset."""
        existing = wm.list_workspaces()
        if data.get("workspace"):
            name = str(data["workspace"])
            if name not in existing:
                raise ValueError(f"The workspace '{name}' does not exist.")
        else:
            name = wm.sanitize_workspace_name(data.get("name"))
        folder = source_folder(data, user)
        summary = importing.summarize(folder, name)
        if summary["error"]:
            raise ValueError(summary["error"])
        prefix = di.sanitize_filename_prefix(data.get("prefix"))
        skip = di.sanitize_skip_frames(data.get("skip_frames", di.DEFAULT_SKIP_FRAMES)) if summary["videos"] \
            else di.DEFAULT_SKIP_FRAMES
        classes = None
        raw = [str(c).strip() for c in (data.get("classes") or []) if str(c).strip()]
        if raw and not wm.existing_classes_for_workspace(name):
            classes = wm.parse_classes_input("{" + ", ".join(raw) + "}")
        return name, folder, prefix, skip, classes

    async def import_start(request):
        user, error = need(request)
        if error:
            return error
        data = await body_of(request)
        try:
            name, folder, prefix, skip, classes = await run_in_threadpool(checked_import, data, user)
        except ValueError as exc:
            return fail(str(exc))
        upload_id = str(data.get("upload_id") or "")

        def work(job):
            try:
                with change_lock:
                    job.progress(0, 0, "Starting the import ...")
                    return di.import_dataset(folder, name, prefix=prefix, classes=classes, skip_frames=skip,
                                             progress_cb=lambda done, total, text: job.progress(done, total, text))
            finally:
                if upload_id:
                    uploads.remove(upload_id)

        job = jobs.start("import", f'Adding to "{name}"', user["name"], work)
        return JSONResponse(job.public(), status_code=202)

    # ---------------------------------------------------------------- delete
    def known(name, instance=None):
        groups = wm.list_workspaces()
        if name not in groups or (instance is not None and instance not in groups[name]):
            raise ValueError("That workspace does not exist.")

    def training_in(name):
        info = process_guard.running_training(base_dir)
        return bool(info and info.get("workspace") == name)

    async def stats(request):
        user, error = need(request)
        if error:
            return error
        name = request.path_params["name"]
        try:
            known(name)
        except ValueError as exc:
            return fail(str(exc), 404)
        return JSONResponse(await run_in_threadpool(wm.workspace_stats, name))

    async def delete_workspace(request):
        user, error = need(request)
        if error:
            return error
        name = request.path_params["name"]
        try:
            known(name)
        except ValueError as exc:
            return fail(str(exc), 404)
        if training_in(name):
            return fail("A training is running in this workspace. Stop it first.", 409)

        def work(job):
            with change_lock:
                wm.delete_workspace(name, progress_cb=lambda done, total, text: job.progress(done, total, text))
                workspace_counts.forget(name)
            return {"deleted": name}

        return JSONResponse(jobs.start("delete", f'Deleting "{name}"', user["name"], work).public(), status_code=202)

    async def delete_instance(request):
        user, error = need(request)
        if error:
            return error
        name, instance = request.path_params["name"], request.path_params["instance"]
        try:
            known(name, instance)
        except ValueError as exc:
            return fail(str(exc), 404)
        if training_in(name):
            return fail("A training is running in this workspace. Stop it first.", 409)

        def work(job):
            with change_lock:
                wm.delete_instance(instance, progress_cb=lambda done, total, text: job.progress(done, total, text))
            return {"deleted": instance}

        return JSONResponse(jobs.start("delete", f'Deleting "{instance}"', user["name"], work).public(), status_code=202)

    return [
        Route("/api/jobs/{job_id}", job_item),
        Route("/api/jobs/{job_id}/cancel", job_cancel, methods=["POST"]),
        Route("/api/folders", folders),
        Route("/api/uploads", upload_new, methods=["POST"]),
        Route("/api/uploads/{upload_id}/file", upload_file, methods=["PUT"]),
        Route("/api/uploads/{upload_id}", upload_remove, methods=["DELETE"]),
        Route("/api/scan", scan, methods=["POST"]),
        Route("/api/import", import_start, methods=["POST"]),
        Route("/api/workspaces/{name}/stats", stats),
        Route("/api/workspaces/{name}", delete_workspace, methods=["DELETE"]),
        Route("/api/workspaces/{name}/instances/{instance}", delete_instance, methods=["DELETE"]),
    ]


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass
