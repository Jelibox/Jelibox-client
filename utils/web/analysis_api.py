"""
Analyze Dataset routes. `routes(guard, jobs, base_dir)` returns the Starlette routes.

POST /api/workspaces/<name>/analysis       reads the XML files (a job); its result holds a token and the first view
POST /api/analysis/<token>/view            the numbers for another size / mode / limit (calculated, nothing on disk)
POST /api/analysis/<token>/remove-small    deletes the flagged objects from the dataset (a job), then re-reads it
"""
import os

from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from starlette.routing import Route

from .. import app_settings, process_guard, workspace_config
from .. import dataset_analysis as da
from .. import workspace_manager as wm
from . import analysis
from .workspaces_api import body_of, change_lock, fail


def routes(guard, jobs, base_dir):
    cache = analysis.Cache()

    def read(name, progress=None):
        folders = {i: wm.instance_path(i) for i in wm.list_workspaces().get(name, [])}
        return da.analyze(name, folders, os.path.join(base_dir, "vocdataset", name), progress)

    def saved_limit():
        try:
            return int(app_settings.get("small_limit", da.SMALL_LIMIT))
        except (TypeError, ValueError):
            return da.SMALL_LIMIT

    def view_of(body):
        size, mode, limit = analysis.check_view(body, saved_limit())
        if limit != saved_limit():
            app_settings.set("small_limit", limit)                     # remembered, like the desktop window
        return size, mode, limit

    async def start(request):
        user, error = guard(request)
        if error:
            return error
        name = request.path_params["name"]
        if name not in wm.list_workspaces():
            return fail("That workspace does not exist.", 404)
        body = await body_of(request)
        try:
            size, mode, limit = view_of(body)
        except ValueError as exc:
            return fail(str(exc))

        def work(job):
            data = read(name, lambda done, total, text: job.progress(done, total, text))
            job.progress(None, None, "Calculating ...")
            return {"token": cache.add(name, user["name"], data), "view": analysis.payload(data, size, mode, limit)}

        return JSONResponse(jobs.start("analyze", f'Analyzing "{name}"', user["name"], work).public(), status_code=202)

    async def view(request):
        user, error = guard(request)
        if error:
            return error
        kept = cache.get(request.path_params["token"])
        if kept is None:
            return fail("This analysis is no longer open. Analyze the workspace again.", 404)
        try:
            size, mode, limit = view_of(await body_of(request))
        except ValueError as exc:
            return fail(str(exc))
        return JSONResponse(await run_in_threadpool(analysis.payload, kept.data, size, mode, limit))

    async def remove_small(request):
        user, error = guard(request)
        if error:
            return error
        token = request.path_params["token"]
        kept = cache.get(token)
        if kept is None:
            return fail("This analysis is no longer open. Analyze the workspace again.", 404)
        try:
            size, mode, limit = view_of(await body_of(request))
        except ValueError as exc:
            return fail(str(exc))
        name = kept.workspace
        info = process_guard.running_training(base_dir)
        if info and info.get("workspace") == name:
            return fail("A training is running in this workspace. Stop it first.", 409)
        if da.small_summary(da.view(kept.data, size, mode) if size else da.view(kept.data), limit) is None:
            return fail("No object is under the limit.", 409)

        def work(job):
            with change_lock:
                result = da.remove_small(kept.data, os.path.join(base_dir, "YOLOdataset", name, "labels"),
                                         workspace_config.get_classes(name) or [], size, mode, limit,
                                         lambda done, total, text: job.progress(done, total, text))
                job.progress(0, 0, "Reading the annotations again ...")
                data = read(name)
            cache.replace(token, data)
            return {**result, "view": analysis.payload(data, size, mode, limit)}

        return JSONResponse(jobs.start("remove-small", f'Removing small objects from "{name}"', user["name"],
                                       work).public(), status_code=202)

    return [
        Route("/api/workspaces/{name}/analysis", start, methods=["POST"]),
        Route("/api/analysis/{token}/view", view, methods=["POST"]),
        Route("/api/analysis/{token}/remove-small", remove_small, methods=["POST"]),
    ]
