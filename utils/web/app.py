"""
The Jelibox web application (Starlette). The API lives under /api and always answers JSON; the page is static.

Every /api call except login needs the session cookie. Calls that change something also need the header
`X-Jelibox: 1` (a form on another site cannot add it) and, when the browser sends an Origin, that Origin must be this
server. Blocking work (reading folders, counting files) runs in a thread so one slow request never stalls the rest.
"""
import os

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware

from .. import assistant_modes, workspace_counts, workspace_manager
from . import analysis_api, assistant_api, editor_api, export_api, sysmon, workspaces_api
from .exporting import Downloads
from .training import TrainingManager
from .importing import Uploads
from .jobs import JobManager
from .users import LoginThrottle, UserError, UserStore

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
COOKIE = "jelibox_session"
SAFE_METHODS = ("GET", "HEAD", "OPTIONS")


class SecurityHeaders(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
        return response


def fail(message, status=400, **extra):
    return JSONResponse({"error": message, **extra}, status_code=status)


async def body_of(request):
    try:
        data = await request.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def create_app(users_path=None, base_dir=BASE_DIR):
    store = UserStore(users_path or os.path.join(base_dir, "configs", "_users.json"))
    throttle = LoginThrottle()
    jobs = JobManager()
    uploads = Uploads(os.path.join(base_dir, "temp", "uploads"))
    downloads = Downloads(os.path.join(base_dir, "temp", "downloads"))
    trainer = TrainingManager(base_dir)

    # ---------------------------------------------------------------- guards
    def user_of(request):
        return store.user_for_token(request.cookies.get(COOKIE))

    def guard(request, admin=False):
        """(user, None) when the call may go on, else (None, the error response to send)."""
        if request.method not in SAFE_METHODS:
            if request.headers.get("x-jelibox") != "1":
                return None, fail("Missing X-Jelibox header.", 403)
            origin = request.headers.get("origin")
            if origin and origin.split("://", 1)[-1] != request.headers.get("host"):
                return None, fail("Wrong origin.", 403)
        user = user_of(request)
        if user is None:
            return None, fail("Sign in first.", 401)
        if user["must_change"]:
            return None, fail("Choose a new password first.", 403, must_change=True)
        if admin and not user["admin"]:
            return None, fail("Only an administrator can do this.", 403)
        return user, None

    def set_session(response, name):
        response.set_cookie(COOKIE, store.make_token(name), max_age=7 * 24 * 3600, httponly=True, samesite="strict",
                            path="/")

    # ---------------------------------------------------------------- session
    async def login(request):
        data = await body_of(request)
        name, password = str(data.get("username", "")).strip(), str(data.get("password", ""))
        key = ((request.client.host if request.client else "?"), name.lower())
        wait = throttle.wait(key)
        if wait:
            return fail(f"Too many wrong passwords. Try again in {wait} s.", 429, retry_after=wait)
        user = await run_in_threadpool(store.verify, name, password)
        if user is None:
            throttle.failed(key)
            return fail("Wrong user name or password.", 401)
        throttle.succeeded(key)
        response = JSONResponse({"user": user})
        set_session(response, user["name"])
        return response

    async def logout(request):
        response = JSONResponse({"ok": True})
        response.delete_cookie(COOKIE, path="/")
        return response

    async def me(request):
        user = user_of(request)
        return JSONResponse({"user": user})

    async def change_password(request):
        user = user_of(request)
        if user is None:
            return fail("Sign in first.", 401)
        if request.headers.get("x-jelibox") != "1":
            return fail("Missing X-Jelibox header.", 403)
        data = await body_of(request)
        if await run_in_threadpool(store.verify, user["name"], str(data.get("old", ""))) is None:
            return fail("The current password is wrong.", 400)
        try:
            await run_in_threadpool(store.set_password, user["name"], str(data.get("new", "")))
        except UserError as exc:
            return fail(str(exc))
        response = JSONResponse({"user": store.get(user["name"])})
        set_session(response, user["name"])                       # the old cookie died with the old epoch
        return response

    # ---------------------------------------------------------------- users (admin)
    async def users(request):
        user, error = guard(request, admin=True)
        if error:
            return error
        if request.method == "GET":
            return JSONResponse({"users": store.users()})
        data = await body_of(request)
        try:
            created = await run_in_threadpool(store.add, str(data.get("username", "")), str(data.get("password", "")),
                                              bool(data.get("admin")))
        except UserError as exc:
            return fail(str(exc))
        return JSONResponse({"user": created}, status_code=201)

    async def user_item(request):
        user, error = guard(request, admin=True)
        if error:
            return error
        name = request.path_params["name"]
        try:
            if request.method == "DELETE":
                if name.lower() == user["name"].lower():
                    return fail("You cannot delete your own account.")
                await run_in_threadpool(store.delete, name)
            else:                                                  # reset: the person must pick a new one at next login
                data = await body_of(request)
                await run_in_threadpool(store.set_password, name, str(data.get("password", "")), True)
        except UserError as exc:
            return fail(str(exc))
        return JSONResponse({"ok": True})

    # ---------------------------------------------------------------- workspaces
    def workspace_rows():
        rows = []
        for name, instances in workspace_manager.list_workspaces().items():
            stats = workspace_counts.get_or_empty(name)
            rows.append({
                "name": name, "images": stats["images"], "counted": stats["counted"],
                "instances": [{"name": inst, "images": workspace_counts.instance_images(name, inst)}
                              for inst in instances],
            })
        return rows

    async def workspaces(request):
        user, error = guard(request)
        if error:
            return error
        return JSONResponse({"workspaces": await run_in_threadpool(workspace_rows)})

    async def recount(request):
        user, error = guard(request)
        if error:
            return error
        await run_in_threadpool(workspace_counts.recount_all)
        return JSONResponse({"workspaces": await run_in_threadpool(workspace_rows)})

    # ---------------------------------------------------------------- modes
    def modes_payload():
        return {"mode": assistant_modes.get_mode(),
                "modes": [{"id": m[0], "title": m[1], "summary": m[2], "needs": m[3]} for m in assistant_modes.MODES]}

    async def modes(request):
        user, error = guard(request)
        if error:
            return error
        if request.method == "PUT":
            data = await body_of(request)
            try:
                assistant_modes.set_mode(str(data.get("mode", "")))
            except ValueError as exc:
                return fail(str(exc))
        return JSONResponse(modes_payload())

    # ---------------------------------------------------------------- page
    async def index(request):
        return FileResponse(os.path.join(STATIC_DIR, "index.html"), headers={"Cache-Control": "no-store"})

    async def not_found(request):
        return fail("Not found.", 404)

    async def system(request):
        user, error = guard(request)
        if error:
            return error
        return JSONResponse(await run_in_threadpool(sysmon.snapshot, base_dir))

    async def health(request):
        return JSONResponse({"ok": True, "name": "jelibox"})

    routes = [
        Route("/", index),
        Route("/api/health", health),
        Route("/api/system", system),
        Route("/api/login", login, methods=["POST"]),
        Route("/api/logout", logout, methods=["POST"]),
        Route("/api/me", me),
        Route("/api/password", change_password, methods=["POST"]),
        Route("/api/users", users, methods=["GET", "POST"]),
        Route("/api/users/{name}", user_item, methods=["DELETE", "POST"]),
        Route("/api/workspaces", workspaces),
        Route("/api/workspaces/recount", recount, methods=["POST"]),
        Route("/api/mode", modes, methods=["GET", "PUT"]),
        *workspaces_api.routes(guard, jobs, uploads, base_dir),
        *analysis_api.routes(guard, jobs, base_dir),
        *editor_api.routes(guard, base_dir),
        *assistant_api.routes(guard, jobs, base_dir),
        *export_api.routes(guard, jobs, base_dir, downloads, trainer),
        Route("/api/{rest:path}", not_found),
        Mount("/static", StaticFiles(directory=STATIC_DIR), name="static"),
    ]
    app = Starlette(routes=routes, middleware=[Middleware(SecurityHeaders)])
    app.state.store = store
    app.state.trainer = trainer
    app.state.jobs = jobs
    app.state.downloads = downloads
    return app
