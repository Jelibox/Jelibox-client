"""
Jelibox web server (Jelibox 1.0): the same workspaces and annotations as the desktop window, in a browser.

    users.py   accounts, password hashes, session cookies, login throttle (standard library only)
    app.py     the Starlette application: API under /api, the page under /
    static/    the browser side (plain HTML, CSS and JavaScript; nothing to build)

Started by utils/server.py. Nothing in this package imports tkinter, torch or ultralytics at import time.
"""
