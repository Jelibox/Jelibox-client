#!/usr/bin/env python3
"""
Jelibox web server: open http://localhost:<port> (or http://<this computer's address>:<port> from another device
on the same network) and sign in. First sign in: user "admin", password "admin"; the page asks for a new one.

    python utils/server.py                        # all network interfaces, port 8420
    python utils/server.py --port 9000
    python utils/server.py --host 127.0.0.1       # this computer only

Settings can also come from configs/_app.json (keys `web_host`, `web_port`); a command-line option wins.
"""
import argparse
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)            # relative paths (assets, configs) are relative to the Jelibox folder

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8420


def lan_addresses():
    """This computer's addresses on the network, for the 'open it from another device' hint."""
    import socket
    found = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in found:
                found.append(ip)
    except OSError:
        pass
    return found


def log_to_file_without_console(limit=2 * 1024 * 1024):
    """Started at login by pythonw there is no console: print and uvicorn would crash on a missing stdout, so the
    output goes to temp/server.log (the old one is kept once as server.log.old when it grows past `limit`)."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    folder = os.path.join(parent_dir, "temp")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "server.log")
    try:
        if os.path.getsize(path) > limit:
            os.replace(path, path + ".old")
    except OSError:
        pass
    log = open(path, "a", buffering=1, encoding="utf-8", errors="replace")
    if sys.stdout is None:
        sys.stdout = log
    if sys.stderr is None:
        sys.stderr = log


def main(argv=None):
    log_to_file_without_console()
    from utils import app_settings
    parser = argparse.ArgumentParser(description="Jelibox web server")
    parser.add_argument("--host", default=app_settings.get("web_host", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(app_settings.get("web_port", DEFAULT_PORT)))
    args = parser.parse_args(argv)

    import uvicorn
    from utils.web.app import create_app

    where = "this computer only" if args.host in ("127.0.0.1", "localhost") else "the whole network"
    print(f"[Jelibox] Web server on port {args.port} ({where})")
    print(f"[Jelibox]   here:           http://localhost:{args.port}")
    if where == "the whole network":
        for ip in lan_addresses():
            print(f"[Jelibox]   other devices:  http://{ip}:{args.port}")
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
