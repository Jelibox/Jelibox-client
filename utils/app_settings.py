"""
App-wide (not per-workspace) settings, stored in configs/_app.json.

Kept dependency-free on purpose: utils.theme reads it at import time, long
before any workspace is loaded.
"""
import json
import os

_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                     "configs", "_app.json")


def _read():
    try:
        with open(_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def get(key, default=None):
    return _read().get(key, default)


def set(key, value):
    data = _read()
    data[key] = value
    os.makedirs(os.path.dirname(_PATH), exist_ok=True)
    tmp = _PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, _PATH)
