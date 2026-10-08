"""
The servers this installation knows about.

Two files, on purpose:
  configs/_app.json     non-secret list:  [{"id", "name", "url", "server_id"}]
  configs/_server.json  secrets, per server id:  {"access_key"} once approved, or
                        {"request_id", "claim_secret"} while waiting for approval

Keeping the secrets apart means _app.json can be shared for support without leaking a key.
No network access here. Callers that later send the access key must first check that the
server answers with the stored server_id - the URL alone may point at another machine after
a DHCP change.
"""
import json
import os
import secrets
import urllib.parse

from .. import app_settings

DEFAULT_PORT = 8420
MAX_NAME = 60
_KEY = "servers"

_SECRETS_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
                             "configs", "_server.json")

NOT_CONNECTED, PENDING, CONNECTED = "not_connected", "pending", "connected"


# ---------------------------------------------------------------- addresses
def normalize_url(text):
    """'192.168.1.5' / 'host:9000' / 'http://host/' -> 'http://host:port'. Raises ValueError."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Enter the server address.")
    if any(c.isspace() for c in raw):
        raise ValueError("The address cannot contain spaces.")
    if "://" not in raw:
        raw = "http://" + raw
    parts = urllib.parse.urlsplit(raw)
    if parts.scheme not in ("http", "https"):
        raise ValueError("The address must start with http:// or https://.")
    if parts.username or parts.password:
        raise ValueError("Do not put a user name or password in the address.")
    host = parts.hostname
    if not host:
        raise ValueError("The address has no host name.")
    try:
        port = parts.port
    except ValueError:
        raise ValueError("The port must be a number between 1 and 65535.") from None
    if port == 0:
        raise ValueError("The port must be a number between 1 and 65535.")
    host_text = f"[{host}]" if ":" in host else host
    return f"{parts.scheme}://{host_text}:{port or DEFAULT_PORT}"


# ---------------------------------------------------------------- the list
def _clean(entry):
    if not isinstance(entry, dict):
        return None
    sid, name, url = entry.get("id"), entry.get("name"), entry.get("url")
    if not (isinstance(sid, str) and sid and isinstance(name, str) and isinstance(url, str)):
        return None
    server_id = entry.get("server_id")
    return {"id": sid, "name": name, "url": url,
            "server_id": server_id if isinstance(server_id, str) else None}


def list_servers():
    stored = app_settings.get(_KEY, [])
    if not isinstance(stored, list):
        return []
    return [e for e in (_clean(x) for x in stored) if e]


def _save(entries):
    app_settings.set(_KEY, entries)


def _check_name(name, url):
    name = (name or "").strip()
    if len(name) > MAX_NAME:
        raise ValueError(f"The name can have at most {MAX_NAME} characters.")
    return name or urllib.parse.urlsplit(url).hostname


def add_server(name, url):
    url = normalize_url(url)
    entries = list_servers()
    if any(e["url"] == url for e in entries):
        raise ValueError("This server is already in the list.")
    entry = {"id": secrets.token_hex(4), "name": _check_name(name, url), "url": url, "server_id": None}
    while any(e["id"] == entry["id"] for e in entries):
        entry["id"] = secrets.token_hex(4)
    entries.append(entry)
    _save(entries)
    return entry


def update_server(entry_id, name=None, url=None):
    entries = list_servers()
    entry = next((e for e in entries if e["id"] == entry_id), None)
    if entry is None:
        raise KeyError(entry_id)
    new_url = normalize_url(url) if url is not None else entry["url"]
    if any(e["url"] == new_url and e["id"] != entry_id for e in entries):
        raise ValueError("This server is already in the list.")
    entry["url"] = new_url
    entry["name"] = _check_name(entry["name"] if name is None else name, new_url)
    _save(entries)
    return entry


def set_server_id(entry_id, server_id):
    """Remember which server answered at this address (a later version fills this in)."""
    entries = list_servers()
    for e in entries:
        if e["id"] == entry_id:
            e["server_id"] = server_id
            _save(entries)
            return
    raise KeyError(entry_id)


def remove_server(entry_id):
    entries = [e for e in list_servers() if e["id"] != entry_id]
    _save(entries)
    clear_credentials(entry_id)


# ---------------------------------------------------------------- secrets
def _read_secrets():
    try:
        with open(_SECRETS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    servers = data.get("servers") if isinstance(data, dict) else None
    return servers if isinstance(servers, dict) else {}


def _write_secrets(servers):
    """Write via a private temp file and rename, so a crash never leaves a half-written file."""
    os.makedirs(os.path.dirname(_SECRETS_PATH), exist_ok=True)
    tmp = _SECRETS_PATH + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "servers": servers}, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, _SECRETS_PATH)
    try:
        os.chmod(_SECRETS_PATH, 0o600)
    except OSError:
        pass


def _item(entry_id):
    item = _read_secrets().get(entry_id)
    return item if isinstance(item, dict) else {}


def store_pending(entry_id, request_id, claim_secret):
    """Remember a join request so closing Jelibox does not lose it."""
    secrets_by_id = _read_secrets()
    secrets_by_id[entry_id] = {"request_id": request_id, "claim_secret": claim_secret}
    _write_secrets(secrets_by_id)


def store_access_key(entry_id, access_key):
    secrets_by_id = _read_secrets()
    secrets_by_id[entry_id] = {"access_key": access_key}
    _write_secrets(secrets_by_id)


def get_access_key(entry_id):
    value = _item(entry_id).get("access_key")
    return value if isinstance(value, str) and value else None


def get_pending(entry_id):
    item = _item(entry_id)
    if isinstance(item.get("request_id"), str) and isinstance(item.get("claim_secret"), str):
        return {"request_id": item["request_id"], "claim_secret": item["claim_secret"]}
    return None


def clear_credentials(entry_id):
    secrets_by_id = _read_secrets()
    if entry_id in secrets_by_id:
        del secrets_by_id[entry_id]
        _write_secrets(secrets_by_id)


def state(entry_id):
    if get_access_key(entry_id):
        return CONNECTED
    if get_pending(entry_id):
        return PENDING
    return NOT_CONNECTED


def mask(access_key):
    """Safe text to show or log for a key: its first characters only."""
    if not access_key:
        return ""
    return access_key[:8] + "…"
