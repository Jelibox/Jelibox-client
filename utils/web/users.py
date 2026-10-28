"""
Users and sessions of the web server (configs/_users.json).

Passwords are stored as PBKDF2-HMAC-SHA256 hashes with a salt per user, never in clear. The file also keeps the
secret that signs the session cookies. A session names the user and the user's `epoch`: changing or resetting a
password (or deleting the user) bumps it, which ends every session that user had.

First start: one admin, password "admin", flagged `must_change` - the web page makes the person choose a new
password before anything else. Nothing here needs more than the standard library.
"""
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time

DEFAULT_ADMIN = "admin"
DEFAULT_PASSWORD = "admin"
MIN_PASSWORD = 6
ITERATIONS = 200_000
SESSION_SECONDS = 7 * 24 * 3600
_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,32}$")


class UserError(ValueError):
    """A message that is safe to show to the person."""


def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), ITERATIONS).hex()


def check_name(name):
    name = (name or "").strip()
    if not _NAME.match(name):
        raise UserError("A user name has 1-32 letters, digits, '_', '-' or '.'.")
    return name


def check_password(name, password):
    if len(password or "") < MIN_PASSWORD:
        raise UserError(f"The password needs at least {MIN_PASSWORD} characters.")
    if password.lower() in (DEFAULT_PASSWORD, name.lower()):
        raise UserError("Choose a password that is not 'admin' and not the user name.")
    return password


class UserStore:
    def __init__(self, path):
        self.path = path
        self._lock = threading.RLock()
        with self._lock:
            data = self._read()
            if not data.get("secret") or not data.get("users"):
                self._seed(data)

    # ---------------------------------------------------------------- file
    def _read(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write(self, data):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        try:
            os.chmod(tmp, 0o600)                       # no effect on Windows, keeps the hashes private on Linux
        except OSError:
            pass
        os.replace(tmp, self.path)

    def _seed(self, data):
        data.setdefault("secret", secrets.token_hex(32))
        if not data.get("users"):
            data["users"] = [self._record(DEFAULT_ADMIN, DEFAULT_PASSWORD, admin=True, must_change=True)]
        self._write(data)

    @staticmethod
    def _record(name, password, admin=False, must_change=False):
        salt = secrets.token_hex(16)
        return {"name": name, "salt": salt, "hash": _hash(password, salt), "admin": admin,
                "must_change": must_change, "epoch": 1, "created": int(time.time())}

    def _find(self, data, name):
        for user in data.get("users", []):
            if user["name"].lower() == (name or "").lower():
                return user
        return None

    @staticmethod
    def public(user):
        return {"name": user["name"], "admin": bool(user.get("admin")), "must_change": bool(user.get("must_change"))}

    # ---------------------------------------------------------------- queries
    def users(self):
        with self._lock:
            return [self.public(u) for u in self._read().get("users", [])]

    def get(self, name):
        with self._lock:
            user = self._find(self._read(), name)
            return self.public(user) if user else None

    def verify(self, name, password):
        """The public user record when the password is right, else None. Takes the same time either way."""
        with self._lock:
            user = self._find(self._read(), name)
        salt = user["salt"] if user else "00" * 16
        given = _hash(password or "", salt)
        if user and hmac.compare_digest(given, user["hash"]):
            return self.public(user)
        return None

    # ---------------------------------------------------------------- changes
    def add(self, name, password, admin=False):
        name = check_name(name)
        check_password(name, password)
        with self._lock:
            data = self._read()
            if self._find(data, name):
                raise UserError(f"The user '{name}' already exists.")
            data["users"].append(self._record(name, password, admin=admin))
            self._write(data)
        return self.get(name)

    def set_password(self, name, password, must_change=False):
        check_password(name, password)
        with self._lock:
            data = self._read()
            user = self._find(data, name)
            if user is None:
                raise UserError("No such user.")
            fresh = self._record(user["name"], password, admin=user.get("admin"), must_change=must_change)
            user.update(salt=fresh["salt"], hash=fresh["hash"], must_change=must_change,
                        epoch=int(user.get("epoch", 1)) + 1)
            self._write(data)

    def delete(self, name):
        with self._lock:
            data = self._read()
            user = self._find(data, name)
            if user is None:
                raise UserError("No such user.")
            if user.get("admin") and sum(1 for u in data["users"] if u.get("admin")) == 1:
                raise UserError("The last administrator cannot be deleted.")
            data["users"] = [u for u in data["users"] if u is not user]
            self._write(data)

    # ---------------------------------------------------------------- sessions
    def _sign(self, body):
        secret = bytes.fromhex(self._read()["secret"])
        return hmac.new(secret, body.encode("ascii"), hashlib.sha256).hexdigest()

    def make_token(self, name, now=None):
        with self._lock:
            user = self._find(self._read(), name)
            if user is None:
                raise UserError("No such user.")
            payload = {"u": user["name"], "e": int(user.get("epoch", 1)), "x": int((now or time.time()) + SESSION_SECONDS)}
            body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode("ascii")
            return body + "." + self._sign(body)

    def user_for_token(self, token, now=None):
        """The public user record behind a cookie value, or None (bad signature, expired, password changed, deleted)."""
        try:
            body, signature = (token or "").split(".", 1)
            with self._lock:
                if not hmac.compare_digest(signature, self._sign(body)):
                    return None
                payload = json.loads(base64.urlsafe_b64decode(body.encode("ascii")))
                if payload["x"] < (now or time.time()):
                    return None
                user = self._find(self._read(), payload["u"])
            if user is None or int(user.get("epoch", 1)) != payload["e"]:
                return None
            return self.public(user)
        except (ValueError, KeyError, TypeError):
            return None


class LoginThrottle:
    """Slows down password guessing: 5 wrong tries for the same (address, user) lock it for a while, doubling."""

    def __init__(self, free=5, base=30, longest=300, clock=time.time):
        self.free, self.base, self.longest, self.clock = free, base, longest, clock
        self._fails = {}
        self._lock = threading.Lock()

    def wait(self, key):
        """Seconds left before another try is allowed (0 = go ahead)."""
        with self._lock:
            fails, until = self._fails.get(key, (0, 0))
            return max(0, int(until - self.clock() + 0.999)) if fails >= self.free else 0

    def failed(self, key):
        with self._lock:
            fails, _ = self._fails.get(key, (0, 0))
            fails += 1
            delay = min(self.longest, self.base * 2 ** (fails - self.free)) if fails >= self.free else 0
            self._fails[key] = (fails, self.clock() + delay)

    def succeeded(self, key):
        with self._lock:
            self._fails.pop(key, None)
