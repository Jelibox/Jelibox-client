import ast
import glob
import json
import os
import stat
import sys
import tempfile
import unittest
import uuid
from unittest import mock

from utils import app_settings, collab
from utils.collab import identity, servers

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class Isolated(unittest.TestCase):
    """Points both settings files at a temp dir so no test touches the real configs."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._old = (app_settings._PATH, servers._SECRETS_PATH)
        app_settings._PATH = os.path.join(self.tmp, "configs", "_app.json")
        servers._SECRETS_PATH = os.path.join(self.tmp, "configs", "_server.json")

    def tearDown(self):
        app_settings._PATH, servers._SECRETS_PATH = self._old


class DeviceIdentityTests(Isolated):
    def test_created_on_first_use_and_then_stable(self):
        self.assertIsNone(app_settings.get("device_id"))
        first = identity.get_device_id()
        self.assertTrue(identity.is_valid_device_id(first))
        self.assertEqual(identity.get_device_id(), first)
        self.assertEqual(app_settings.get("device_id"), first)

    def test_survives_other_settings_changes(self):
        first = identity.get_device_id()
        app_settings.set("theme", "dark")
        self.assertEqual(identity.get_device_id(), first)

    def test_a_damaged_value_is_replaced_by_a_valid_one(self):
        for bad in ("", "not-a-uuid", 123, None, ["x"], str(uuid.uuid4()).upper()):
            app_settings.set("device_id", bad)
            value = identity.get_device_id()
            self.assertTrue(identity.is_valid_device_id(value), bad)

    def test_reset_gives_a_new_identity(self):
        first = identity.get_device_id()
        second = identity.reset_device_id()
        self.assertNotEqual(first, second)
        self.assertEqual(identity.get_device_id(), second)

    def test_ids_are_unique(self):
        seen = set()
        for _ in range(50):
            seen.add(identity.reset_device_id())
        self.assertEqual(len(seen), 50)


class NormalizeUrlTests(unittest.TestCase):
    def test_accepted_forms(self):
        cases = {
            "192.168.1.5": "http://192.168.1.5:8420",
            "192.168.1.5:9000": "http://192.168.1.5:9000",
            "jelibox.local": "http://jelibox.local:8420",
            "  http://lab-server:8420/  ": "http://lab-server:8420",
            "http://lab-server/some/path?x=1": "http://lab-server:8420",
            "https://lab.example.org": "https://lab.example.org:8420",
            "HTTP://Lab.Local": "http://lab.local:8420",
            "[fe80::1]": "http://[fe80::1]:8420",
        }
        for text, expected in cases.items():
            self.assertEqual(servers.normalize_url(text), expected, text)

    def test_rejected_forms(self):
        for bad in ("", "   ", None, "a b", "ftp://host", "http://", "host:abc", "host:0", "host:70000",
                    "user:pw@host", "http://user@host", "javascript:alert(1)"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                servers.normalize_url(bad)


class ServerListTests(Isolated):
    def test_empty_by_default(self):
        self.assertEqual(servers.list_servers(), [])

    def test_add_list_update_remove(self):
        a = servers.add_server("Lab", "192.168.1.10")
        b = servers.add_server("", "jelibox.local")
        self.assertEqual(b["name"], "jelibox.local")                       # defaults to the host
        self.assertEqual([e["url"] for e in servers.list_servers()],
                         ["http://192.168.1.10:8420", "http://jelibox.local:8420"])
        servers.update_server(a["id"], name="Lab 2", url="192.168.1.11:9000")
        self.assertEqual(servers.list_servers()[0]["name"], "Lab 2")
        self.assertEqual(servers.list_servers()[0]["url"], "http://192.168.1.11:9000")
        servers.remove_server(a["id"])
        self.assertEqual([e["id"] for e in servers.list_servers()], [b["id"]])

    def test_duplicate_address_is_refused(self):
        servers.add_server("A", "10.0.0.2")
        with self.assertRaises(ValueError):
            servers.add_server("B", "http://10.0.0.2:8420/")
        other = servers.add_server("C", "10.0.0.3")
        with self.assertRaises(ValueError):
            servers.update_server(other["id"], url="10.0.0.2")

    def test_name_length_limit_and_unknown_id(self):
        with self.assertRaises(ValueError):
            servers.add_server("x" * (servers.MAX_NAME + 1), "10.0.0.2")
        with self.assertRaises(KeyError):
            servers.update_server("nope", name="x")
        with self.assertRaises(KeyError):
            servers.set_server_id("nope", "abc")

    def test_server_id_can_be_remembered(self):
        e = servers.add_server("A", "10.0.0.2")
        servers.set_server_id(e["id"], "3f0c-aaaa")
        self.assertEqual(servers.list_servers()[0]["server_id"], "3f0c-aaaa")

    def test_malformed_stored_entries_are_ignored(self):
        app_settings.set("servers", [{"id": "ok", "name": "A", "url": "http://a:8420"}, "junk", {"id": 1},
                                     {"name": "no id", "url": "x"}, None])
        self.assertEqual([e["id"] for e in servers.list_servers()], ["ok"])
        app_settings.set("servers", "not a list")
        self.assertEqual(servers.list_servers(), [])


class CredentialTests(Isolated):
    def setUp(self):
        super().setUp()
        self.entry = servers.add_server("Lab", "10.0.0.2")["id"]

    def test_state_follows_the_credentials(self):
        self.assertEqual(servers.state(self.entry), servers.NOT_CONNECTED)
        servers.store_pending(self.entry, "req-1", "claim-secret")
        self.assertEqual(servers.state(self.entry), servers.PENDING)
        self.assertEqual(servers.get_pending(self.entry), {"request_id": "req-1", "claim_secret": "claim-secret"})
        servers.store_access_key(self.entry, "jbk_ABCDEFGH12345678")
        self.assertEqual(servers.state(self.entry), servers.CONNECTED)
        self.assertIsNone(servers.get_pending(self.entry))                 # the key replaces the pending request
        servers.clear_credentials(self.entry)
        self.assertEqual(servers.state(self.entry), servers.NOT_CONNECTED)

    def test_secrets_never_reach_the_app_settings_file(self):
        servers.store_pending(self.entry, "req-1", "claim-secret")
        servers.store_access_key(self.entry, "jbk_SECRETVALUE")
        with open(app_settings._PATH, encoding="utf-8") as f:
            text = f.read()
        self.assertNotIn("jbk_SECRETVALUE", text)
        self.assertNotIn("claim-secret", text)
        with open(servers._SECRETS_PATH, encoding="utf-8") as f:
            self.assertIn("jbk_SECRETVALUE", f.read())

    @unittest.skipIf(sys.platform == "win32", "POSIX permission bits")
    def test_secrets_file_is_private(self):
        servers.store_access_key(self.entry, "jbk_x")
        mode = stat.S_IMODE(os.stat(servers._SECRETS_PATH).st_mode)
        self.assertEqual(mode, 0o600)

    def test_write_is_atomic_and_leaves_no_temp_file(self):
        servers.store_access_key(self.entry, "jbk_x")
        self.assertFalse(os.path.exists(servers._SECRETS_PATH + ".tmp"))
        with mock.patch("os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                servers.store_access_key(self.entry, "jbk_new")
        self.assertEqual(servers.get_access_key(self.entry), "jbk_x")      # old file untouched

    def test_removing_a_server_deletes_its_credentials_only(self):
        other = servers.add_server("Other", "10.0.0.3")["id"]
        servers.store_access_key(self.entry, "jbk_one")
        servers.store_access_key(other, "jbk_two")
        servers.remove_server(self.entry)
        self.assertIsNone(servers.get_access_key(self.entry))
        self.assertEqual(servers.get_access_key(other), "jbk_two")

    def test_corrupt_or_odd_secrets_file_means_not_connected(self):
        os.makedirs(os.path.dirname(servers._SECRETS_PATH), exist_ok=True)
        for content in ("{not json", "[]", '{"servers": []}', '{"servers": {"%s": "x"}}' % self.entry):
            with open(servers._SECRETS_PATH, "w", encoding="utf-8") as f:
                f.write(content)
            self.assertEqual(servers.state(self.entry), servers.NOT_CONNECTED, content)

    def test_mask_shows_only_a_prefix(self):
        self.assertEqual(servers.mask(""), "")
        self.assertEqual(servers.mask(None), "")
        masked = servers.mask("jbk_ABCDEFGH12345678")
        self.assertTrue(masked.startswith("jbk_ABCD"))
        self.assertNotIn("12345678", masked)


class FeatureFlagTests(Isolated):
    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JELIBOX_COLLAB", None)
            self.assertFalse(collab.enabled())

    def test_environment_variable_turns_it_on(self):
        for value in ("1", "true", "YES", " on "):
            with mock.patch.dict(os.environ, {"JELIBOX_COLLAB": value}):
                self.assertTrue(collab.enabled(), value)
        with mock.patch.dict(os.environ, {"JELIBOX_COLLAB": "0"}):
            self.assertFalse(collab.enabled())

    def test_setting_turns_it_on(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JELIBOX_COLLAB", None)
            app_settings.set("collab_enabled", True)
            self.assertTrue(collab.enabled())
            app_settings.set("collab_enabled", "yes")                      # only a real true counts
            self.assertFalse(collab.enabled())


class NoNetworkYetTests(unittest.TestCase):
    """Jelibox must stay silent on the network until the connection features exist."""

    FORBIDDEN = {"socket", "ssl", "http", "urllib.request", "urllib3", "requests", "httpx", "aiohttp",
                 "ftplib", "smtplib", "telnetlib", "xmlrpc", "websockets", "paramiko"}

    def test_collab_package_and_dialog_import_no_network_modules(self):
        files = glob.glob(os.path.join(REPO, "utils", "collab", "*.py")) + \
            [os.path.join(REPO, "utils", "ServerSettingsDialog.py")]
        self.assertGreaterEqual(len(files), 4)
        for path in files:
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read(), path)
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
                for name in names:
                    self.assertFalse(
                        name in self.FORBIDDEN or name.split(".")[0] in self.FORBIDDEN,
                        f"{os.path.basename(path)} imports {name}")


if __name__ == "__main__":
    unittest.main()
