"""The web API: sign in, forced password change, users, workspaces, modes. Skipped without starlette + httpx."""
import os
import shutil
import tempfile
import unittest

try:
    from starlette.testclient import TestClient
    from utils.web.app import create_app
except Exception:                                   # CI installs only pillow, numpy and opencv
    TestClient = None

H = {"X-Jelibox": "1"}


@unittest.skipIf(TestClient is None, "starlette / httpx not installed")
class WebBase(unittest.TestCase):
    def setUp(self):
        from utils import app_settings, dataset_import, workspace_config, workspace_manager
        self.tmp = tempfile.mkdtemp(prefix="jelibox_web_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        configs = os.path.join(self.tmp, "configs")
        datasets = os.path.join(self.tmp, "datasetsInput")
        os.makedirs(configs)
        os.makedirs(datasets)
        saved = (app_settings._PATH, workspace_config.CONFIGS_ROOT, workspace_manager.BASE_DIR,
                 workspace_manager.DATASETS_ROOT, workspace_manager.CONFIGS_ROOT, dataset_import.BASE_DIR,
                 dataset_import.DATASETS_ROOT, dataset_import.CONFIGS_ROOT)
        self.addCleanup(self._restore, saved)
        app_settings._PATH = os.path.join(configs, "_app.json")
        workspace_config.CONFIGS_ROOT = workspace_manager.CONFIGS_ROOT = configs
        workspace_manager.BASE_DIR = self.tmp
        workspace_manager.DATASETS_ROOT = datasets
        dataset_import.BASE_DIR, dataset_import.DATASETS_ROOT, dataset_import.CONFIGS_ROOT = self.tmp, datasets, configs
        self.datasets = datasets
        self.app = create_app(os.path.join(configs, "_users.json"), base_dir=self.tmp)
        self.client = TestClient(self.app, base_url="http://testserver")

    @staticmethod
    def _restore(saved):
        from utils import app_settings, dataset_import, workspace_config, workspace_manager
        (app_settings._PATH, workspace_config.CONFIGS_ROOT, workspace_manager.BASE_DIR,
         workspace_manager.DATASETS_ROOT, workspace_manager.CONFIGS_ROOT, dataset_import.BASE_DIR,
         dataset_import.DATASETS_ROOT, dataset_import.CONFIGS_ROOT) = saved

    def login(self, name="admin", password="admin", client=None):
        return (client or self.client).post("/api/login", json={"username": name, "password": password}, headers=H)

    def signed_in(self, new="adminpass1"):
        """Sign in as admin and get past the forced password change."""
        self.login()
        self.assertEqual(self.client.post("/api/password", json={"old": "admin", "new": new}, headers=H).status_code, 200)


class PageTests(WebBase):
    def test_the_page_and_its_files_are_served_without_signing_in(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Jelibox", page.text)
        for path in ("/static/app.js", "/static/ui.js", "/static/editor.js", "/static/theme.js", "/static/style.css",
                     "/static/jelibox.svg"):
            self.assertEqual(self.client.get(path).status_code, 200, path)

    def test_security_headers(self):
        headers = self.client.get("/").headers
        self.assertEqual(headers["x-content-type-options"], "nosniff")
        self.assertEqual(headers["x-frame-options"], "DENY")
        self.assertIn("frame-ancestors 'none'", headers["content-security-policy"])

    def test_the_static_folder_does_not_leak_other_files(self):
        self.assertEqual(self.client.get("/static/../app.py").status_code, 404)
        self.assertEqual(self.client.get("/static/..%2Fapp.py").status_code, 404)

    def test_the_page_has_no_inline_script_the_csp_would_block(self):
        import re
        html = self.client.get("/").text
        for tag in re.findall(r"<script[^>]*>", html):
            self.assertIn("src=", tag, "an inline script is blocked by the Content-Security-Policy")

    def test_the_web_code_imports_no_tkinter_and_no_workspace_globals(self):
        import subprocess
        import sys
        code = ("import sys; sys.path.insert(0, '.'); import utils.web.app, utils.dataset_import, utils.dataset_analysis; "
                "bad = [m for m in ('tkinter', 'utils.config', 'torch', 'ultralytics') if m in sys.modules]; "
                "print(bad); sys.exit(1 if bad else 0)")
        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        done = subprocess.run([sys.executable, "-c", code], cwd=repo, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

    def test_unknown_api_paths_answer_json_404(self):
        res = self.client.get("/api/nothing")
        self.assertEqual(res.status_code, 404)
        self.assertIn("error", res.json())


class LoginTests(WebBase):
    def test_everything_needs_a_session(self):
        for path in ("/api/workspaces", "/api/mode", "/api/users"):
            self.assertEqual(self.client.get(path).status_code, 401, path)
        self.assertIsNone(self.client.get("/api/me").json()["user"])

    def test_default_admin_must_choose_a_password_first(self):
        res = self.login()
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["user"]["must_change"])
        cookie = res.headers["set-cookie"].lower()
        self.assertIn("httponly", cookie)
        self.assertIn("samesite=strict", cookie)
        blocked = self.client.get("/api/workspaces")
        self.assertEqual((blocked.status_code, blocked.json().get("must_change")), (403, True))

    def test_changing_the_password_opens_the_app(self):
        self.signed_in()
        self.assertEqual(self.client.get("/api/workspaces").status_code, 200)
        self.assertFalse(self.client.get("/api/me").json()["user"]["must_change"])
        self.client.post("/api/logout", json={}, headers=H)
        self.assertEqual(self.client.get("/api/workspaces").status_code, 401)
        self.assertEqual(self.login(password="admin").status_code, 401)           # the old one is gone
        self.assertEqual(self.login(password="adminpass1").status_code, 200)

    def test_wrong_current_password_and_weak_new_one_are_refused(self):
        self.login()
        for old, new in (("wrong", "adminpass1"), ("admin", "admin"), ("admin", "abc")):
            res = self.client.post("/api/password", json={"old": old, "new": new}, headers=H)
            self.assertEqual(res.status_code, 400, (old, new))

    def test_a_forged_cookie_is_not_a_session(self):
        self.client.cookies.set("jelibox_session", "eyJ1IjoiYWRtaW4ifQ.deadbeef")
        self.assertIsNone(self.client.get("/api/me").json()["user"])
        self.assertEqual(self.client.get("/api/workspaces").status_code, 401)

    def test_many_wrong_passwords_lock_the_account_for_a_while(self):
        for _ in range(5):
            self.assertEqual(self.login(password="bad").status_code, 401)
        locked = self.login(password="bad")
        self.assertEqual(locked.status_code, 429)
        self.assertGreater(locked.json()["retry_after"], 0)
        self.assertEqual(self.login(password="admin").status_code, 429)           # even the right one waits

    def test_changing_state_without_the_header_is_refused(self):
        self.signed_in()
        self.assertEqual(self.client.put("/api/mode", json={"mode": "sam2_dynamic"}).status_code, 403)
        self.assertEqual(self.client.put("/api/mode", json={"mode": "sam2_dynamic"},
                                         headers={**H, "Origin": "http://evil.example"}).status_code, 403)


class UserManagementTests(WebBase):
    def test_admin_adds_resets_and_deletes_users(self):
        self.signed_in()
        made = self.client.post("/api/users", json={"username": "ana", "password": "secret1"}, headers=H)
        self.assertEqual(made.status_code, 201)
        self.assertEqual({u["name"] for u in self.client.get("/api/users").json()["users"]}, {"admin", "ana"})
        self.assertEqual(self.client.post("/api/users", json={"username": "ana", "password": "secret1"},
                                          headers=H).status_code, 400)
        self.assertEqual(self.client.post("/api/users/ana", json={"password": "tempo123"}, headers=H).status_code, 200)
        self.assertEqual(self.client.delete("/api/users/ana", headers=H).status_code, 200)
        self.assertEqual(self.client.delete("/api/users/admin", headers=H).status_code, 400)   # not yourself

    def test_a_plain_user_cannot_manage_users_but_can_change_their_password(self):
        self.signed_in()
        self.client.post("/api/users", json={"username": "ana", "password": "secret1"}, headers=H)
        other = TestClient(self.client.app, base_url="http://testserver")
        self.assertEqual(self.login("ana", "secret1", other).status_code, 200)
        self.assertEqual(other.get("/api/users").status_code, 403)
        self.assertEqual(other.post("/api/users", json={"username": "eve", "password": "secret1"},
                                    headers=H).status_code, 403)
        self.assertEqual(other.get("/api/workspaces").status_code, 200)
        self.assertEqual(other.post("/api/password", json={"old": "secret1", "new": "another1"}, headers=H).status_code, 200)

    def test_a_reset_forces_the_new_password_choice(self):
        self.signed_in()
        self.client.post("/api/users", json={"username": "ana", "password": "secret1"}, headers=H)
        self.client.post("/api/users/ana", json={"password": "tempo123"}, headers=H)
        other = TestClient(self.client.app, base_url="http://testserver")
        self.assertTrue(self.login("ana", "tempo123", other).json()["user"]["must_change"])
        self.assertEqual(other.get("/api/workspaces").status_code, 403)


class WorkspaceTests(WebBase):
    def test_the_workspace_list_groups_instances_like_the_picker(self):
        from tests.helpers import make_images
        make_images(os.path.join(self.datasets, "cats-1"), count=2)
        make_images(os.path.join(self.datasets, "cats-2"), count=1)
        make_images(os.path.join(self.datasets, "dogs-1"), count=3)
        self.signed_in()
        rows = {w["name"]: w for w in self.client.get("/api/workspaces").json()["workspaces"]}
        self.assertEqual(list(rows), ["cats", "dogs"])
        self.assertEqual([i["name"] for i in rows["cats"]["instances"]], ["cats-1", "cats-2"])
        self.assertIsNone(rows["cats"]["counted"])                                   # nothing counts by itself
        self.assertIsNone(rows["cats"]["instances"][0]["images"])

    def test_refresh_counts_the_pictures(self):
        from tests.helpers import make_images
        make_images(os.path.join(self.datasets, "cats-1"), count=2)
        make_images(os.path.join(self.datasets, "cats-2"), count=1)
        self.signed_in()
        rows = self.client.post("/api/workspaces/recount", json={}, headers=H).json()["workspaces"]
        self.assertEqual((rows[0]["images"], [i["images"] for i in rows[0]["instances"]]), (3, [2, 1]))
        again = self.client.get("/api/workspaces").json()["workspaces"][0]
        self.assertEqual(again["images"], 3)

    def test_an_empty_install_lists_nothing(self):
        self.signed_in()
        self.assertEqual(self.client.get("/api/workspaces").json(), {"workspaces": []})


class ModeTests(WebBase):
    def test_the_four_modes_are_listed_and_the_choice_is_kept(self):
        self.signed_in()
        data = self.client.get("/api/mode").json()
        self.assertEqual([m["id"] for m in data["modes"]],
                         ["yolo_world", "locate_anything", "custom_head", "sam2_dynamic"])
        changed = self.client.put("/api/mode", json={"mode": "sam2_dynamic"}, headers=H)
        self.assertEqual(changed.json()["mode"], "sam2_dynamic")
        self.assertEqual(self.client.get("/api/mode").json()["mode"], "sam2_dynamic")
        self.assertEqual(self.client.put("/api/mode", json={"mode": "bogus"}, headers=H).status_code, 400)


if __name__ == "__main__":
    unittest.main()
