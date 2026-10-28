"""Accounts, sessions and the login throttle of the web server (standard library only)."""
import json
import os
import tempfile
import unittest

from utils.web import users as u


class StoreBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "configs", "_users.json")
        self.store = u.UserStore(self.path)


class FirstStartTests(StoreBase):
    def test_a_new_install_has_admin_admin_that_must_be_changed(self):
        self.assertEqual(self.store.users(), [{"name": "admin", "admin": True, "must_change": True}])
        self.assertEqual(self.store.verify("admin", "admin")["name"], "admin")

    def test_passwords_are_not_stored_in_clear(self):
        with open(self.path, encoding="utf-8") as f:
            record = json.load(f)["users"][0]
        self.assertNotIn("password", record)
        self.assertNotEqual(record["hash"], "admin")
        self.assertEqual(len(record["hash"]), 64)

    def test_the_secret_and_users_survive_a_restart(self):
        self.store.add("ana", "secret1")
        again = u.UserStore(self.path)
        self.assertEqual({x["name"] for x in again.users()}, {"admin", "ana"})
        token = self.store.make_token("ana")
        self.assertEqual(again.user_for_token(token)["name"], "ana")


class PasswordTests(StoreBase):
    def test_wrong_password_and_unknown_user_are_refused(self):
        self.assertIsNone(self.store.verify("admin", "nope"))
        self.assertIsNone(self.store.verify("ghost", "admin"))
        self.assertIsNone(self.store.verify("admin", ""))

    def test_user_names_are_not_case_sensitive(self):
        self.assertIsNotNone(self.store.verify("ADMIN", "admin"))
        with self.assertRaises(u.UserError):
            self.store.add("Admin", "secret1")

    def test_weak_passwords_are_refused(self):
        for bad in ("short", "admin", "", "ana"):
            with self.assertRaises(u.UserError, msg=bad):
                self.store.add("ana", bad)

    def test_bad_names_are_refused(self):
        for bad in ("", " ", "a b", "x" * 33, "../etc", "ana/bob"):
            with self.assertRaises(u.UserError, msg=bad):
                self.store.add(bad, "secret1")

    def test_changing_the_password_clears_the_must_change_flag(self):
        self.store.set_password("admin", "newpass1")
        self.assertFalse(self.store.get("admin")["must_change"])
        self.assertIsNone(self.store.verify("admin", "admin"))
        self.assertIsNotNone(self.store.verify("admin", "newpass1"))

    def test_an_admin_reset_forces_a_new_choice(self):
        self.store.add("ana", "secret1")
        self.store.set_password("ana", "tempo123", must_change=True)
        self.assertTrue(self.store.get("ana")["must_change"])


class SessionTests(StoreBase):
    def test_a_token_names_the_user(self):
        token = self.store.make_token("admin")
        self.assertEqual(self.store.user_for_token(token)["name"], "admin")

    def test_forged_and_broken_tokens_are_refused(self):
        token = self.store.make_token("admin")
        body, sig = token.split(".")
        for bad in (None, "", "x", body + ".", body + "." + "0" * 64, "A" + body[1:] + "." + sig, token + "0"):
            self.assertIsNone(self.store.user_for_token(bad), bad)

    def test_a_token_from_another_installation_is_refused(self):
        other = u.UserStore(os.path.join(self.tmp.name, "other", "_users.json"))
        self.assertIsNone(self.store.user_for_token(other.make_token("admin")))

    def test_tokens_expire(self):
        token = self.store.make_token("admin", now=1000)
        self.assertIsNotNone(self.store.user_for_token(token, now=1000 + u.SESSION_SECONDS - 1))
        self.assertIsNone(self.store.user_for_token(token, now=1000 + u.SESSION_SECONDS + 1))

    def test_a_password_change_ends_the_old_sessions(self):
        token = self.store.make_token("admin")
        self.store.set_password("admin", "newpass1")
        self.assertIsNone(self.store.user_for_token(token))
        self.assertIsNotNone(self.store.user_for_token(self.store.make_token("admin")))

    def test_a_deleted_user_loses_the_session(self):
        self.store.add("ana", "secret1")
        token = self.store.make_token("ana")
        self.store.delete("ana")
        self.assertIsNone(self.store.user_for_token(token))

    def test_the_last_administrator_cannot_be_deleted(self):
        with self.assertRaises(u.UserError):
            self.store.delete("admin")
        self.store.add("boss", "secret1", admin=True)
        self.store.delete("admin")
        self.assertEqual([x["name"] for x in self.store.users()], ["boss"])


class ThrottleTests(unittest.TestCase):
    def setUp(self):
        self.now = [1000.0]
        self.t = u.LoginThrottle(free=3, base=30, longest=100, clock=lambda: self.now[0])

    def test_free_tries_then_a_growing_wait(self):
        key = ("1.2.3.4", "admin")
        for _ in range(2):
            self.t.failed(key)
        self.assertEqual(self.t.wait(key), 0)
        self.t.failed(key)                                  # third wrong one: locked for 30 s
        self.assertEqual(self.t.wait(key), 30)
        self.now[0] += 31
        self.assertEqual(self.t.wait(key), 0)
        self.t.failed(key)                                  # fourth: 60 s
        self.assertEqual(self.t.wait(key), 60)
        self.t.failed(key)                                  # capped at `longest`
        self.assertEqual(self.t.wait(key), 100)

    def test_success_resets_and_keys_are_independent(self):
        a, b = ("1.1.1.1", "admin"), ("2.2.2.2", "admin")
        for _ in range(3):
            self.t.failed(a)
        self.assertGreater(self.t.wait(a), 0)
        self.assertEqual(self.t.wait(b), 0)
        self.t.succeeded(a)
        self.assertEqual(self.t.wait(a), 0)


if __name__ == "__main__":
    unittest.main()
