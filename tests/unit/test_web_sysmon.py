"""The server resource monitor: parsing of what the machine reports, the snapshot, and the /api/system route."""
import os
import unittest
from collections import namedtuple
from unittest import mock

from tests.unit.test_web_app import H, TestClient, WebBase

Reading = namedtuple("Reading", "label current high critical")


class ParsingTests(unittest.TestCase):
    def setUp(self):
        from utils.web import sysmon
        self.sm = sysmon

    def test_nvidia_smi_rows_become_gpus_with_bytes(self):
        text = "NVIDIA GeForce RTX 3060, 37, 61, 1024, 12288\nNVIDIA T400, 0, 40, 100, 2048\n"
        gpus = self.sm.parse_nvidia_smi(text)
        self.assertEqual([g["name"] for g in gpus], ["NVIDIA GeForce RTX 3060", "NVIDIA T400"])
        self.assertEqual((gpus[0]["percent"], gpus[0]["temp"]), (37.0, 61.0))
        self.assertEqual((gpus[0]["mem_used"], gpus[0]["mem_total"]), (1024 * 1024 * 1024, 12288 * 1024 * 1024))

    def test_not_available_fields_are_none_and_junk_is_skipped(self):
        gpus = self.sm.parse_nvidia_smi("Quadro, [N/A], [N/A], [N/A], [N/A]\n\nno driver\n")
        self.assertEqual(len(gpus), 1)
        self.assertEqual((gpus[0]["percent"], gpus[0]["temp"], gpus[0]["mem_total"]), (None, None, None))
        self.assertEqual(self.sm.parse_nvidia_smi(""), [])
        self.assertEqual(self.sm.parse_nvidia_smi(None), [])

    def test_cpu_temperature_prefers_a_cpu_sensor_chip_and_its_hottest_core(self):
        sensors = {"nvme": [Reading("Composite", 48.0, 80, 90)],
                   "coretemp": [Reading("Package id 0", 55.0, 80, 100), Reading("Core 0", 58.5, 80, 100)]}
        self.assertEqual(self.sm.pick_cpu_temp(sensors), 58.5)
        self.assertEqual(self.sm.pick_cpu_temp({"k10temp": [Reading("Tctl", 44.0, 0, 0)]}), 44.0)

    def test_cpu_temperature_falls_back_to_any_chip_and_ignores_nonsense(self):
        self.assertEqual(self.sm.pick_cpu_temp({"board": [Reading("x", 39.0, 0, 0)]}), 39.0)
        self.assertIsNone(self.sm.pick_cpu_temp({"board": [Reading("x", 0.0, 0, 0), Reading("y", 400.0, 0, 0)]}))
        self.assertIsNone(self.sm.pick_cpu_temp({}))
        self.assertIsNone(self.sm.pick_cpu_temp(None))

    def test_windows_thermal_zones_are_kelvin_and_the_hottest_wins(self):
        self.assertEqual(self.sm.parse_thermal_zones("315\n305\n"), 41.9)          # 315 K
        self.assertIsNone(self.sm.parse_thermal_zones("0\n2\n"))
        self.assertIsNone(self.sm.parse_thermal_zones(""))
        self.assertIsNone(self.sm.parse_thermal_zones("Access denied"))

    def test_the_drive_that_holds_jelibox_comes_first_and_is_marked(self):
        first, second = os.path.abspath(os.sep), os.path.abspath(os.path.join(os.sep, "tools"))    # spelled for this OS
        parts = [{"mount": first, "fstype": "NTFS", "fixed": True}, {"mount": second, "fstype": "NTFS", "fixed": True},
                 {"mount": first, "fstype": "NTFS", "fixed": True}]
        usage = lambda mount: (50, 100) if mount == second else (80, 100)
        rows = self.sm.disk_rows(parts, usage, os.path.join(second, "Jelibox"))
        self.assertEqual([(r["mount"], r["here"]) for r in rows], [(second, True), (first, False)])
        self.assertEqual(rows[0]["percent"], 50.0)

    def test_the_deepest_mount_is_the_one_that_holds_the_folder(self):
        top, data = os.path.abspath(os.sep), os.path.abspath(os.path.join(os.sep, "data"))     # spelled for this OS
        parts = [{"mount": top, "fstype": "ext4", "fixed": False}, {"mount": data, "fstype": "ext4", "fixed": False}]
        rows = self.sm.disk_rows(parts, lambda m: (10, 100), os.path.join(data, "jelibox"))
        self.assertEqual([(r["mount"], r["here"]) for r in rows], [(data, True), (top, False)])
        rows = self.sm.disk_rows(parts, lambda m: (10, 100), data + "base" + os.sep + "x")        # /database is not inside /data
        self.assertEqual([(r["mount"], r["here"]) for r in rows], [(top, True), (data, False)])

    def test_pseudo_filesystems_and_unreadable_drives_are_left_out(self):
        parts = [{"mount": "/", "fstype": "ext4", "fixed": False}, {"mount": "/snap/x", "fstype": "squashfs", "fixed": False},
                 {"mount": "/run", "fstype": "tmpfs", "fixed": False}, {"mount": "/gone", "fstype": "ext4", "fixed": False}]

        def usage(mount):
            if mount == "/gone":
                raise OSError("not ready")
            return 1, 10
        rows = self.sm.disk_rows(parts, usage, "/home/me/jelibox")
        self.assertEqual([r["mount"] for r in rows], ["/"])


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        from utils.web import sysmon
        self.sm = sysmon
        sysmon._cache.update(snap=None, at=0.0, gpu=(0.0, None), cpu_temp=(0.0, None))

    def tearDown(self):
        self.sm._cache.update(snap=None, at=0.0, gpu=(0.0, None), cpu_temp=(0.0, None))

    def test_a_snapshot_has_every_number_the_page_shows(self):
        with mock.patch.object(self.sm, "read_gpus", return_value=[{"name": "G", "percent": 5.0, "temp": 50.0, "mem_used": 1, "mem_total": 2}]), \
                mock.patch.object(self.sm, "read_cpu_temp", return_value=47.0):
            snap = self.sm.snapshot(os.getcwd())
        self.assertEqual(set(snap), {"time", "cpu", "ram", "gpus", "disks", "platform"})
        self.assertEqual(snap["cpu"]["temp"], 47.0)
        self.assertEqual(snap["gpus"][0]["name"], "G")
        if self.sm._psutil() is not None:
            self.assertTrue(0 <= snap["cpu"]["percent"] <= 100)
            self.assertTrue(0 < snap["ram"]["percent"] <= 100)
            self.assertGreaterEqual(snap["ram"]["total"], snap["ram"]["used"])
        self.assertTrue(snap["disks"] and snap["disks"][0]["here"])

    def test_slow_sources_are_not_asked_again_within_their_cache_time(self):
        with mock.patch.object(self.sm, "read_gpus", return_value=[]) as gpus, \
                mock.patch.object(self.sm, "read_cpu_temp", return_value=None) as temp:
            self.sm.snapshot(os.getcwd())
            self.sm._cache["at"] = 0.0                       # the snapshot is stale, the slow sources are not
            self.sm.snapshot(os.getcwd())
        self.assertEqual((gpus.call_count, temp.call_count), (1, 1))

    def test_a_computer_without_gpu_or_sensor_reports_empty_not_an_error(self):
        with mock.patch.object(self.sm, "read_gpus", return_value=[]), mock.patch.object(self.sm, "read_cpu_temp", return_value=None):
            snap = self.sm.snapshot(os.getcwd())
        self.assertEqual(snap["gpus"], [])
        self.assertIsNone(snap["cpu"]["temp"])

    def test_without_psutil_the_disk_still_shows(self):
        with mock.patch.object(self.sm, "_psutil", return_value=None), mock.patch.object(self.sm, "read_gpus", return_value=[]), \
                mock.patch.object(self.sm, "read_cpu_temp", return_value=None):
            snap = self.sm.snapshot(os.getcwd())
        self.assertIsNone(snap["cpu"]["percent"])
        self.assertIsNone(snap["ram"]["percent"])
        self.assertEqual(len(snap["disks"]), 1)


class RouteTests(WebBase):
    def test_signed_in_users_get_the_snapshot_and_others_get_nothing(self):
        self.assertEqual(self.client.get("/api/system").status_code, 401)
        self.signed_in()
        with mock.patch("utils.web.sysmon.read_gpus", return_value=[]), mock.patch("utils.web.sysmon.read_cpu_temp", return_value=None):
            from utils.web import sysmon
            sysmon._cache.update(snap=None, at=0.0, gpu=(0.0, None), cpu_temp=(0.0, None))
            res = self.client.get("/api/system")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(set(res.json()), {"time", "cpu", "ram", "gpus", "disks", "platform"})


if __name__ == "__main__":
    unittest.main()
