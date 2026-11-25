"""Archives in a dataset folder: .zip, .rar, .7z and split sets, found anywhere, unpacked safely.

Real .rar files are made with the `rar` program and read with the 7-Zip program py7zz brings along; the tests that
need either skip when it is not there (CI only has a few packages installed). Names and grouping are tested with
empty files, no tool needed."""
import io
import os
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from unittest import mock

from utils import archive_support as a

HAS_7Z = a.find_7z() is not None
HAS_RAR = shutil.which("rar") is not None


def rar_readable():
    """True when a rar can be made here and the 7-Zip program found reads it (a system p7zip often has no RAR method)."""
    if not (HAS_7Z and HAS_RAR):
        return False
    work = tempfile.mkdtemp()
    try:
        make_rar_parts({"a.bin": os.urandom(5000)}, os.path.join(work, "src"), "t")      # split, like the tests below
        a.unpack_all(os.path.join(work, "src"), dest=os.path.join(work, "out"))
        return True
    except (ValueError, OSError, subprocess.CalledProcessError):
        return False
    finally:
        shutil.rmtree(work, ignore_errors=True)


def touch(folder, *names):
    for name in names:
        path = os.path.join(folder, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "wb").close()


def write(path, data, mode="wb"):
    with open(path, mode) as f:
        f.write(data)


def read(path):
    with open(path, "rb") as f:
        return f.read()


def make_rar_parts(files, out_dir, name, volume="2k"):
    """{relative name: bytes} -> name.part1.rar, name.part2.rar ... in out_dir; the list of their paths."""
    src = tempfile.mkdtemp()
    try:
        for rel, data in files.items():
            path = os.path.join(src, *rel.split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
        os.makedirs(out_dir, exist_ok=True)
        subprocess.run(["rar", "a", "-inul", "-r", f"-v{volume}", os.path.join(out_dir, name + ".rar"),
                        *os.listdir(src)], cwd=src, check=True)
    finally:
        shutil.rmtree(src, ignore_errors=True)
    return sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.startswith(name + ".part"))


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def listing(self, root):
        return sorted(os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/")
                      for d, _dirs, files in os.walk(root) for f in files)


class FindingTests(Base):
    def found(self):
        return [(os.path.relpath(entry, self.tmp).replace(os.sep, "/"), len(parts))
                for entry, parts in a.find_archives(self.tmp)]

    def test_a_split_rar_is_one_archive_wherever_it_sits(self):
        touch(self.tmp, "x/y/fire.part1.rar", "x/y/fire.part2.rar", "x/y/fire.part3.rar", "annotations.rar", "pic.jpg")
        self.assertEqual(self.found(), [("annotations.rar", 1), ("x/y/fire.part1.rar", 3)])

    def test_part_numbers_may_be_padded(self):
        touch(self.tmp, "fire.part01.rar", "fire.part02.rar", "fire.part10.rar", *[f"fire.part{i:02d}.rar" for i in range(3, 10)])
        self.assertEqual(self.found(), [("fire.part01.rar", 10)])

    def test_the_old_style_rar_with_r00_r01_is_one_archive(self):
        touch(self.tmp, "old.rar", "old.r00", "old.r01")
        self.assertEqual(self.found(), [("old.rar", 3)])

    def test_split_7z_and_zip_are_found_by_their_001(self):
        touch(self.tmp, "a.7z.001", "a.7z.002", "b.zip.001", "b.zip.002", "c.7z", "d.zip", "e.rar")
        self.assertEqual(self.found(), [("a.7z.001", 2), ("b.zip.001", 2), ("c.7z", 1), ("d.zip", 1), ("e.rar", 1)])

    def test_two_sets_in_one_folder_do_not_mix(self):
        touch(self.tmp, "a.part1.rar", "a.part2.rar", "b.part1.rar", "b.part2.rar", "b.part3.rar")
        self.assertEqual(self.found(), [("a.part1.rar", 2), ("b.part1.rar", 3)])

    def test_a_missing_part_is_named(self):
        touch(self.tmp, "fire.part1.rar", "fire.part3.rar")
        with self.assertRaises(ValueError) as caught:
            a.find_archives(self.tmp)
        self.assertIn("'fire'", str(caught.exception))
        self.assertIn("part 2", str(caught.exception))

    def test_a_set_without_its_first_part_is_reported(self):
        touch(self.tmp, "fire.part2.rar", "fire.part3.rar")
        with self.assertRaises(ValueError) as caught:
            a.find_archives(self.tmp)
        self.assertIn("part 1", str(caught.exception))

    def test_an_r00_without_its_rar_is_reported(self):
        touch(self.tmp, "old.r00")
        with self.assertRaises(ValueError) as caught:
            a.find_archives(self.tmp)
        self.assertIn("old.rar", str(caught.exception))

    def test_ordinary_files_are_not_archives(self):
        touch(self.tmp, "a.jpg", "a.txt", "a.xml", "classes.txt", "notes.rarx", "r00")
        self.assertEqual(a.find_archives(self.tmp), [])

    def test_the_name_of_a_set(self):
        for name, stem in (("fire.part1.rar", "fire"), ("fire.rar", "fire"), ("fire.7z.001", "fire"),
                           ("old.r00", "old"), ("data.zip", "data"), ("my.data.part02.rar", "my.data")):
            self.assertEqual(a._stem(name), stem, name)


class ZipTests(Base):
    def make_zip(self, path, entries):
        with zipfile.ZipFile(path, "w") as z:
            for name, data in entries.items():
                z.writestr(name, data)

    def test_into_a_temp_folder_the_source_is_not_touched(self):
        src, dest = os.path.join(self.tmp, "src"), os.path.join(self.tmp, "dest")
        os.makedirs(src)
        self.make_zip(os.path.join(src, "data.zip"), {"d/a.txt": b"1", "b.txt": b"2"})
        made = a.unpack_all(src, dest=dest)
        self.assertEqual(len(made), 1)
        self.assertEqual(self.listing(made[0]), ["b.txt", "d/a.txt"])
        self.assertEqual(os.listdir(src), ["data.zip"])

    def test_in_place_with_remove_leaves_a_folder_and_no_archive(self):
        self.make_zip(os.path.join(self.tmp, "data.zip"), {"a.txt": b"1"})
        made = a.unpack_all(self.tmp, remove=True)
        self.assertEqual(self.listing(self.tmp), ["data/a.txt"])
        self.assertEqual(made, [os.path.join(self.tmp, "data")])

    def test_unsafe_names_are_skipped(self):
        os.makedirs(os.path.join(self.tmp, "w"))
        self.make_zip(os.path.join(self.tmp, "w", "bad.zip"),
                      {"ok.txt": b"1", "../../escape.txt": b"2", "C:/win.txt": b"3", "con.txt": b"4"})
        a.unpack_all(os.path.join(self.tmp, "w"), remove=True)
        self.assertEqual(self.listing(os.path.join(self.tmp, "w")), ["bad/ok.txt"])
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "escape.txt")))

    def test_an_archive_inside_an_archive_is_opened_too(self):
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w") as z:
            z.writestr("deep.txt", b"x")
        self.make_zip(os.path.join(self.tmp, "outer.zip"), {"inner.zip": inner.getvalue(), "top.txt": b"y"})
        made = a.unpack_all(self.tmp, remove=True)
        self.assertEqual(len(made), 2)
        self.assertEqual(self.listing(self.tmp), ["outer/inner/deep.txt", "outer/top.txt"])

    def test_nesting_has_a_limit(self):
        data = b"x"
        for level in range(a.MAX_DEPTH + 2):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                z.writestr("n.zip" if level else "f.txt", data)
            data = buf.getvalue()
        write(os.path.join(self.tmp, "start.zip"), data)
        with self.assertRaises(ValueError) as caught:
            a.unpack_all(self.tmp, remove=True)
        self.assertIn("nested", str(caught.exception))

    def test_a_broken_zip_is_reported(self):
        write(os.path.join(self.tmp, "bad.zip"), b"not a zip")
        with self.assertRaises(ValueError) as caught:
            a.unpack_all(self.tmp)
        self.assertIn("not a valid zip", str(caught.exception))

    def test_too_little_room_is_reported_before_writing(self):
        self.make_zip(os.path.join(self.tmp, "big.zip"), {"a.txt": b"1" * 1000})
        with mock.patch.object(a, "free_bytes", return_value=10):
            with self.assertRaises(ValueError) as caught:
                a.unpack_all(self.tmp, remove=True)
        self.assertIn("free disk space", str(caught.exception))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "big.zip")), "nothing is deleted when it failed")

    def test_links_are_removed(self):
        if os.name == "nt":
            self.skipTest("links need rights on Windows")
        os.symlink("/etc", os.path.join(self.tmp, "etc-link"))
        os.symlink("/etc/hostname", os.path.join(self.tmp, "file-link"))
        write(os.path.join(self.tmp, "real.txt"), "", "w")
        a._drop_links(self.tmp)
        self.assertEqual(os.listdir(self.tmp), ["real.txt"])

    def test_without_the_7zip_program_the_message_says_what_to_do(self):
        touch(self.tmp, "a.rar")
        with mock.patch.object(a, "find_7z", return_value=None):
            with self.assertRaises(ValueError) as caught:
                a.unpack_all(self.tmp)
        self.assertIn("py7zz", str(caught.exception))


@unittest.skipUnless(rar_readable(), "needs the rar program and a 7-Zip program that reads rar (py7zz)")
class RarTests(Base):
    FILES = {"img/a.jpg": os.urandom(3000), "img/b.jpg": os.urandom(3000), "labels/a.txt": b"0 0.5 0.5 0.2 0.2\n"}

    def test_split_rar_is_unpacked_from_its_first_part(self):
        src = os.path.join(self.tmp, "src", "deep")
        parts = make_rar_parts(self.FILES, src, "fire")
        self.assertGreater(len(parts), 2)
        dest = os.path.join(self.tmp, "dest")
        made = a.unpack_all(os.path.join(self.tmp, "src"), dest=dest)
        self.assertEqual(len(made), 1)
        self.assertEqual(self.listing(made[0]), sorted(self.FILES))
        self.assertEqual(read(os.path.join(made[0], "img", "a.jpg")), self.FILES["img/a.jpg"])
        self.assertEqual(len(os.listdir(src)), len(parts), "the archives are not touched when a dest is given")

    def test_in_place_with_remove_deletes_every_part(self):
        make_rar_parts(self.FILES, self.tmp, "fire")
        a.unpack_all(self.tmp, remove=True)
        self.assertEqual(self.listing(self.tmp), sorted("fire/" + n for n in self.FILES))

    def test_annotations_and_pictures_in_separate_archives(self):
        make_rar_parts({"labels/a.txt": b"0 .5 .5 .2 .2\n"}, os.path.join(self.tmp, "x"), "annotations", volume="100k")
        make_rar_parts({"img/a.jpg": os.urandom(5000)}, os.path.join(self.tmp, "y", "z"), "pictures")
        made = a.unpack_all(self.tmp, dest=os.path.join(self.tmp, "out"))
        names = sorted(n for m in made for n in self.listing(m))
        self.assertEqual(names, ["img/a.jpg", "labels/a.txt"])

    def test_a_missing_middle_part_is_reported_with_its_number(self):
        parts = make_rar_parts(self.FILES, self.tmp, "fire")
        os.remove(parts[1])
        with self.assertRaises(ValueError) as caught:
            a.unpack_all(self.tmp)
        self.assertIn("part 2", str(caught.exception))

    def test_a_cut_off_last_part_is_reported(self):
        parts = make_rar_parts(self.FILES, self.tmp, "fire")
        with open(parts[-1], "r+b") as f:
            f.truncate(30)
        with self.assertRaises(ValueError):
            a.unpack_all(self.tmp)

    def test_a_password_protected_rar_is_reported(self):
        src = os.path.join(self.tmp, "s")
        os.makedirs(src)
        write(os.path.join(src, "a.txt"), "x", "w")
        subprocess.run(["rar", "a", "-inul", "-hpsecret", os.path.join(self.tmp, "pw.rar"), "a.txt"], cwd=src, check=True)
        with self.assertRaises(ValueError) as caught:
            a.unpack_all(self.tmp, dest=os.path.join(self.tmp, "out"))
        self.assertIn("password", str(caught.exception))

    def test_a_file_that_is_not_a_rar_is_reported(self):
        write(os.path.join(self.tmp, "fake.rar"), b"hello, not a rar" * 20)
        with self.assertRaises(ValueError) as caught:
            a.unpack_all(self.tmp, dest=os.path.join(self.tmp, "out"))
        self.assertIn("fake.rar", str(caught.exception))


@unittest.skipUnless(HAS_7Z, "needs the 7-Zip program (py7zz)")
class SevenZipTests(Base):
    def test_a_split_7z_is_unpacked_from_its_001(self):
        src, data = os.path.join(self.tmp, "src"), os.path.join(self.tmp, "data")
        os.makedirs(data)
        for n in ("a", "b"):
            write(os.path.join(data, n + ".bin"), os.urandom(5000))
        os.makedirs(src)
        subprocess.run([a.find_7z(), "a", "-bd", "-v2k", os.path.join(src, "pack.7z"), data], check=True,
                       stdout=subprocess.DEVNULL)
        self.assertTrue(os.path.exists(os.path.join(src, "pack.7z.003")))
        made = a.unpack_all(src, dest=os.path.join(self.tmp, "out"))
        self.assertEqual(self.listing(made[0]), ["data/a.bin", "data/b.bin"])

    def test_a_plain_zip_through_7zip_when_it_is_split_naming(self):
        data = os.path.join(self.tmp, "z.zip")
        with zipfile.ZipFile(data, "w") as z:
            z.writestr("a.txt", b"hello")
        os.rename(data, data + ".001")
        made = a.unpack_all(self.tmp, dest=os.path.join(self.tmp, "out"))
        self.assertEqual(self.listing(made[0]), ["a.txt"])


class StaleTests(Base):
    def test_old_leftovers_go_and_fresh_ones_stay(self):
        old, fresh = os.path.join(self.tmp, "old"), os.path.join(self.tmp, "fresh")
        os.makedirs(old)
        os.makedirs(fresh)
        past = os.path.getmtime(old) - 2 * a.STALE_SECONDS
        os.utime(old, (past, past))
        a.forget_stale(self.tmp)
        self.assertEqual(os.listdir(self.tmp), ["fresh"])
        a.forget_stale(os.path.join(self.tmp, "missing"))                  # no folder: nothing happens


if __name__ == "__main__":
    unittest.main()
