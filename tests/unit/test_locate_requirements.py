"""LocateAnything's requirement check, model pinning, and the provider helpers - without importing torch or
transformers, and without ever loading the model."""
import re
import unittest
from unittest import mock

from utils import assistant_providers as ap
from utils import locate_anything as la
from utils import workspace_config as wc

ALL = {"torch": "2.5.1", "torchvision": "0.20.1", "transformers": "4.57.6", "huggingface_hub": "0.36.2",
       "peft": "0.21.2", "lmdb": "3.0.0", "accelerate": "1.15.0", "safetensors": "0.8.0", "decord": "0.6.0"}


def fake_versions(**overrides):
    table = dict(ALL, **overrides)
    return mock.patch.object(la, "_version", side_effect=lambda dist: table.get(dist))


class RequirementCheckTests(unittest.TestCase):
    def test_a_complete_environment_passes(self):
        with fake_versions():
            self.assertIsNone(la.check_requirements())

    def test_missing_packages_are_named(self):
        with fake_versions(peft=None, lmdb=None):
            msg = la.check_requirements()
        self.assertIn("peft", msg)
        self.assertIn("lmdb", msg)
        self.assertIn("requirements.txt", msg)

    def test_transformers_5_is_refused(self):
        with fake_versions(transformers="5.0.0"):
            msg = la.check_requirements()
        self.assertIn("below 5", msg)
        self.assertIn("5.0.0", msg)

    def test_huggingface_hub_1_is_refused(self):
        with fake_versions(huggingface_hub="1.0.1"):
            self.assertIn("below 1.0", la.check_requirements())

    def test_old_transformers_4_still_counts_as_installed(self):
        with fake_versions(transformers="4.57.6"):
            self.assertIsNone(la.check_requirements())


class ModelPinTests(unittest.TestCase):
    def test_the_model_code_is_pinned_to_a_commit(self):
        self.assertRegex(la.MODEL_REVISION, r"^[0-9a-f]{40}$")
        self.assertEqual(la.MODEL_ID, "nvidia/LocateAnything-3B")

    def test_the_model_stays_inside_jelibox_unless_the_user_chose_a_cache(self):
        self.assertTrue(la.HF_HOME.replace("\\", "/").endswith("models/_huggingface"))

    def test_importing_the_module_does_not_pull_in_torch_or_transformers(self):
        import subprocess
        import sys
        code = ("import sys; import utils.locate_anything; "
                "print([m for m in ('torch', 'transformers') if m in sys.modules])")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             cwd=la.BASE_DIR).stdout.strip()
        self.assertEqual(out, "[]")


class LocateProviderTests(unittest.TestCase):
    def test_no_target_class_is_refused_politely_before_anything_is_loaded(self):
        settings = dict(wc.LOCATE_DEFAULTS, target_classes=[{"prompt": "  ", "map_to": "a"}, {"prompt": "x", "map_to": ""}])
        with self.assertRaises(ap.AssistantError) as e:
            ap.locate_predictor(settings)
        self.assertIn("target class", str(e.exception))

    def test_missing_packages_become_an_assistant_error(self):
        settings = dict(wc.LOCATE_DEFAULTS, target_classes=[{"prompt": "cat", "map_to": "cat"}])
        with fake_versions(transformers=None):
            with self.assertRaises(ap.AssistantError) as e:
                ap.locate_predictor(settings)
        self.assertIn("transformers", str(e.exception))


class HeadProviderTests(unittest.TestCase):
    def test_mapping_prefers_explicit_choices_then_names(self):
        out = ap.map_head_classes(["Sleeping", "working", "other"], {"working": "busy", "other": "nope"},
                                  ["sleeping", "busy"])
        self.assertEqual(out, {"Sleeping": "sleeping", "working": "busy", "other": None})

    def test_no_checkpoint_chosen(self):
        with self.assertRaises(ap.AssistantError) as e:
            ap.head_predictor(dict(wc.HEAD_DEFAULTS), 0.3, ["a"], ".")
        self.assertIn("head_best.pt", str(e.exception))

    def test_missing_checkpoint_file(self):
        with self.assertRaises(ap.AssistantError) as e:
            ap.head_predictor(dict(wc.HEAD_DEFAULTS, head_path="nope/head_best.pt"), 0.3, ["a"], ".")
        self.assertIn("not found", str(e.exception))

    def test_relative_head_paths_are_taken_from_the_jelibox_folder(self):
        self.assertEqual(ap.resolve_head_path("models/x/h.pt", "/base").replace("\\", "/"), "/base/models/x/h.pt")
        self.assertEqual(ap.resolve_head_path("/abs/h.pt", "/base"), "/abs/h.pt")


class RequirementsFileTests(unittest.TestCase):
    """requirements.txt is the one place that keeps the three modes compatible - pin the pins."""

    @classmethod
    def setUpClass(cls):
        import os
        with open(os.path.join(la.BASE_DIR, "requirements.txt"), encoding="utf-8") as f:
            cls.lines = [re.sub(r"\s+#.*$", "", l).strip() for l in f if l.strip() and not l.lstrip().startswith("#")]

    def has(self, prefix):
        return any(l.startswith(prefix) for l in self.lines)

    def test_ultralytics_is_new_enough_for_dual_head_models(self):
        self.assertIn("ultralytics>=8.4.68", self.lines)

    def test_transformers_is_pinned_below_5_with_a_matching_hub(self):
        self.assertIn("transformers==4.57.6", self.lines)
        self.assertIn("huggingface_hub>=0.34.0,<1.0", self.lines)

    def test_torch_is_left_to_the_installer(self):
        self.assertFalse(self.has("torch"), "the installers choose the CUDA or CPU build of torch")

    def test_remote_code_dependencies_are_listed(self):
        for name in ("peft", "lmdb", "decord", "accelerate", "safetensors"):
            self.assertTrue(self.has(name), name)


if __name__ == "__main__":
    unittest.main()
