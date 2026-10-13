"""Offline regression checks for benchmark discovery and orchestration."""

import contextlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import train

ROOT = Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    def test_public_config_matrix(self):
        logdirs = set()
        models = {"clft-base", "clft-hybrid", "clft-large", "clftv2-tiny", "clftv2-base", "clftv2-large"}
        for dataset in ("iseauto", "waymo", "zod"):
            manifest = json.loads((ROOT / "configs" / dataset / "dataset.json").read_text())
            self.assertEqual(manifest["Dataset"]["name"], dataset)
            files = train.config_files(ROOT / "configs" / dataset)
            self.assertEqual(len(files), 18)
            self.assertEqual({p.parent.name for p in files}, models)
            for model in models:
                self.assertEqual({p.stem for p in files if p.parent.name == model}, {"rgb", "lidar", "fusion"})
            for path in files:
                settings = json.loads(path.read_text())
                self.assertEqual(settings["CLI"]["mode"], path.stem)
                self.assertEqual(settings["Dataset"]["dataset_root"], f"data/{dataset}")
                self.assertGreater(settings["General"]["epochs"], 0)
                self.assertTrue(settings["General"]["model_license"])
                logdir = settings["Log"]["logdir"]
                self.assertNotIn(logdir, logdirs)
                logdirs.add(logdir)

    def test_discovery_rejects_missing_empty_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "dataset.json"
            manifest.write_text("{}")
            for path in (root / "missing", root, manifest):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    train.config_files(path)
            (root / "b.json").write_text("{}")
            (root / "a.json").write_text("{}")
            self.assertEqual([p.name for p in train.config_files(root)], ["a.json", "b.json"])

    def test_invalid_stages_and_task_fail_before_importing_training(self):
        config = ROOT / "configs/iseauto/clftv2-tiny/rgb.json"
        for stages in ("", "unknown", "train,", "train,train"):
            with self.subTest(stages=stages), self.assertRaises(ValueError):
                train.run(config, stages)
        for task in (0, 2):
            with self.subTest(task=task), self.assertRaises(SystemExit):
                train.run(config, "train", task)

    def test_pipeline_continues_after_failure_and_reports_exit(self):
        pipeline = Mock(side_effect=[RuntimeError("failed stage"), None])
        fusion = types.ModuleType("visin_fusion")
        fusion.run = pipeline
        logging = types.ModuleType("visin_fusion.logging_setup")
        logging.configure_logging = Mock()
        with tempfile.TemporaryDirectory() as directory:
            for name in ("a.json", "b.json"):
                (Path(directory) / name).write_text('{"General": {"epochs": 1}}')
            with patch.dict("sys.modules", {"visin_fusion": fusion, "visin_fusion.logging_setup": logging}):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaisesRegex(SystemExit, "1 of 2 failed"):
                        train.run(directory, "train")
            self.assertEqual(pipeline.call_count, 2)
            self.assertEqual(pipeline.call_args.args[1], ["train"])

    def test_task_selection_and_export_dispatch(self):
        fusion = types.ModuleType("visin_fusion")
        fusion.run = Mock()
        logging = types.ModuleType("visin_fusion.logging_setup")
        logging.configure_logging = Mock()
        with patch.dict("sys.modules", {"visin_fusion": fusion, "visin_fusion.logging_setup": logging}):
            with patch.object(train, "export") as export, contextlib.redirect_stdout(io.StringIO()):
                folder = ROOT / "configs/iseauto/clftv2-tiny"
                selected = train.config_files(folder)[1]
                train.run(folder, "test,export", task=2)
                fusion.run.assert_called_once_with(selected, ["test"], upload=True)
                export.assert_called_once_with(selected)


if __name__ == "__main__":
    unittest.main()
