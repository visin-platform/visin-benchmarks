"""Train the configs under configs/ and report everything to Visin.

    python train.py download iseauto                         once, with internet: dataset + pretrained backbones
    python train.py run configs/iseauto/clftv2-tiny/rgb.json one config
    python train.py run configs/iseauto                      every config in a folder, one after another
    python train.py run configs/iseauto --task 3             only the 3rd config of the folder
    python train.py run configs/zod --stages test,benchmark  only some stages

A run trains, tests the best checkpoint on every test set, renders and uploads visualizations, benchmarks the
model, uploads the best checkpoint, and exports it to ONNX. Settings (VISIN_URL, VISIN_TOKEN, ...) come from .env.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

STAGES = "train,test,visualize,benchmark,export"


def config_files(path):
    """A config file, or every config in a folder (dataset.json files are manifests, not configs)."""
    path = Path(path)
    if path.is_file():
        if path.suffix != ".json" or path.name == "dataset.json":
            raise ValueError(f"not a training config: {path}")
        return [path]
    if not path.is_dir():
        raise ValueError(f"config path does not exist: {path}")
    configs = sorted(p for p in path.rglob("*.json") if p.name != "dataset.json")
    if not configs:
        raise ValueError(f"no training configs under {path}")
    return configs


def download(dataset):
    """Download a dataset and link it as data/<dataset>, then cache the pretrained backbones its configs use."""
    import timm
    import visin
    from visin_fusion.dataset_tools import make_manifest

    visin.enable_console_logging(stream=sys.stdout)
    root = visin.Datasets().download(dataset).resolve()  # resumes if interrupted; skipped when already complete

    # The archives have no manifest (splits, classes, normalization): write one from configs/<dataset>/dataset.json.
    if not (root / "dataset.json").exists():
        make_manifest.main(["--root", str(root), "--config", f"configs/{dataset}/dataset.json"])

    # Which Visin dataset this is: a run reads it from its dataset folder and records it, so copy it there.
    marker = root.parent / ".visin-dataset.json"
    if marker.exists():
        shutil.copy(marker, root / marker.name)

    link = Path("data") / dataset  # what the configs' dataset_root names
    link.parent.mkdir(exist_ok=True)
    link.unlink(missing_ok=True)
    link.symlink_to(root)
    print(f"{link} -> {root}")

    # Compute nodes may have no internet: download the pretrained weights now, into the cache ($HF_HOME).
    backbones = set()
    for config in config_files(f"configs/{dataset}"):
        for section in json.loads(config.read_text()).values():
            if isinstance(section, dict) and "model_timm" in section:
                backbones.add(section["model_timm"])
    for name in sorted(backbones):
        print("pretrained weights:", name, flush=True)
        timm.create_model(name, pretrained=True)


def export(config):
    """Export the run's published best checkpoint to ONNX, check it against PyTorch on real frames, and save it in
    the run next to the .pth. Needs the run's Visin UUID (from its epoch logs), so it works on a finished run."""
    from visin_fusion.cli import export as export_model

    settings = json.loads(config.read_text())
    general = settings.get("General", {})
    logdir = Path(settings["Log"]["logdir"])
    root = Path(settings["Dataset"]["dataset_root"])

    epochs = sorted((logdir / "epochs").glob("epoch_*.json"), key=lambda p: p.stat().st_mtime)
    published = sorted((logdir / "published").glob("*/checkpoint.pth"), key=lambda p: p.stat().st_mtime)
    if not epochs or not published:
        raise RuntimeError(f"nothing to export in {logdir}: no epochs or no published checkpoint (train first)")
    run_uuid = json.loads(epochs[-1].read_text()).get("training_uuid")
    if not run_uuid:
        raise RuntimeError(f"{epochs[-1]} names no training_uuid: the run was not reported to Visin")

    # Parity needs real frames: the few the run visualizes, else the smallest test set.
    frames = root / "visualization.txt"
    if not frames.is_file():
        frames = min(root.glob("test_*.txt"), key=lambda p: len(p.read_text().split()), default=None)
    if frames is None:
        raise RuntimeError(f"no frame list under {root} to check the export against")

    argv = ["--checkpoint", str(published[-1]), "--output", str(logdir / "model.onnx"), "--frames", str(frames),
            "--dataset-root", str(root), "--publish", "--run", run_uuid]
    if general.get("model_license"):
        argv += ["--licence", general["model_license"]]
    if general.get("model_credit"):
        argv += ["--credit", general["model_credit"]]
    try:
        export_model(argv)
    except SystemExit as error:  # the CLI reports failures by exiting with a message
        raise RuntimeError(error.code) from None


def run(path, stages, task=None):
    """Run the stages of every config the path names; a failing config does not stop the others."""
    configs = config_files(path)
    stage_list = stages.split(",")
    if any(stage not in STAGES.split(",") for stage in stage_list):
        raise ValueError(f"--stages must be comma-separated names from: {STAGES}")
    if len(stage_list) != len(set(stage_list)):
        raise ValueError("--stages must not contain duplicates")
    pipeline_stages = [s for s in stage_list if s != "export"]
    if task is not None:
        if not 1 <= task <= len(configs):
            sys.exit(f"--task {task}: {path} has {len(configs)} configs")
        configs = [configs[task - 1]]

    import visin_fusion
    from visin_fusion.logging_setup import configure_logging

    configure_logging()  # show the pipeline's own messages: each stage as it starts, "Done" at the end

    failed = []
    for config in configs:
        epochs = json.loads(config.read_text()).get("General", {}).get("epochs", "?")
        print(f"=== {config}: {stages}, {epochs} epochs (the log counts them from 0)", flush=True)
        try:
            if pipeline_stages:
                visin_fusion.run(config, pipeline_stages, upload=True)
            if "export" in stage_list:
                print(f"=== export: {config}", flush=True)
                export(config)
        except Exception as error:
            print(f"FAILED {config}: {error}", file=sys.stderr, flush=True)
            failed.append(str(config))
    if failed:
        sys.exit(f"{len(failed)} of {len(configs)} failed: {', '.join(failed)}")
    print(f"All {len(configs)} finished: {stages}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)  # configs/, data/ and logs/ are relative to this folder

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    download_parser = commands.add_parser("download", help="a dataset and its pretrained backbones (needs internet)")
    download_parser.add_argument("dataset", choices=["zod", "waymo", "iseauto"])
    run_parser = commands.add_parser("run", help="train, test, visualize and benchmark configs")
    run_parser.add_argument("path", help="a config file, or a folder of them")
    run_parser.add_argument("--stages", default=STAGES, help=f"comma-separated (default: {STAGES})")
    run_parser.add_argument("--task", type=int, help="only the Nth config of the folder, counting from 1")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(".env")  # variables already exported win

    try:
        if args.command == "download":
            download(args.dataset)
        else:
            run(args.path, args.stages, args.task)
    except ValueError as error:
        parser.error(str(error))
