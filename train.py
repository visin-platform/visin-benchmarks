"""Train the configs under configs/ and report everything to Visin.

    python train.py download iseauto                         once, with internet: dataset + pretrained backbones
    python train.py run configs/iseauto/clftv2-tiny/rgb.json one config
    python train.py run configs/iseauto                      every config in a folder, one after another
    python train.py run configs/iseauto --task 3             only the 3rd config of the folder (slurm.sh uses this)
    python train.py run configs/zod --stages test,benchmark  only some stages

A run trains, tests the best checkpoint on every test set, renders and uploads visualizations, benchmarks the
model, and uploads the best checkpoint. Settings (VISIN_URL, VISIN_TOKEN, ...) come from .env.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

STAGES = "train,test,visualize,benchmark"


def config_files(path):
    """A config file, or every config in a folder (dataset.json files are manifests, not configs)."""
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.json") if p.name != "dataset.json")


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


def run(path, stages, task=None):
    """Run the stages of every config the path names; a failing config does not stop the others."""
    import visin_fusion
    from visin_fusion.logging_setup import configure_logging

    configure_logging()  # show the pipeline's own messages: each stage as it starts, "Done" at the end
    configs = config_files(path)
    if task is not None:
        if not 1 <= task <= len(configs):
            sys.exit(f"--task {task}: {path} has {len(configs)} configs")
        configs = [configs[task - 1]]

    failed = []
    for config in configs:
        epochs = json.loads(config.read_text()).get("General", {}).get("epochs", "?")
        print(f"=== {config}: {stages}, {epochs} epochs (the log counts them from 0)", flush=True)
        try:
            visin_fusion.run(config, stages.split(","), upload=True)
        except Exception as error:
            print(f"FAILED {config}: {error}", file=sys.stderr, flush=True)
            failed.append(str(config))
    if failed:
        sys.exit(f"{len(failed)} of {len(configs)} failed: {', '.join(failed)}")
    print(f"All {len(configs)} finished: {stages}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)  # configs/, data/ and logs/ are relative to this folder
    load_dotenv(".env")  # variables already exported win

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    download_parser = commands.add_parser("download", help="a dataset and its pretrained backbones (needs internet)")
    download_parser.add_argument("dataset", choices=["zod", "waymo", "iseauto"])
    run_parser = commands.add_parser("run", help="train, test, visualize and benchmark configs")
    run_parser.add_argument("path", help="a config file, or a folder of them")
    run_parser.add_argument("--stages", default=STAGES, help=f"comma-separated (default: {STAGES})")
    run_parser.add_argument("--task", type=int, help="only the Nth config of the folder, counting from 1")
    args = parser.parse_args()

    if args.command == "download":
        download(args.dataset)
    else:
        run(args.path, args.stages, args.task)
