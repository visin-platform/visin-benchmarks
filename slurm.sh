#!/bin/bash
#SBATCH --job-name=fusion
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out

# Submit from this folder, after `python train.py download <dataset>`:
#
#   sbatch slurm.sh configs/iseauto/clftv2-tiny/rgb.json    one config
#   sbatch --array=1-18%4 slurm.sh configs/iseauto          a folder: array task N runs its Nth config, 4 at a time

source .venv/bin/activate
export TQDM_MININTERVAL=60   # redraw the progress bar once a minute in this log, not ten times a second

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    python train.py run "$1" --task "$SLURM_ARRAY_TASK_ID"
else
    python train.py run "$1"
fi
