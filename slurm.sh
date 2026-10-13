#!/bin/bash
#SBATCH --job-name=fusion
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=slurm-%j.out
#SBATCH --account=

# Submit from this folder, after `python train.py download <dataset>`, with your account in SBATCH_ACCOUNT or
# -A <account>. One job trains a config file, or every config in a folder one after another:
#
#   sbatch slurm.sh configs/iseauto/clftv2-tiny/rgb.json    one config
#   sbatch slurm.sh configs/iseauto/clftv2-tiny             a folder: rgb, lidar and fusion

source .venv/bin/activate
export TQDM_MININTERVAL=60   # redraw the progress bar once a minute in this log, not ten times a second

python train.py run "$1"
