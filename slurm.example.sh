#!/bin/bash
#SBATCH --job-name=fusion
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH -t 5-0

# Submit from this folder, after `python train.py download <dataset>`, with your account in SBATCH_ACCOUNT or
# -A <account>. One job trains a config file, or every config in a folder one after another:
#
#   sbatch slurm.sh configs/iseauto/clftv2-tiny/rgb.json    one config
#   sbatch slurm.sh configs/iseauto/clftv2-tiny             a folder: rgb, lidar and fusion

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch slurm.sh <config.json|config-folder>" >&2
    exit 2
fi

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
source .venv/bin/activate

python train.py run "$1"
