#!/bin/bash
#SBATCH --job-name=fusion
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint="A100-80"
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH -t 5-0
#SBATCH --mail-type=ALL
#SBATCH --account=project_autolab

# Submit from this folder, after `python train.py download <dataset>`, with your account in SBATCH_ACCOUNT or
# -A <account>. One job trains a config file, or every config in a folder one after another:
#
#   sbatch slurm.sh configs/iseauto/clftv2-tiny/rgb.json    one config
#   sbatch slurm.sh configs/iseauto/clftv2-tiny             a folder: rgb, lidar and fusion

source .venv/bin/activate

python train.py run "$1"
