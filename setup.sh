#!/bin/bash
# Create .venv with everything train.py needs. Run once, in this folder, on a machine with internet.
# Needs Python 3.10 or newer: on a cluster, `module load` a Python first if python3 is older.
set -e

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env: set VISIN_TOKEN in it."
fi
