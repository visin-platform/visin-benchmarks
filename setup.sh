#!/bin/bash
# Create .venv with everything train.py needs. Run once, in this folder, on a machine with internet.
# uv downloads its own Python 3.11 into your home folder, so the system's python3 can be any version.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh   # into ~/.local/bin, .bashrc untouched
    export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env: set VISIN_TOKEN in it."
fi
