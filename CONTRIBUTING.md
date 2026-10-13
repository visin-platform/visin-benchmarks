# Contributing

Issues and pull requests are welcome. This repository holds benchmark configs and
orchestration; model architectures and training engines belong in
[visin-fusion](https://github.com/visin-platform/visin-fusion).

## Local checks

Use Python 3.11. The offline tests need only the standard library; they do not
install PyTorch, download data, use a GPU, or contact Visin.

```sh
python3 -m unittest discover -s tests -v
python3 -m pip install 'ruff>=0.6,<1'
ruff check .
bash -n setup.sh slurm.sh
```

For real training, follow the README and install `requirements.txt` with
`./setup.sh`. Record the dataset, config, GPU, package versions, seed, and command
when reporting a result. Training and dataset downloads are separate from CI.
The top-level dependencies are pinned; transitive dependencies are not locked.

When changing a benchmark setting, explain the scientific reason and update the
README's tables if needed. Preserve unique log directories and the RGB/LiDAR/fusion
config matrix. Include a small runner regression test for changes to orchestration.
Never include credentials, downloaded datasets, generated logs, or checkpoints.
