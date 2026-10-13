# Fusion training

Train CLFT and CLFTv2 on ZOD, iSeAuto and Waymo, and put everything into [Visin](https://app.visin.eu): each run
with its epochs, test results on every test set, visualizations, a benchmark, and the best checkpoint (`.pth`).
The training code is [visin-fusion](https://github.com/visin-platform/visin-fusion); this repo is the configs and
a small runner.

```
configs/<dataset>/<model>/<mode>.json   54 configs: one file is one training, every setting written out
configs/<dataset>/dataset.json          the dataset's classes and splits (the downloaded archives have none)
train.py                                download a dataset; run configs
slurm.sh                                train.py as a SLURM job
setup.sh, requirements.txt              create .venv with exact package versions
.env.example                            Visin address, your key, where data goes
```

## On the HPC, from nothing to every run in Visin

**1. Get the code and set up** (once, on the login node):

```sh
git clone git@github.com:visin-platform/visin-benchmarks.git && cd visin-benchmarks
./setup.sh                       # creates .venv and .env
nano .env                        # VISIN_TOKEN, and VISIN_DATA_DIR / HF_HOME on scratch space
source .venv/bin/activate
export SBATCH_ACCOUNT=<account>  # the cluster account the GPU hours are billed to; or sbatch -A <account> each time
```

`VISIN_TOKEN` is a pipeline key limited to one Visin project (project *Settings → Pipeline keys*); every run goes
to that project.

**2. Download a dataset** (login node, needs internet). This also downloads the pretrained backbones, so the
compute nodes need no internet for them:

```sh
python train.py download iseauto        # 9 GB; links it as data/iseauto
```

**3. One short job to check the whole chain** (~1.5 h):

```sh
sbatch slurm.sh configs/iseauto/clftv2-tiny/rgb.json
```

When it ends, `slurm-<jobid>.out` should end with `Done: train, test, visualize, benchmark`, and in Visin the run
should have its epochs, test results, visualizations, a benchmark, and a model marked *ready*.

**4. All of iSeAuto**: 18 configs (~46 GPU-hours):

```sh
sbatch slurm.sh configs/iseauto                  # one job, all 18 configs one after another
sbatch slurm.sh configs/iseauto/clftv2-base      # or one job per model folder (rgb, lidar, fusion), in parallel
```

**5. ZOD and Waymo**:

```sh
python train.py download zod
python train.py download waymo
sbatch slurm.sh configs/zod
sbatch slurm.sh configs/waymo                    # 4-27 h per config
```

`slurm.sh` takes one config file or one folder, and trains a folder's configs one after another in alphabetical
order in a single job.

### Watching a job

```sh
squeue -u $USER                                # which jobs are running
tail -f slurm-<jobid>.out                      # live: the progress bar, then each epoch's metrics
grep "Epoch:" slurm-<jobid>.out | tail -1      # the epoch it is on
```

The log's first line names the config and how many epochs it trains. Epochs are counted from 0, so `Epoch: 57` is
the 58th. Within an epoch the progress bar shows batches done out of the total and the time left in the epoch,
live in `tail -f`. After each epoch the log prints training and validation mIoU, loss and per-class IoU. Visin
shows the same: the run's page fills in as each epoch finishes.

### Before submitting

- **An account is required**: `slurm.sh` sets none, so set `SBATCH_ACCOUNT` (step 1) or pass `-A <account>`. A
  cluster that requires an account refuses the job without one.
- **Edit the `#SBATCH` lines in `slurm.sh`** for your cluster, e.g. a partition. Add a `module load` if the GPUs
  need one.
- **Compute nodes that cannot reach Visin**: put `VISIN_MODE=offline` in `.env`. Reports and the checkpoint upload
  then wait on disk, and `visin sync` on the login node sends them.
- **A job that is killed** is trained from scratch when submitted again. To continue it instead, set
  `General.resume_training` to `true` in its config.

## Without SLURM

```sh
python train.py run configs/iseauto/clftv2-tiny/rgb.json     # one config
python train.py run configs/iseauto                          # a folder, one config after another
python train.py run configs/zod --stages test,benchmark      # some stages again, on what was trained
```

Outputs go to `logs/<dataset>/<model>/<mode>/`.

## The configs

| Model | Backbone | Input | Batch |
|---|---|---|---|
| `clft-base` | `vit_base_patch16_384` | 384 | 8 |
| `clft-hybrid` | `vit_base_resnet50_384` | 384 | 8 |
| `clft-large` | `vit_large_patch16_384` | 384 | 8 |
| `clftv2-tiny` | `swinv2_tiny_window16_256` | 256 | 8 |
| `clftv2-base` | `swinv2_base_window12to24_192to384_22kft1k` | 384 | 2 x 4 |
| `clftv2-large` | `swinv2_large_window12to24_192to384_22kft1k` | 384 | 2 x 4 |

Each model has `rgb`, `lidar` and `fusion`. Settings follow the runs of the
[CLFTv2 (final)](https://visin.eu/projects/69a0eb9ec480969a8673ca62) project.

| Dataset | CLFT epochs | CLFTv2 epochs | Annotations | Train / val | Test sets | Visualized |
|---|---|---|---|---|---|---|
| `zod` | 300 | 200 | `annotation_camera_only` | 1,148 / 573 | day_fair 356, day_rain 59, night_fair 97, night_rain 26, snow 41 | 7 |
| `iseauto` | 300 | 200 | `annotation` | 1,200 / 600 | four sets of 150 | 12 |
| `waymo` | 200 | 100 | `annotation` | 13,200 / 4,400 | day_fair 2,988, day_rain 904, night_fair 328, night_rain 180 | 11 |

- **Annotations:** ZOD has three annotation folders. The project trained every mode, LiDAR and fusion included, on
  `annotation_camera_only`, so the targets are the same across modes. iSeAuto's `annotation_enhanced` covers only
  2,288 of 2,400 frames, so `annotation` is used.
- **Epochs:** CLFTv2 converged: best validation mIoU at epoch 175-190 of 200, and 86-97 of 100 on Waymo. CLFT had
  not converged at 200. Its learning rate falls by x0.99 per epoch whatever `epochs` is, and validation mIoU was
  still rising at 300 (epoch 199 → 299: ZOD Base fusion 0.412 → 0.447, iSeAuto Hybrid fusion 0.641 → 0.690).
- **Licences** (`General.model_license`, `model_credit`): ZOD `cc-by-sa-4.0`, iSeAuto `cc-by-4.0`, Waymo `other`.
  Without one, Visin refuses to serve the model.
- **`batch_size` x `accumulate_batches`**: the optimizer steps once per `accumulate_batches` batches, so the
  effective batch is their product at the memory of `batch_size`. CLFTv2-Base and Large use 2 x 4 because batch 8
  does not fit a 16 GB GPU. It is not quite the same as batch 8: CLFTv2's head has a BatchNorm layer, which sees 2
  images at a time. The project ran batch 8. Batch 8 needs about 28 GiB for CLFTv2-Base and about 41 GiB for Large,
  so on a GPU with that much memory set `"batch_size": 8` and delete `accumulate_batches`.
- **`max_checkpoints`**: how many checkpoint files are kept, ranked by validation mIoU. 1 keeps only the best one.

To change a value in many files at once:

```sh
sed -i 's/"batch_size": 2,/"batch_size": 8,/; /"accumulate_batches"/d' configs/*/clftv2-base/*.json
```

## Disk

A checkpoint is written after every epoch and pruned straight away, so **more epochs do not use more disk**. At
most two `.pth` files exist per run: the best, and the last epoch's (written when training ends). A checkpoint holds
the weights and the optimizer's state, about 3x the weights: about 0.5 GB for CLFTv2-Tiny, 1.2-1.5 GB for the Base
and Hybrid models, 2.4 GB for CLFTv2-Large, 3.9 GB for CLFT-Large.

| Where | What | Size |
|---|---|---|
| `logs/.../checkpoints/` | best + last checkpoint | 2 x the sizes above |
| `logs/.../published/` | weights only: what was uploaded to Visin | a third of one checkpoint |
| `logs/...` (the rest) | epochs, TensorBoard, test results, benchmark, visualizations | tens of MB |
| `$VISIN_DATA_DIR` | datasets: ZOD 3.9 GB, Waymo 6.9 GB, iSeAuto 9.0 GB | ~20 GB |
| `$HF_HOME` | pretrained backbones | ~3 GB |
| `$VISIN_DIR` | an upload waiting to be sent | until sent |

All 54 runs, nothing cleaned up: about 225 GB. Once Visin shows a model as ready, its `checkpoints/` folder is no
longer needed:

```sh
rm -rf logs/*/*/*/checkpoints
```
