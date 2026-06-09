# NeDM HMMWV Bootstrap

This workspace treats a local Project Chrono checkout as the simulator backend and adds a first data-collection pipeline for a single-vehicle overfit experiment on the HMMWV full model.

The attached research note at [deep-research-report-vehicle.md](deep-research-report-vehicle.md) points toward hybrid or physics-guided models and multi-step validation. The code added here focuses on the prerequisite step: collecting clean, episode-level PyChrono data from deliberately excited HMMWV maneuvers on a fixed terrain.

Start with [docs/data_collection_pipeline.md](docs/data_collection_pipeline.md), then run:

```bash
conda env create -f environment.yml
conda activate tutorial
python scripts/collect_hmmwv_dataset.py --config configs/hmmwv_overfit_v1.json
```

If your local conda install shows the same plugin issue I hit in the sandbox, use:

```bash
export CONDA_NO_PLUGINS=true
source /home/harry/anaconda3/etc/profile.d/conda.sh
conda activate tutorial
python scripts/collect_hmmwv_dataset.py --config configs/hmmwv_overfit_v1.json
```

## Sequence Model Training

The first HMMWV training pipeline is documented in [docs/hmmwv_training_pipeline.md](docs/hmmwv_training_pipeline.md).

Pretrained checkpoints are tracked with Git LFS. See [docs/model_checkpoints.md](docs/model_checkpoints.md) for the current best HMMWV dynamics checkpoints and loading instructions.

Set up the same environment used for training:

```bash
conda env create -f environment.yml
conda activate tutorial
git lfs install
git lfs pull
```

`environment.yml` is the recommended portable setup file. `environment.lock.yml` is a fuller no-build export of the current `tutorial` environment for closer reproduction on Linux.

Build the processed training cache from the episode CSVs:

```bash
conda activate tutorial
python scripts/build_hmmwv_training_dataset.py \
  --dataset-root artifacts/datasets/hmmwv_overfit_6k \
  --output-dir artifacts/training_datasets/hmmwv_overfit_6k_seq_v1
```

Train the GPT-style sequence model:

```bash
conda activate tutorial
python scripts/train_hmmwv_dynamics.py --config configs/hmmwv_transformer_v1.json
```

## RL Tracking

Trajectory-tracking PPO against the frozen NN HMMWV dynamics model is documented in [docs/rl_tracking.md](docs/rl_tracking.md). The default setup uses the v07 dynamics checkpoint, 20 compact training-set references, and policy control every 5 NN steps.
