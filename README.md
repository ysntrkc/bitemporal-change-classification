# Bitemporal Change Classification

Multi-label change classification on aerial RGB image pairs. Given two RGB
views of the same scene at different times, the model predicts three
independent label families: objects (12 classes), events (12), attributes
(24). No-change samples carry all-zero label vectors. Outputs use sigmoid
activations throughout (multi-label).

Course project for BLM5135 (Deep Learning and Neural Networks Fundamentals).

## Layout

```
src/                shared modules (dataset, models, losses, metrics,
                    augment, ema, utils, config)
configs/            YAML deltas, one per experiment
scripts/            launchers and reporting helpers
train_phase{1,2}.py training entry points
eval_phase{1,2}.py  evaluation entry points
```

The YAMLs are intentionally small. Common hyperparameters and architecture
choices live as Python dicts (`DEFAULTS_PHASE1`, `DEFAULTS_PHASE2`) in
`src/config.py`; each YAML only declares the delta. `load_config(path)`
merges the YAML over the matching phase defaults at runtime.

Config naming follows `phase{1,2}_{main,ablation_X}_{family?}.yaml`, e.g.
`phase1_main_object.yaml`, `phase1_ablation_resnet50_event.yaml`,
`phase2_ablation_no_bit.yaml`.

## Setup

```
conda create -n blm5135 python=3.11 -y
conda activate blm5135
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

GPU sanity:

```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
```

## Training and evaluation

Bulk launchers under `scripts/` are resume-safe: they skip any
(config, seed) pair whose `best_ema.pth` already exists.

```
bash scripts/run_phase1_main.sh                # canonical Phase 1, 3 families x 3 seeds
bash scripts/run_phase1_ablation_resnet50.sh   # A1: ResNet-50 backbone swap
bash scripts/run_phase1_ablation_dbloss.sh     # DBLoss on the object family
bash scripts/run_phase2_main.sh                # canonical Phase 2 (BIT + linear + fixed)
bash scripts/run_phase2_ablation_no_bit.sh     # fusion ablation (passthrough)
bash scripts/run_phase2_ablation_q2l_uwl.sh    # full stack (Q2L heads + UWL)
```

Single-run usage:

```
python train_phase1.py --config configs/phase1_main_object.yaml --seed 42
python eval_phase1.py  --ckpt <ckpt> --config <cfg> --tta --split test
python eval_phase2.py  --ckpt <ckpt> --config <cfg> --tta --gate
```

Phase 1 eval supports `--mode tune-thresholds --split val` (which writes
a `thresholds.json` next to the checkpoint) and `--apply-thresholds <path>`
to use those thresholds at test time. Phase 2 eval has `--gate` / `--no-gate`
for the multiplicative no-change gate; the default is taken from
`cfg.inference.nochg_gate`.

After all runs land, `python scripts/ablation_table.py` regenerates
`results/ablation_table.md` from whatever metrics JSONs are on disk, and
the per-class breakdown comes from `python scripts/per_class_report.py`.

## Reproducibility

Three seeds (42, 1337, 2024) per ablation row. Each run writes
`git_hash.txt` and `config_snapshot.yaml` to its output directory; cuDNN
is set to deterministic, and the train DataLoader uses a seeded generator
plus a deterministic `worker_init_fn`.

## Dataset

The dataset is not distributed in this repository. Place `dataset.json`
and the `train/`, `val/`, `test/` image directories under `dataset/`; the
paths in `src/config.py`'s defaults assume this layout. Build the label
vocabulary once with:

```
python -m src.dataset build-vocab --json dataset/dataset.json --out configs/label_vocab.json
```
