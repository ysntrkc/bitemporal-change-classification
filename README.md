# Bitemporal Change Classification

Multi-label change classification on aerial RGB image pairs — course project for BLM5135 (Deep Learning and Neural Networks Fundamentals).

Given two RGB images of the same scene captured at different times, the model predicts three independent label families simultaneously: objects (12 classes), events (12 classes), and attributes (24 classes). Examples showing no significant change carry all-zero label vectors across every family. The task is multi-label, so outputs use sigmoid activations throughout.

## Repository layout

```
src/          Implementation modules (dataset, models, losses, augmentation, metrics, utils)
configs/      YAML experiment configurations
scripts/      Bash launchers for training and evaluation runs
reports/      LaTeX sources for the final report
train_phase1.py  Training entry point (Phase 1 single-task models)
train_phase2.py  Training entry point (Phase 2 unified multi-task model)
eval_phase1.py   Evaluation entry point (Phase 1)
eval_phase2.py   Evaluation entry point (Phase 2)
```

## Setup

```
conda create -n blm5135 python=3.11 -y
conda activate blm5135
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Training

```
python train_phase1.py --config configs/phase1_object.yaml --seed 42
python train_phase1.py --config configs/phase1_event.yaml --seed 42
python train_phase1.py --config configs/phase1_attribute.yaml --seed 42
python train_phase2.py --config configs/phase2_unified.yaml --seed 42
```

## Evaluation

```
python eval_phase1.py --ckpt <phase1-checkpoint> --config <phase1-config>
python eval_phase2.py --ckpt <phase2-checkpoint> --config <phase2-config>
```

## Dataset

The dataset is not distributed with this repository. Place `dataset.json` and the `train/`, `val/`, `test/` image directories under `dataset/` following the layout referenced by the configuration files.
