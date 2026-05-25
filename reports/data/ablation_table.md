# Ablation table: macro-F1 on test split (EMA weights)

All rows: mean ± std over 3 seeds (42, 1337, 2024). DBLoss row is object-only (heaviest 270:1 imbalance).

| variant | object | event | attribute | **mean** | notes |
|---|---:|---:|---:|---:|---|
| A1: ResNet-50 + ASL  (backbone swap) | 0.2647 ± 0.0043 | 0.2489 ± 0.0088 | 0.2281 ± 0.0043 | **0.2472 ± 0.0184** | 3-seed mean ± std |
| P1: ConvNeXt-V2 + ASL, no TTA | 0.2971 ± 0.0078 | 0.2670 ± 0.0155 | 0.2499 ± 0.0062 | **0.2713 ± 0.0239** | 3-seed mean ± std |
| P1: ConvNeXt-V2 + ASL, +TTA  (canonical) | 0.3025 ± 0.0111 | 0.2719 ± 0.0138 | 0.2575 ± 0.0051 | **0.2773 ± 0.0230** | 3-seed mean ± std |
| P1: canonical, +TTA, +gate | 0.3005 ± 0.0102 | 0.2682 ± 0.0122 | 0.2558 ± 0.0063 | **0.2748 ± 0.0230** | multiplicative gate from head_nochg |
| P1: DBLoss, default 0.5 +TTA  (uncalibrated) | 0.2656 ± 0.0153 | — | — | **—** | object only (270:1 imbalance) |
| P1: DBLoss, tuned thr +TTA  (calibrated) | 0.2849 ± 0.0075 | — | — | **—** | object only (270:1 imbalance) |
| P2: BIT, linear heads, fixed weights  (canonical) | 0.3121 ± 0.0104 | 0.2797 ± 0.0032 | 0.2538 ± 0.0062 | **0.2819 ± 0.0292** | 3-seed mean ± std |
| P2: no BIT, linear heads, fixed weights | 0.2846 ± 0.0230 | 0.2766 ± 0.0074 | 0.2585 ± 0.0079 | **0.2732 ± 0.0134** | fusion ablation |
| P2: BIT, Q2L heads, UWL  (full stack) | 0.1730 ± 0.0076 | 0.2266 ± 0.0153 | 0.1443 ± 0.0043 | **0.1813 ± 0.0418** | head + loss ablation |
