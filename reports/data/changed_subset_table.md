# Changed-subset macro-F1 (test split, EMA, TTA, no gate)

Two columns per family: full = all test pairs (1190 of which ~71% have a change), changed-only = restricted to is_change=1.
Mean ± std over 3 seeds.

| model | object full | object changed | event full | event changed | attr full | attr changed |
|---|---:|---:|---:|---:|---:|---:|
| P1 (default 0.5 + TTA) | 0.3025 ± 0.0111 | 0.3211 ± 0.0109 | 0.2719 ± 0.0138 | 0.2784 ± 0.0131 | 0.2575 ± 0.0051 | 0.2638 ± 0.0043 |
| P2 BIT-only (+TTA, no gate) | 0.3128 ± 0.0099 | 0.3224 ± 0.0033 | 0.2812 ± 0.0025 | 0.2870 ± 0.0018 | 0.2550 ± 0.0055 | 0.2619 ± 0.0055 |
