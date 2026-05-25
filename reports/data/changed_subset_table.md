# Changed-subset macro-F1 (test split, EMA, TTA [+ gate for P2])

Two columns per family: full = all test pairs (1190 of which ~71% have a change), changed-only = restricted to is_change=1.
Mean ± std over 3 seeds.

| model | object full | object changed | event full | event changed | attr full | attr changed |
|---|---:|---:|---:|---:|---:|---:|
| P1 (default 0.5 + TTA) | 0.3025 ± 0.0111 | 0.3211 ± 0.0109 | 0.2719 ± 0.0138 | 0.2784 ± 0.0131 | 0.2575 ± 0.0051 | 0.2638 ± 0.0043 |
| P2 BIT-only (+TTA + gate) | 0.3121 ± 0.0104 | 0.3212 ± 0.0037 | 0.2797 ± 0.0032 | 0.2849 ± 0.0027 | 0.2538 ± 0.0062 | 0.2599 ± 0.0061 |
