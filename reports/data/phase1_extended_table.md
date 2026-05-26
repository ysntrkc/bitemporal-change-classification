# Phase-1 canonical — extended metrics (main per-family method)

Mean ± std over 3 seeds (42, 1337, 2024). Macro-F1 is the headline metric; micro variants and precision/recall are reported here for the canonical Phase 1 configuration only (TTA + default threshold 0.5, EMA, test split).

| family    | macro-F1 | micro-F1 | P (macro) | P (micro) | R (macro) | R (micro) | mAP |
|---|---:|---:|---:|---:|---:|---:|---:|
| object    | 0.3025 ± 0.0111 | 0.7024 ± 0.0092 | 0.3129 ± 0.0185 | 0.6637 ± 0.0155 | 0.3165 ± 0.0049 | 0.7460 ± 0.0054 | 0.2791 ± 0.0127 |
| event     | 0.2719 ± 0.0138 | 0.4298 ± 0.0066 | 0.2308 ± 0.0272 | 0.3439 ± 0.0236 | 0.3658 ± 0.0345 | 0.5831 ± 0.0780 | 0.2522 ± 0.0068 |
| attribute | 0.2575 ± 0.0051 | 0.4375 ± 0.0081 | 0.2366 ± 0.0184 | 0.3652 ± 0.0388 | 0.3173 ± 0.0297 | 0.5560 ± 0.0610 | 0.2378 ± 0.0153 |
| **mean**  | 0.2773 ± 0.0230 | 0.5232 ± 0.1552 | 0.2601 ± 0.0458 | 0.4576 ± 0.1788 | 0.3332 ± 0.0282 | 0.6284 ± 0.1028 | 0.2563 ± 0.0210 |
