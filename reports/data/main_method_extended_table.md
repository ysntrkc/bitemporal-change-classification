# Phase-2 BIT-only — extended metrics (main method only)

Mean ± std over 3 seeds (42, 1337, 2024). Macro-F1 is the headline metric; micro variants and precision/recall are reported here for the canonical Phase 2 configuration only (TTA + no-change gate, EMA, test split).

| family    | macro-F1 | micro-F1 | P (macro) | P (micro) | R (macro) | R (micro) | mAP |
|---|---:|---:|---:|---:|---:|---:|---:|
| object    | 0.3121 ± 0.0104 | 0.6945 ± 0.0172 | 0.3316 ± 0.0232 | 0.6431 ± 0.0460 | 0.3303 ± 0.0314 | 0.7579 ± 0.0255 | 0.2724 ± 0.0078 |
| event     | 0.2797 ± 0.0032 | 0.4155 ± 0.0022 | 0.2529 ± 0.0293 | 0.3578 ± 0.0282 | 0.3326 ± 0.0443 | 0.5027 ± 0.0595 | 0.2517 ± 0.0044 |
| attribute | 0.2538 ± 0.0062 | 0.4326 ± 0.0121 | 0.2225 ± 0.0178 | 0.3575 ± 0.0343 | 0.3180 ± 0.0362 | 0.5551 ± 0.0486 | 0.2405 ± 0.0093 |
| **mean**  | 0.2819 ± 0.0292 | 0.5142 ± 0.1564 | 0.2690 ± 0.0563 | 0.4528 ± 0.1648 | 0.3270 ± 0.0078 | 0.6052 ± 0.1348 | 0.2549 ± 0.0162 |
