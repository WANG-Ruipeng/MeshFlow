# Full trajectory-arm learning diagnostics

All 1000 completed optimizer steps. No model calls or recipe changes.

The first 96 and last 96 updates each cover every fixed 48-step auxiliary cycle position twice. All paired Xt/u/target/queue/parent/time identities match exactly.

| Window / t | Steps | FM mean | Auxiliary MSE | Weighted auxiliary / FM (ratio of means) | Clip fraction |
|---|---:|---:|---:|---:|---:|
| first96 / all | 96 | 0.122595 | 14.869 | 12.1286 | 0.8958 |
| first96 / 0.6 | 48 | 0.121901 | 6.17271 | 5.0637 | 0.8125 |
| first96 / 0.8 | 48 | 0.123288 | 23.5652 | 19.114 | 0.9792 |
| last96 / all | 96 | 0.118728 | 1.58609 | 1.33591 | 0.6458 |
| last96 / 0.6 | 48 | 0.117345 | 0.531832 | 0.453219 | 0.3750 |
| last96 / 0.8 | 48 | 0.12011 | 2.64035 | 2.19828 | 0.9167 |
| all1000 / all | 1000 | 0.12323 | 5.19569 | 4.21624 | 0.8550 |
| all1000 / 0.6 | 504 | 0.124121 | 2.02812 | 1.63399 | 0.7599 |
| all1000 / 0.8 | 496 | 0.122325 | 8.41435 | 6.87866 | 0.9516 |

Auxiliary MSE decreased at 48/48 matched cycle positions.

Fitting the immutable START replay targets is distinct from generated-control efficacy. Loss magnitude does not bound gradient magnitude. Hook statistics use only recorded step1/every25 observations. No DEV/CONFIRM metrics were read and no recipe decision is made.
