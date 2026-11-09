# Severe parent I/E costs (existing-result extraction)

**7 original joint-gate parent/comparison rows carry a severe-cost warning; these must be reported explicitly even if a numeric gate were to pass.**

All deltas below are candidate minus reference, in percentage points (pp). Positive I = worse conflict; negative E = worse exposed retention. Original severe thresholds are I increase >10 pp or E decrease >10 pp. Bounds are the original saved PARTIAL-aware bounds. CONFIRMED means the whole bound exceeds the adverse threshold; possible/unresolved flags remain separate.

OLD and FRESH each average the two registered noises within each parent (eight parents per panel). ALL, when shown as a supplementary CSV result, averages four registered noises within the same eight parents; it is not another independent parent panel.

The original `severe_cost_caveat` is a separate warning, not an extra undocumented numeric veto. Neither original `numeric_status` nor overall `status` incorporates it. Therefore numeric PASS alone does not establish a cost-free or acceptable promotion. Original definitions and statuses are unchanged.

|Panel|Candidate|Reference|Parent UID|I delta [lower, upper] pp|I gate|E delta [lower, upper] pp|E gate|Risk|
|---|---|---|---|---:|---|---:|---|---|
|FRESH|G1000|START|534b0e6e2c53a8e1a7cdfd3874ed3723|+3.8818 [+3.8818, +3.8818]|PASS|-13.9909 [-13.9909, -13.9909]|FAIL|CONFIRMED|
|FRESH|G1000|T1000|341bcae1e7f366a233d93f57729688db|+6.9092 [+6.9092, +6.9092]|PASS|-10.9478 [-10.9478, -10.9478]|FAIL|CONFIRMED|
|FRESH|G1000|T1000|534b0e6e2c53a8e1a7cdfd3874ed3723|+3.8086 [+3.8086, +3.8086]|PASS|-21.3135 [-21.3135, -21.3135]|FAIL|CONFIRMED|
|FRESH|G500|T500|534b0e6e2c53a8e1a7cdfd3874ed3723|+1.0498 [+1.0498, +1.0498]|PASS|-12.8334 [-12.8334, -12.8334]|FAIL|CONFIRMED|
|FRESH|HIST_CAPPED500|START|f59627a077dcc2566c63b8b48495261a|+20.6787 [+20.6787, +20.6787]|FAIL|+7.9390 [+7.9390, +7.9390]|PASS|CONFIRMED|
|FRESH|L500|T500|534b0e6e2c53a8e1a7cdfd3874ed3723|+1.5137 [+1.5137, +1.5137]|PASS|-10.3674 [-10.3674, -10.3674]|FAIL|CONFIRMED|
|OLD|HIST_CAPPED500|HIST_TEACHER500|534b0e6e2c53a8e1a7cdfd3874ed3723|+13.9160 [+13.9160, +13.9160]|FAIL|+9.8138 [+9.8138, +9.8138]|PASS|CONFIRMED|

The accompanying [CSV](severe_parent_costs.csv) also retains 3 supplementary flagged rows already present in the frozen paired CSV (ALL and control-versus-START comparisons). Each row identifies whether its source is the original joint gate or the paired CSV. No new scores or point samples were generated.

Original joint-gate parent I/E values and bounds were cross-checked against the paired CSV: 960 scalar comparisons passed (absolute difference <1e-12).

Resource evidence:

- GPU: NVIDIA GeForce RTX 5080, 16303 MiB total. Maximum monitored device-wide usage: 6320 MiB. Own-process peak: NA under WSL/WDDM; device usage includes unrelated desktop processes.
- CPU: 24 logical cores; registered two threads per worker. Initial Windows CPU load [11]%; free RAM 68.326 GiB. WSL total 62.515 GiB; available RAM across recorded dispatch snapshots 58.018 to 61.138 GiB. These are snapshots, not continuous CPU/RAM peak monitoring.
- GPU lock: 6501.261100 seconds (108.354 minutes), 17 closed intervals, one worker; limit 14400 seconds. Includes the preserved 0-call loader failure, loading, saving and in-worker CPU waiting.
- Stage-specific observed peaks and lock duration are in [resource_costs.csv](resource_costs.csv). Sampled peaks are not continuous hardware maxima; no process/allocator equivalence is claimed.

|Stage|Device peak MiB|GPU utilization max %|Min free MiB|Lock seconds|
|---|---:|---:|---:|---:|
|preflight_real_1000_BOTH|6119|17|9861|30.189|
|sample_G1000_1000_BOTH|3356|24|12624|107.794|
|sample_G500_1000_BOTH|3358|26|12622|95.299|
|sample_HIST_CAPPED500_1000_FRESH|3362|26|12617|54.828|
|sample_HIST_TEACHER500_1000_FRESH|3349|26|12630|53.115|
|sample_L1000_1000_BOTH|3344|26|12636|109.933|
|sample_L500_1000_BOTH|3357|23|12623|104.897|
|sample_START_1000_FRESH|3355|23|12625|60.977|
|sample_T1000_1000_BOTH|3342|26|12638|102.960|
|sample_T500_1000_BOTH|3366|23|12614|103.838|
|train_G_1000_BOTH|6203|34|9777|944.057|
|train_G_500_BOTH|6215|32|9765|936.356|
|train_L_1000_BOTH|6200|39|9780|933.357|
|train_L_500_BOTH|6211|30|9769|949.247|
|train_T_1000_BOTH|6200|30|9780|946.665|
|train_T_500_BOTH|6320|35|9660|967.750|

Extraction source and all input SHA256 bindings are retained under `audit/severe_parent_costs_receipt.json`. The extraction imports no scoring, model, mesh, or training module.
