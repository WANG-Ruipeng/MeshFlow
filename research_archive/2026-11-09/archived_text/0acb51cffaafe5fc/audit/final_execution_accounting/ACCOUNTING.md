# Final execution accounting

Status: PASS

Counts are derived from completed per-phase ledgers and cross-checked against frozen budgets.

| Item | Actual |
|---|---:|
| new_unique_updates | 42000 |
| new_physical_updates | 42000 |
| training_forward_calls | 336000 |
| training_backward_calls | 336000 |
| new_rollouts | 568 |
| reused_RAW_cells | 72 |
| registered_distinct_cells | 640 |
| actual_new_sampling_forwards | 28400 |
| scored_cells | 640 |

Training forward and backward counts are separate. Reused RAW cells incur no new rollout/forward here.

| Phase | Kind | Wall seconds | Max workers | Peak device-used MiB | Min free MiB |
|---|---|---:|---:|---:|---:|
| main | training | 20249.214 | 2 | 9317.0 | 6663.0 |
| course | training | 10885.870 | 2 | 9721.0 | 6259.0 |
| confirm | training | 11010.873 | 2 | 9654.0 | 6326.0 |
| main | sampling | 623.320 | 2 | 3959.0 | 12021.0 |
| course | sampling | 250.535 | 2 | 4067.0 | 11913.0 |
| confirm | sampling | 266.142 | 2 | 4115.0 | 11865.0 |

All recorded failure, unfinished call, replay, retry and OOM counters: {'training_failed_attempts': 0, 'training_failed_calls': 0, 'training_unfinished_calls': 0, 'training_resumes': 0, 'training_replayed_updates': 0, 'sampling_failed_calls': 0, 'sampling_unfinished_forward_calls': 0, 'sampling_unfinished_rollouts': 0, 'sampling_cell_retries': 0, 'sampling_dispatch_retry_markers': 0, 'sampling_failure_markers': 0, 'global_oom_downgrades': 0}

Source registries: [{'registry': 'source_hashes.json', 'schema': 'original97_frozen_scientific_sources', 'files': 97, 'status': 'PASS', 'failures': []}, {'registry': 'curriculum_additional_source_hashes.json', 'schema': 'curriculum_additional_source_freeze_v1', 'files': 8, 'status': 'PASS', 'failures': []}, {'registry': 'confirmation_additional_source_hashes.json', 'schema': 'recipe_confirmation_additional_sources_v1', 'files': 18, 'status': 'PASS', 'failures': []}, {'registry': 'confirmation_sampling_adapter_sources.json', 'schema': 'confirmation_sampling_adapter_source_freeze_v1', 'files': 3, 'status': 'PASS', 'failures': []}, {'registry': 'all_baselines_analysis_sources.json', 'schema': 'all_baselines_analysis_source_freeze_v1', 'files': 2, 'status': 'PASS', 'failures': []}, {'registry': 'all_baselines_analysis_v2_sources.json', 'schema': 'all_baselines_analysis_registered_binding_source_freeze_v2', 'files': 6, 'status': 'PASS', 'failures': []}]

Per-wave elapsed walls and resource samples include concurrency. Their sum is active-wave accounting, not end-to-end project elapsed time or an isolated algorithm benchmark.
Device-used includes desktop/context/other GPU processes; PyTorch allocated/reserved is a separate narrower scope.

Original97 sources, curriculum additions, confirmation additions, sampling adapter freeze, and v1/v2 analysis corrections remain independent registries; no original result/source was overwritten.

No quality tables, aliases, weights, model operations, scoring or rendering were used by this aggregation.

Default/Git statements are scoped observations, not global absence proofs.
