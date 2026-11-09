# Actual execution accounting

All registered call counts and protected file hashes pass. No model or geometry scorer was run by this tool.

| Item | Actual |
|---|---:|
| training_forwards | 1600 |
| training_backwards | 1600 |
| optimizer_updates | 200 |
| new_rollouts | 96 |
| new_generator_forwards | 4800 |
| new_D0_forwards | 96 |
| reused_RAW | 96 |
| reused_POST | 96 |
| reused_generator_or_D0_calls_this_experiment | 0 |
| new_scored_stages | 192 |
| reused_scored_stages | 192 |

Training update sum: 294.631 s; input wait sum: 0.092 s. Last trainer invocation wall time: 340.603 s.

Full actual phase durations, worker timings and memory scopes are in summary.json. 569 frozen files were independently rehashed.

PyTorch allocated/reserved counters exclude CUDA context; WDDM may not expose per-process memory. A 5GiB allocator cap is not a measured 6GiB process guarantee.

Device VRAM/compute include unrelated processes. Concurrent wall time must not be compared to earlier isolated wall time as a compute benchmark.

PHASE_END durations include subprocess startup and phase overhead. Training status wall_seconds covers the most recent trainer invocation if resumed; phase ledger and calls retain all actual invocations.

Cached RAW/POST receipts may contain historical new_* counters; none are added to this experiment model-call budget.
