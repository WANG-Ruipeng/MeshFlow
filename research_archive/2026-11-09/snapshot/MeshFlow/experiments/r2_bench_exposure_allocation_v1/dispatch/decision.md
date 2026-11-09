# Measured A/B training concurrency

Keep the two already-authorized workers running. The observed interval has adequate memory headroom and higher aggregate update throughput. No extra arm or probe was launched by this observer.

| Observation | Actual value |
| --- | ---: |
| GPU | RTX 5080, 16,303 MiB reported total |
| Training-before baseline used | 1,771 MiB |
| A-only observed aggregate peak | 5,329 MiB |
| A-only minimum free | 10,651 MiB |
| A/B observed aggregate peak | 8,896 MiB |
| A/B minimum free | 7,084 MiB |
| A/B utilization, full 44-sample observation | Mean 57.80%, peak 67% |
| A/B stable interval after B step25 | 30.027 seconds |
| A progress in stable interval | 83 to 99: 16 updates, 0.533 updates/s |
| B progress in stable interval | 25 to 42: 17 updates, 0.566 updates/s |
| Combined stable throughput | 1.099 updates/s |
| A-only first25 estimated active throughput | 0.702 updates/s |
| Short-interval aggregate ratio | 1.566 times the early A-only estimate |

The minimum observed free memory remains above the 2,048 MiB safety margin. Both workers were still training, with no failure or OOM recorded in the observation. Their individual updates slowed under contention, but the combined progress increased. Low CPU load and available RAM at dispatch did not prohibit the second worker. There are only two authorized arms; this result does not authorize a third task.

WDDM reports process memory as NA. All peaks above are **aggregate device memory**, not measured per-process peaks. The 3,558 MiB A increment over the baseline is an estimate, not an exact process allocation. The conservative second-worker calculation used the entire A-only 5,329 MiB aggregate peak: floor((10,651 - 2,048) / (1.2 * 5,329)) = 1 additional slot, capped at the one remaining authorized arm.

The first external A observer began at step12; native training records at steps1 and5 supplement the early boundary. The A-only throughput includes the first25 update warmup and excludes model-loading/checkpoint IO; the A/B estimate is a 30-second progress interval. These are useful scheduling observations, **not an isolated full-run benchmark or a guarantee of later peaks**. Continue normal resource and failure monitoring, especially around checkpoint IO and independent user GPU activity. Sampling and D0 need their own resource decisions.

Evidence: `A_first25_gpu_observations.json`, `AB_steady_observations.json`, and the two training status/log/resource records. This file records a scheduling conclusion; it does not mutate running jobs, frozen code, RNG, scientific budgets or evaluation requests.
