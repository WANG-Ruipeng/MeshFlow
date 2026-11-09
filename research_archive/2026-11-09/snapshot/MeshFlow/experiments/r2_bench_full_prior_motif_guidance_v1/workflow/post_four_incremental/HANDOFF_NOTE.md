# CPU scheduling handoff

The original 184 selected CPU cascades remain unchanged. The old even worker (PID 6050) completed 15 outputs and the old odd worker (PID 5564) completed 26 outputs before handoff. The remaining 143 outputs were partitioned into exclusive lists of 39, 38, 33, and 33 tasks. No model invocation, D0 replay, score replay, geometry change, or scientific parameter change was added.

Each old worker was identified by its exact command and process creation time. A prospective boundary was first checked without signals. Only when every task with an existing BASE cache or CPU directory had complete BASE and final cascade receipts was that owned process stopped briefly. The directories were rechecked while stopped; a safe handoff receipt was written before SIGTERM followed by SIGCONT. An incomplete cascade was never terminated or restarted. The observer return code -15 records this authorized handoff and is not an output failure. The first observer's delayed reap caused a process-wait timeout after successful termination; its preserved evidence and subsequent process absence are explained in `../post_four_v1/operator_revision_01.json`.

The new workers were launched incrementally, with actual CPU/RAM observations before each additional worker. Worker PIDs are 8251, 8315, 8543, and 8594; their observer PIDs are 8244, 8308, 8534, and 8585. First launch was 05:07:01 UTC; the fourth was 05:09:50 UTC on 2026-10-29. Before the fourth launch, measured host CPU utilization was 13.4%, available RAM was 65.16 GB, and comparable process RSS was at most 165.45 MB. Each scientific worker retains one CPU thread.

The process-local barrier cache authenticates the existing full verification evidence using complete hashes of 403 RAW/producer-receipt/tasks/lock/requests files, then protects their immutable size/mtime state. Every original per-output RAW/D0/cache/CPU identity and result hash check remains active. The frozen wrapper performs one final global full RAW verification after all workers finish; each worker does not repeat that final audit. Completed original outputs are verified and excluded, not re-scored.

Evidence paths relative to this experiment:

- `workflow/post_four_v1/handoff_registration.json`, `handoff_0.json`, `handoff_1.json`, and `bounded_poll_waves.jsonl` identify and record the original handoffs.
- `workflow/post_four_incremental/even/plan.json` and `odd/plan.json` freeze the exact original parity ownership and four disjoint task lists.
- Each parity directory contains `launch_<n>.json`, `worker_entry_<n>.json`, and `worker_<n>/progress.jsonl` for resources, process identity, and actual per-task wall/CPU time.
- `workflow/post_four_incremental/resource_throughput_snapshot_01.json` records an actual five-second CPU/RSS sample and 66/184 completed outputs at 05:13:29 UTC.
- Final `complete.json` and `final_global_full_RAW_verification.json` are written only after all 184 complete identities pass; they are pending at this note's creation.

Orchestration source SHA-256 values are unchanged: `post_handoff_four.py` = `3e038eec473f8bdf33485b918f67ac16d1c7ba9307fbb9479afd8e833341a9f5`; `post_four_incremental.py` = `e51a7f294bed97b127a5d8b362fea15193934203236d0896c838b60e83e2e466`. Frozen preparation, evaluation, scorer, and geometry sources were not edited. Concurrent mixed-task throughput is reported as such; it is not an isolated compute speed benchmark.
