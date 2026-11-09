# Independent training execution and curve audit

**PASS.** Both fixed arms completed200 optimizer updates,1600 training forwards and1600 backwards, with no failed/pending calls or replay. The final global step is12200. Each arm saw the same16 parents exactly100 times, in batches of8 distinct parents with4 unaugmented and4 width-augmented examples. Parent, width, time and random seed identities match across arms; target N differs, so realized noise dimensions and OT assignments need not match.

The initialization receipts match the same FULL_S6000 live generator, AdamW moments/hyperparameters and RNG. Each final checkpoint was read exactly once on CPU using mmap. All stored generator/optimizer/RNG hashes match, every AdamW parameter state is at12200, and both final generator hashes differ from their common initialization. No model was constructed or executed by this audit.

| Recorded quantity | ORIGINAL_C200 | DEDUP_C200 |
|---|---:|---:|
| Free scalar exposures | 2,742,300 | 2,445,300 |
| Mean training FM, updates1–50 | 0.106380 | 0.133058 |
| Mean training FM, updates51–100 | 0.109660 | 0.118943 |
| Mean training FM, updates101–150 | 0.105458 | 0.110610 |
| Mean training FM, updates151–200 | 0.108474 | 0.114395 |
| Recorded training wall seconds | 340.366 | 357.616 |
| Failed / unresolved model calls | 0 / 0 | 0 / 0 |

These curves do not rank free-generation quality. Target multiplicity, N, OT dimensions and the free-coordinate denominator differ. Detached endpoint error is `(1-t)^2` times the normal training velocity error, reconstructed from logged loss contributions; it is not a rollout or an independent fixed-input validation loss. Per-time-bin curves preserve this distinction. DEDUP begins with higher supervised FM and later windows are lower than its first window, but this alone is not evidence of improved meshes.

The training wall times are not an isolated speed comparison: DEDUP partially overlapped with ORIGINAL endpoint sampling. Per-update PyTorch allocated/reserved peaks and device-wide observer peaks are retained separately. WSL did not expose matching-PID process VRAM, so process peaks remain unavailable rather than being inferred from allocator memory or device totals.

The saved duplicate identities confirm330 opposite-winding source pairs with330 identical canonical vertex rows. The original author may have intended two-sided thin surfaces; intent remains UNKNOWN. Deduplication preserves the unoriented geometric surface union while changing target multiplicity and source sidedness. It should not be described as deleting330 random corrupt triangles.

Artifacts:

- `completion.json`: independent execution conclusion and all audited source hashes.
- `per_update.csv`, `per_sample_detached.csv`, `per_parent_exposure.csv`: complete update/sample/exposure records.
- `learning_curve_50_updates.csv`, `time_bin_learning_curve.csv`, `learning_curve.png`: four fixed50-update windows and five time bins.
- `checkpoint_audit.csv`, `initializations.json`, `ot_cache_audit.csv`: checkpoint, initial-state and cache proofs.
- `arm_execution_audit.json`: call budgets, durations, resource measurements and failure counts.
- `input_duplicate_winding_audit.json`: independent duplicate-identity interpretation.

All frozen training code and protected inputs were rehashed. This audit performed0 new OT solves,0 model calls and0 geometric scoring calls. The old ledger's6001-based counter was corrected before registration and before training; it was not a failed model attempt. The rendered learning-curve PNG was inspected for label/readability.
