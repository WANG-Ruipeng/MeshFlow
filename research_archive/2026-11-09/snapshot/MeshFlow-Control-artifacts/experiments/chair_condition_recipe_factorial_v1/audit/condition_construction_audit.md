# Condition construction audit: policy heterogeneity confirmed

**The strict same-candidate-selection-policy requirement is NOT satisfied.** The running study can be interpreted as an exploratory comparison of existing condition-pool recipes. MIX versus 40% does not identify the pure causal effect of condition face ratio. No frozen plan, source, training job or RAW was changed by this audit.

## Verified provenance

All 411 eligible imported task rows exactly match every field of their original Domain manifest rows. All 104 inherited20 tasks also match the original MVP N/K, known/free source-face IDs and start ID. The Domain source hash matches the import provenance; the MVP repair and topology source hashes match the original DATA_READY registration. Complete paths/hashes and source-line references are in condition_construction_audit.json.

| Fixed pool family | Tasks | MIX planned samples | Selection source |
|---|---:|---:|---|
| MVP_LEGACY20: train20_0..3, old_patch_id p0..p3 |104|12,015|Original MVP farthest-start procedure|
| DOMAIN_NEW20: train20_4..7 |104|11,985|Domain ranked-candidate procedure|
| DOMAIN_NEW40: READY train40_0..7 |203|24,000|Domain ranked-candidate procedure after20 selection|

All 26 parents have 4 legacy plus 4 new 20% tasks.25 parents have 8 READY 40% tasks;55740c07554a726257f40ff86fe708ff has 3. These are existing task pools, not new condition construction.

## Common geometry and actual selection differences

Both procedures use the same repair_v2 source geometry, complete source-edge adjacency, sorted-neighbor FIFO BFS, exact K, connected C and nonempty trusted original degree-2 interfaces. Given the same parent/start/K, their BFS growth rule is the same.

The original MVP repair constructor (data_repair_prepare.py:44-79) chooses a first face start farthest from the global face-centroid mean, then maximizes distance to previously attempted starts and accepts the first 4 valid patches. It additionally rejects repeated start centroids. It uses deterministic geometry, not an RNG selection seed.

Domain prepare.py:135-173 enumerates every source-face start, deduplicates face sets and ranks accepted candidates by minimum maximum Jaccard against anchors, maximum minimum centroid distance, maximum minimum area-fraction difference, then a fixed domain-hash tie and source ID. prepare.py:204-241 restores the old 4 twenty-percent patches, reserves evaluation conditions, selects 4 more twenty-percent patches, and only then selects up to 8 forty-percent patches. Thus 40% ranking uses already-selected20 regions as anchors; this is not merely a change in K.

The per-task manifest has no explicit policy or selection_seed field. Provenance is established from old_patch_id, the original manifest's global rules and the hash-verified generator source. Current training task-choice RNG selects from these frozen pools and does not regenerate their regions.

## Planned exposure, not completion

condition_construction_planned_exposure.csv contains 208 recipe/parent/ratio/family rows, including planned clean/augmented counts, effective free scalar supervision, and three 2000-update windows. Each recipe's registered full trajectory has 6000 updates and 48,000 samples. R1 rows describe its historical reused trajectory and authorize 0 new R1 updates. The audit never read live completion logs; none of these counts is claimed as actual completed exposure.

The shared MIX plan has 12,015 legacy 20% samples (6,002 clean;6,013 augmented),11,985 new 20% samples (5,998 clean;5,987 augmented), and 24,000 new 40% samples (12,000 clean;12,000 augmented). R2 and R4 have the same family exposure. R1 and R3 each use 48,000 Domain40-family samples.

## Interpretation and zero-extra-training response

- R3 minus R1 remains a HYBRID timing comparison on the same fixed 40% stream; R4 minus R2 remains a timing comparison on the same fixed MIX stream, subject to the experiment's other caveats.
- MIX versus 40% compares complete existing condition distributions: face ratio, historical selection family and selected-region distribution change together. Their interaction with timing is distribution-dependent, not identified as ratio-only.
- Shared evaluation inputs support practical recipe comparison; they do not remove the training-distribution confound.
- A later independent repeat using these same historical pools would confirm the complete recipe, not remove this confound.
- Preserve the current exploratory trajectories, prominently report the unmet strict common-policy requirement, and distinguish planned exposure from actual exposure. Do not swap tasks mid-run, hide this limitation, or rename the experiment as a pure-ratio ablation.

This read-only evidence audit performed 0 OT calls, 0 model calls and 0 training updates. Only its three audit deliverables were newly written.
