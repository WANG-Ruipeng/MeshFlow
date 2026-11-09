# Source geometry audit

Completed: 26 retained training parents / 5122 faces, 411 existing conditions (208 C20 + 203 C40).
This report summarizes existing numerical results only. No model, training, OT, mesh repair or new mesh was run.

All 411 C are one complete-edge-connected component under source-index and exact-coordinate definitions. Branching C boundaries: 195/411; C internal nonmanifold edges: 88/411.
No exact zero-area or duplicate retained faces. Minimum angle below 5 degrees: 1367/5122 faces, distributed over all 26 parents. This diagnostic does not by itself make a face invalid.
The 517381 unique source-parent triangle pairs contain 157 verified proper penetration pairs (10 parents), 153 verified positive-area coplanar overlap pairs (6 parents), and 3 unresolved contact-subtype disagreements. There are 14 parents with either verified conflict.
Conditions with a verified conflict: CC 20/411, CF 101/411, FF 178/411; these subsets overlap and cannot be added. F with at least one component not connected to C by a complete edge: 112/411.

| Pool | Conditions | K min / median / max | C area fraction min / q25 / median / q75 / max |
|---|---:|---|---|
| C20 | 208 | 27 / 41 / 49 | 0.012605 / 0.129973 / 0.212014 / 0.312287 / 0.619121 |
| C40 | 203 | 53 / 82 / 98 | 0.093346 / 0.318601 / 0.420827 / 0.558478 / 0.875642 |

Full retained face IDs, component memberships, per-face areas/angles and all classifier details remain in the parent/condition JSON files. CSV summaries preserve all 26/411 registered cases.
See same_parent_area_ranges.csv for all 52 parent/ratio groups; fixed K can correspond to very different fractions of surface area.

## Limitations

- Static source geometry audit, not evaluation of generated outputs or proof of the cause of training failures.
- All C are edge-connected; one component does not establish semantic usefulness, compactness, a simple disk, or a user-like intended surface.
- Multiple complete-parent components and F components unattached by a complete edge can be valid separately assembled parts; counts are not rejection criteria.
- Five-degree minimum-angle diagnostic is inherited and not a validity threshold; zero area and skinny triangles are different.
- Verified conflict requires detector agreement. Three contact-subtype disagreements remain UNRESOLVED even though both classifiers label each illegal=False.
- Near-duplicate query metric is NA for all 411 conditions: no registered query bank exists; no query points were sampled.
- L is fixed sqrt(3)*2/.3762 for every parent and condition, not re-estimated from C or each parent.
- CC/CF/FF pair sums across conditions repeatedly count source pairs; they are not counts of independent defects.
- Exact coordinate connectivity uses stored float32 model coordinates without tolerance welding; source-index connectivity preserves original vertex IDs.
- The C20 and C40 source selector pools have known construction-policy differences; this descriptive audit does not turn their comparison into a pure ratio causal experiment.

## Reuse and execution

pair_cache_manifest.json binds all 26 full pair caches by SHA256 and records the face-ID convention; reuse it instead of rescanning for subsets.
data_audit.py is the resumable entry with --stage basic, intersections, or all. Existing results are immutable and inputs/source hashes must still match.
12 CPU synthetic tests passed; tests.stdout.txt and tests.xml contain the original run. Actual scoring source is preserved under source_snapshot/.
Source/input hashes before and after match. Basic stage 11.865 seconds; pair stage 82.972 seconds. Peak RSS 127568 KiB; one CPU process with two numerical threads.
