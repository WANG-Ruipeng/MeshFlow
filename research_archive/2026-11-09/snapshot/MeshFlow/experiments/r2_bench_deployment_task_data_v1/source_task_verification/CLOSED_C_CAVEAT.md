# Closed C: existing Bench support, not a new model blocker

The staging label `CLOSED_C_EXISTING_SEAM_EDGE_ENTRY_REQUIRES_ADAPTER` is too broad for the actual Bench production path. Eight accepted source-resolution tasks have closed C, but closed C itself does **not** require a new model or a new input adapter.

- The generic legacy `r2/src/meshflow_control/seam/slot_data.py:37` rejects a C with no open boundary. Its `inference_layout` reaches that guard through `edge_features`. Calling this generic entry directly would fail.
- The existing Bench wrapper `r2/pilots/bench_parts_convergence_v1/data.py:106` returns an empty boundary array of shape `(0,3)`. At lines 139–159 it constructs an empty-occupancy layout: no SEAM slots, original C anchors unchanged, every remaining face is bulk, and the free-coordinate denominator is `9*(N-K)`. No occupancy-head evaluation is needed.
- The same file at lines 212–216 routes closed-C training to `_closed_sample`, whose documented ordinary bulk OT/HYBRID path already exists. No training or OT was run in this verification.
- Actual Bench sampling uses this wrapper: `bench_parts_convergence_v1/sample.py:132–137` skips the head when there are no edges; `bench_clean_multipart_seam_v1/sample.py:42,67` explicitly uses the empty-layout branch; `bench_relation_training_v1/eval_plan.py:90–98` reconstructs the cached production layout through the Bench wrapper.
- The completed independent layout audit already reconstructed two current TRAIN closed-C tasks and four unique frozen-panel closed-C tasks with exact target bijections. These are existing CPU/cache evidence, not a new forward or generation claim.

Use `CLOSED_C_EXISTING_BENCH_EMPTY_LAYOUT_SUPPORTED` as the descriptive status. This is an instruction to reuse the established Bench entry point, not to add a mechanism or to change the generic legacy entry.

All eight new accepted closed-C tasks still have source-resolution N outside 128–256. The Bench wrapper enforces that budget at `data.py:132`; therefore their actual model-ready status remains `BLOCKED_N_OUTSIDE_128_256`. Empty boundary removes the occupancy ambiguity, but does not certify target geometry, justify changing N, guarantee connectivity, or make these source-resolution tasks trainable. No old code/report or model was modified.
