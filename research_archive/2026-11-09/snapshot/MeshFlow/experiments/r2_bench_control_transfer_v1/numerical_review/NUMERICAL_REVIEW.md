# Independent Bench numerical review

This review only reads saved per-output tables, indexed meshes, legal-edge classifications and GEO face-cleanup records. It adds no model calls, distance/intersection scans, noise, queries or metric definitions.

The numerical tables agree with independent paired recomputation. SEAM establishes C-side indexed interfaces, but these outputs do not show a legal shared-edge path from seam into bulk. FF and C conflict coverage increase, while the retained-face small-angle fraction decreases. This is a bounded interface result, not an overall quality upgrade.

| POST group | Requests | C/F coverage PLAIN→SEAM | FF % | I % | Small-angle % | Retained free faces |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 8 | 1/8 → 8/8 | 15.551 → 21.170 | 16.083 → 17.981 | 43.738 → 45.807 | 65.125 → 69.500 |
| DEV | 16 | 0/16 → 16/16 | 20.009 → 21.577 | 6.522 → 10.864 | 53.727 → 43.334 | 71.062 → 75.312 |
| DEV_OFFICIAL_TEST | 14 | 0/14 → 14/14 | 21.376 → 23.475 | 7.453 → 12.416 | 53.137 → 40.851 | 67.786 → 73.071 |
| DEV_PRETRAIN_TRAIN | 2 | 0/2 → 2/2 | 10.440 → 8.292 | 0.000 → 0.000 | 57.856 → 60.713 | 94.000 → 91.000 |
| ALL | 24 | 1/24 → 24/24 | 18.523 → 21.441 | 9.709 → 13.236 | 50.397 → 44.158 | 69.083 → 73.375 |

## Graph zero: verified, not a missing or inapplicable field

Independently reconstructed adjacency from all 48 saved SEAM RAW/POST edge tables. Every edge incidence matches the saved indexed faces; every retained role matches raw_seam_slot_mask[raw_slot_ids]; source IDs and retained masks agree. All per-face seed/reach flags and positive bulk-area denominators match saved records.
POST: 0/24 outputs reach any bulk face; legal seam/bulk edges total 0. Non-legal indexed seam/bulk two-face edges total 0. RAW: legal seam/bulk edges total 0. Thus zero reach is a genuine zero under the existing legal indexed-edge graph, not NA coerced to zero. PLAIN seam-specific quantities remain NA.

This graph does not rule out near contact, shared vertices without edges, or visual continuity. C/F seed faces are not by themselves evidence that the bulk is incorporated. Traversal through C is deliberately excluded in the unchanged definition.

## Cleanup loss accounting

| POST group/arm | Requested free mean | Retained free mean | Deleted mean | Deletion reasons (total faces) |
|---|---:|---:|---:|---|
| TRAIN/PLAIN | 179.750 | 65.125 | 114.625 | duplicate_C: 4; duplicate_free: 131; exact_repeated_vertex: 782 |
| TRAIN/SEAM | 179.750 | 69.500 | 110.250 | duplicate_C: 4; duplicate_free: 113; exact_repeated_vertex: 765 |
| DEV/PLAIN | 181.625 | 71.062 | 110.562 | duplicate_free: 204; exact_repeated_vertex: 1565 |
| DEV/SEAM | 181.625 | 75.312 | 106.312 | duplicate_free: 227; exact_repeated_vertex: 1474 |
| DEV_OFFICIAL_TEST/PLAIN | 175.857 | 67.786 | 108.071 | duplicate_free: 185; exact_repeated_vertex: 1328 |
| DEV_OFFICIAL_TEST/SEAM | 175.857 | 73.071 | 102.786 | duplicate_free: 202; exact_repeated_vertex: 1237 |
| DEV_PRETRAIN_TRAIN/PLAIN | 222.000 | 94.000 | 128.000 | duplicate_free: 19; exact_repeated_vertex: 237 |
| DEV_PRETRAIN_TRAIN/SEAM | 222.000 | 91.000 | 131.000 | duplicate_free: 25; exact_repeated_vertex: 237 |
| ALL/PLAIN | 181.000 | 69.083 | 111.917 | duplicate_C: 4; duplicate_free: 335; exact_repeated_vertex: 2347 |
| ALL/SEAM | 181.000 | 73.375 | 107.625 | duplicate_C: 4; duplicate_free: 340; exact_repeated_vertex: 2239 |

The reason exact_repeated_vertex means a triangle has repeated vertex indices after the existing GEO consolidation. It is not a measurement of its original RAW area, nor proof that D0 alone created the problem. duplicate_free means repeated retained free-face vertex triples. These records establish why cleanup deleted faces; without a separate intervention they do not identify how much was caused by generator geometry, D0 displacement, or consolidation. Lower small-angle fraction on the surviving faces must be read alongside the much smaller surviving set. Retained surface area and face counts remain in the paired tables.

## Per-parent paired direction

Counts below are lower / equal / higher SEAM than PLAIN after averaging the two fixed noises within each parent. For FF, CF, I and small-angle fraction lower is favorable; no significance or generalization claim is attached to these small counts.

| Group | Parents | FF | CF | I | Small-angle |
|---|---:|---:|---:|---:|---:|
| TRAIN | 4 | 0/1/3 | 2/1/1 | 1/0/3 | 1/0/3 |
| DEV | 8 | 3/0/5 | 3/2/3 | 1/2/5 | 7/0/1 |
| DEV_OFFICIAL_TEST | 7 | 2/0/5 | 3/1/3 | 1/1/5 | 7/0/0 |
| DEV_PRETRAIN_TRAIN | 1 | 1/0/0 | 0/1/0 | 0/1/0 | 0/0/1 |
| ALL | 12 | 3/1/8 | 5/3/4 | 2/2/8 | 8/0/4 |

DEV contains eight condition-training-held-out parents: seven official-test parents (14 outputs), and one official-training parent (two outputs). These are reported separately. Official split metadata is not an independent audit of every possible upstream pretraining source.

No unique-source reconstruction ranking is introduced. Saved-head/source-owner agreement remains a separate exploratory diagnostic. Actual shape completeness, floating fragments and use of the supplied patch require the separately recorded image review; this numerical review does not invent those labels.
