# Bench control transfer — dataset card

**Frozen data:** 64 TRAIN parents and 8 DEV parents, one existing mesh and one geometric `PLANAR_PATCH` condition per parent. No source mesh was repaired, welded, remeshed, deduplicated, or replaced.

| Quantity | TRAIN | DEV |
|---|---:|---:|
| Independent source parents | 64 | 8 |
| N: min / median / max | 128 / 190 / 256 | 154 / 179 / 244 |
| K: min / median / max | 2.0 / 2.0 / 51.0 | 2.0 / 2.5 / 16.0 |
| K/N (%): min / median / max | 0.78 / 1.19 / 25.00 | 0.89 / 1.41 / 10.26 |
| Counted C/M area (%): min / median / max | 1.30 / 10.83 / 29.08 | 4.35 / 13.24 / 31.37 |
| C open boundary edges | 429 | 41 |
| Verified source owner / alias / unsupported | 311 / 0 / 118 | 31 / 0 / 10 |
| Parents with zero verified source owners | 11 | 1 |

Area uses the sum of **all retained original source face areas**, including duplicate faces. It is not unique exterior surface area. Distribution summaries give equal weight to each parent; the edge totals pool edges.

![Frozen input distributions](dataset_overview.png)

## Source and split

The existing official `shapenet-cls.tar.gz` archive contains 560 Bench NPZ meshes. Of these, 88 have N128–256; 84 have a qualifying connected region, with 84 distinct exact normalized unordered-triangle geometry hashes. The 72 pilot parents were selected deterministically before model outputs. Remaining candidates are not used as a result-dependent replacement pool.

DEV contains all seven qualifying parents from the official Bench test split and one parent from the official training split, held out from this conditional adaptation. Results must distinguish these two exposure groups. TRAIN contains 64 other official training parents. Exact-geometry duplicates were grouped for parent selection; near-duplicate or original-family equivalence has not been exhaustively established.

This is a category-specific adaptation test using the official Bench prior. It does not show zero-shot transfer of the Chair checkpoint or arbitrary-category generalization.

## Input construction and preservation

The constructor selects an existing source-index-connected nearly planar region, preferring a broad horizontal surface above the object bottom. Qualification requires K≥2, nonzero constituent area, consistent internal C winding, no C nonmanifold edge, sufficient planar extent, and area/convex-hull compactness at least 0.60. Local adjacent normals must have dot product >0.999 and adjacent plane offset <1e-4 in normalized object units. No new faces are created.

The selected regions include broad boards, narrow bench slats, lower/bottom boards and connected frame-like patches with holes. 3 TRAIN regions use a non-horizontal fallback. These are geometric `PLANAR_PATCH` inputs, **not a set of verified seat-top semantic labels**. K≥2 alone does not imply a rectangle or a complete semantic part.

Each full object receives the official FP32 bounding-box normalization once. Pre-OT coordinates use division by 0.3762; model coordinates are exactly pre-OT ×2. C is never normalized separately. The case packs retain source vertices, source faces, canonical corner permutations, C source vertex IDs, original/model C topology, source face identities, and the untouched free complement.

All 72 full/source views and C-only principal-plane views were actually inspected by AI_PROXY reviewers. C-only views avoid confusing full-mesh occlusion with a missing input region. A separate finite-triangle pair audit on all selected C regions found **0 illegal pairs and 0 unresolved pairs**. This result certifies that bounded detector check only; it is not a quality certificate for the unknown completion or full source mesh.

## Source supervision and remaining defects

A source owner is an open C edge with exactly one source C face and one source free face, opposite edge winding, and a verified legal finite shared-edge contact. If one free face serves multiple C edges, only the lowest edge ID owns the seam slot and the others are aliases. No alias case occurred in this selected pool. Unsupported edges combine absent neighbors, incidence conflicts, winding mismatches and nonlegal/uncertain contacts; they should not all be interpreted as semantic “do not connect” labels.

| Retained source defect diagnostic | TRAIN | DEV |
|---|---:|---:|
| Same unordered source-vertex-ID face copies beyond first | 1334 | 85 |
| Parents containing those copies | 45/64 | 5/8 |
| Source degenerate faces (twice area ≤1e-12 model units²) | 190 | 12 |
| Parents containing those faces | 33/64 | 3/8 |

These original defects remain in the full target/free complement. Whole-source FF was **NOT_RUN** in this preparation; there is no claim that all other source faces are clean. The pilot isolates applying the existing method to another category, rather than combining category transfer with target cleaning.

## Fixed evaluation inputs

The panel contains four training parents and all eight DEV parents, one fixed C each, two preregistered Gaussian arrays each: 24 requests per model, shared by PLAIN and SEAM. The model and occupancy head receive C geometry/topology and requested N only. Source owner labels and complete source geometry are training/audit information, not inference inputs. Source reconstruction is not the open-ended generation success criterion.

## Files

- [Frozen manifest](manifest.json), [per-parent numerical summary](dataset_card_statistics.json), and [fixed request list](evaluation_requests.json).
- [C finite-pair audit](condition_geometry_audit.json), [source interface audit](source_interface_audit.json), [actual visual review](input_visual_review.json), and [review rows](input_visual_review.csv).
- Full/source sheets: `gallery/`; input-only sheets: `gallery_C_only/`; source-preserving NPZ cases: `cases/`.
- Constructor: `r2/pilots/bench_control_transfer_v1/data_prepare.py`; this summary: `dataset_card.py`.

Frozen manifest SHA256: `a2d6ed3e3afe608d4e7ef5cb6b596ebcee5be1665946bd3438c1ee14d16f3894`. This card and its plot were generated from saved inputs on CPU with zero model calls and without changing the manifest.
