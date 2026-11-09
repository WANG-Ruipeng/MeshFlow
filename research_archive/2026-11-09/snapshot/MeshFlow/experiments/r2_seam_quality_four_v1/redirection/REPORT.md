# Scheme 4: original-radius seam-tip redirection

Status: COMPLETE. This bounded CPU change did not improve overall quality. It added one legal seam–bulk shared edge on one of 16 requests, with no new illegal CF/FF pairs, deleted faces or nonmanifold edges relative to the capacity-guard reference. It did not improve CF, FF or I, and did not add a request with nonzero or at least 30% reachable bulk area.

## Frozen scope and actual method

The inputs are the existing 16 C-only PREDICTED SEAM_SLOT_C200 outputs, saved single-D0 PRE_GEO, and original capacity-guard rejection/mapping records. There were zero generator/D0 calls, zero updates, no new noise/query, no changed occupancy and no radius sweep. The original radius is 0.04197095609837992 model units.

Only a retained, non-hard bulk corner rejected by capacity protection can nominate the already active owner seam tip of that blocked C edge. The final actual capacity cluster, including all its input corners, is assigned to the existing tip representative. Every moved corner must lie within the original radius measured from its own PRE_GEO position. C and seam anchors and the destination cluster cannot move. This freezes the final capacity map; it is not a rerun or reordering of the original greedy GEO_C clustering.

All affected source faces are checked for new degeneracy, duplicates, deleted/revived identities and new/worse nonmanifold edges. Every affected retained face is compared with every retained face before/after using the original verified triangle classifier. New illegal CF/FF or unresolved classifications reject the edit. Exact shared-index decode and original source-winding export follow. The unchanged full scorer ran on all 16 endpoints.

This differs from R10 arbitrary terminal partner attraction and the partial-edge 1/2/3.5%L test: the destination is an existing occupied owner tip, the displacement remains within the original radius, and actual clusters/identities are preserved. These restrictions test one local decoder operation. They neither infer bulk intent nor establish a physical upper bound on other decoders.

## Paired results

| Metric, macro mean unless count | Original GEO | Capacity guard | Tip redirect |
|---|---:|---:|---:|
| CF affected free faces | 1.70357% | 1.66101% | 1.66101% |
| FF affected free faces | 14.56293% | 15.34847% | 15.34847% |
| I, frozen C-query conflict coverage | 3.30200% | 3.29895% | 3.29895% |
| Small-angle free faces | 18.09988% | 18.12708% | 18.12708% |
| Legal C/F shared edges, total | 175 | 192 | 192 |
| C/F nonmanifold edges, total | 18 | 0 | 0 |
| Free-only reachable bulk area | 49.24826% | 48.89648% | 48.90113% |
| Deleted free faces | 32.2500 | 31.9375 | 31.9375 |

The reachable-bulk metric uses the existing legal free-only graph seeded by seam faces attached legally to C. It excludes propagation through C. It is distinct from direction-consistent reachable total free area and does not certify a complete seam.

The sole accepted request is 73b369b9/n3: input corner 262 of source slot 87 is assigned to owner seam slot 41 tip corner 125. The actual PRE_GEO displacement is 0.041214196654182016, below the fixed radius. One affected face and 282 before/after pair tests passed. A second proposed corner of the same face would collapse it and was rejected.

On that request, legal seam–bulk edges increase 3→4; their total length increases 2.87828→6.53101 model units. Free-only reached bulk faces increase 19→20 and bulk reachable area 28.87880%→28.95322%, still below 30%. The macro increase is only 0.004651 percentage points. The overall direction-consistent free-area fraction changes 30.00431%→30.07654%; these are different denominators and must not be substituted for bulk reach. No new legal C/F edge is created.

Relative to the original GEO, the capacity guard's FF cost remains: +0.78554 percentage points. The redirection did not repair that cost. 3f/n2 retains its capacity I tail of 42.48047%; no route-wide success should be inferred from the one new edge.

## Rejection accounting

| Candidate disposition | Count |
|---|---:|
| Cluster exceeds original radius | 20 |
| New exact degeneracy | 2 |
| Accepted | 1 |
| Source face deleted by capacity | 1 |
| No occupied owner | 1 |
| Source is not non-hard bulk | 1 |

These are all logged attempts, including early ineligible events, not 26 independent requests. The full per-request candidate files retain actual displacement, cluster members, owner identity and every rejection. No candidate reached a new-CF/FF rejection after the earlier filters; this does not show that such conflicts are generally absent.

## Actual visual inspection and limits

The assistant opened the 73/n3 changed-request image and 3f/n2 tail-control image, each with three methods, two same-scale whole views and two finite central cuts. This is nonblind AI_PROXY inspection, not human review. No clear whole-shape or gap improvement is discernible. In 73/n3 a lower gray layer remains separated below blue C in the side cut. The 3f/n2 chair silhouette looks coherent, but these views do not reliably reveal its measured coplanar conflict. Neither appearance certifies watertightness or no overlap.

All other 15 tip-redirect endpoint arrays equal the capacity endpoint exactly; their byte/array evidence is in capacity_geometry_identity.csv. Only the two opened composites receive new visual observations. Unviewed identical cases have no fabricated fresh rating.

## Interpretation and recovery

Retain this as a safe but very limited local engineering result. It does not support adopting tip redirection as a quality upgrade or increasing its radius. The current constraints leave most destinations out of reach; the experiment does not demonstrate that bulk intended to connect to those tips. Keep the existing capacity and original GEO comparisons visible.

Outputs: per_output.csv (48 rows), paired_per_noise.csv, per_parent.csv, both source-identity edge/pair ledgers, per-request candidate ledgers, 16 indexed POSTs, original full scores, gallery/actual_visual_review.csv, execution_ledger.json and all input/source hashes in registration.json. The two rendered images are linked from gallery/index.html.

Recovery of this completed branch is verification-only:

```text
python -B -m r2.pilots.seam_quality_four_v1.redirection.closeout
```

It checks frozen sources, protected inputs and completed artifact hashes, then exits without re-scoring or model work. The original run --stage all entry is retained as execution provenance; do not use it to overwrite a completed score-timing receipt. No default, commit or push is changed by this branch.
