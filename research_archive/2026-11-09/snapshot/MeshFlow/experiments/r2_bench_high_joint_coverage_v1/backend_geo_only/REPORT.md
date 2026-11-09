# High-N RAW → GEO_ONLY backend pilot

Four fixed NATURAL_HIGH HIGH_TRAIN requests; no new model calls, D0, noise, queries or training. This is a backend comparison on two familiar high-N parents, not generalization.

| Stage | FF | CF | free faces | legal CF edges | small-angle % | isolated 1face | nonmanifold | primary I |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RAW | 68.5105 | 10.8010 | 319.0000 | 0.0000 | 53.4537 | 319.0000 | 0.0000 | NA: LOW_SUPPORT |
| ORIGINAL_D0_BASE | 1.5838 | 0.0000 | 51.0000 | 1.0000 | 31.4143 | 32.7500 | 0.0000 | NA: LOW_SUPPORT |
| ORIGINAL_D0_FINAL | 1.5838 | 0.0000 | 51.0000 | 1.0000 | 31.4143 | 30.5000 | 0.0000 | NA: LOW_SUPPORT |
| GEO_BASE | 30.8340 | 5.6375 | 181.0000 | 16.0000 | 53.5843 | 94.7500 | 10.5000 | NA: LOW_SUPPORT |
| GEO_FINAL | 22.6752 | 4.2601 | 181.0000 | 17.0000 | 53.1641 | 45.5000 | 10.5000 | NA: LOW_SUPPORT |

Both fixed C query sets have zero interior samples (required at least 32). Raw legacy I numbers are retained in full177 tables; they are not an applicable primary comparison. The four-request mean equals parent-equal mean here because each parent has two noises.

| Request | original FINAL free / FF% / CF% | GEO FINAL free / FF% / CF% |
|---|---:|---:|
| 28b9f2/n0 | 24 / 0.000 / 0.000 | 119 / 15.966 / 0.000 |
| 28b9f2/n1 | 21 / 0.000 / 0.000 | 132 / 7.576 / 0.000 |
| abf250/n0 | 82 / 2.439 / 0.000 | 242 / 26.033 / 5.785 |
| abf250/n1 | 77 / 3.896 / 0.000 | 231 / 41.126 / 11.255 |

## Actual visual observations

On these four familiar high-N requests, the strict visual conjunction (reasonable overall shape, plausible C use, support present) is RAW 4/4, original D0 FINAL 0/4, GEO_BASE 4/4 and GEO_FINAL 4/4. For 28b9 both noises, GEO_ONLY retains the gray seat-strip sidewalls and underside contours that the original D0 chain largely removes. For abf both noises, GEO_ONLY retains the end supports instead of splitting them into narrow sheets. This is a bounded shape-preservation positive. Small corner splits remain visible on 28b9, and abf n1 has an uncertain fine support seam; BASE abf n1 has a visible side-strip gap reduced by the final CPU cascade. These observations do not certify legal connectivity, zero intersections, watertightness or unseen-parent generalization; all geometric costs must be reported separately.

The labels are nonblind AI_PROXY, not independent human ratings. The comparison is not a unique-source-style reconstruction task.

## Accounting and limits

The slot and corner ledgers preserve original RAW identity across pre-GEO, merged-before-cleanup, BASE and FINAL. Deleted slots remain explicit. Areas sum triangles including overlapping/double-sided multiplicity; neither lower area nor fewer faces alone establishes missing semantic structure.

The original C geometry and vertex identities are preserved. The downstream guard ledgers separately check existing legal edges, newly illegal pairs, deletions, nonmanifold and same-winding additions. D0-free shape retention does not exempt GEO_ONLY from geometric quality costs.

[Full 177 paired metrics](analysis/paired_full177.csv) · [Area stages](analysis/area_stages.csv) · [Deleted reasons](analysis/deletion_reasons.csv) · [Slot ledger](analysis/slot_stage_ledger.csv) · [Gallery](gallery/index.html)

## Resume

`python -B -m r2.pilots.bench_high_joint_coverage_v1.geo_only verify` verifies completed science; no replay. `geo_report collect/render` reuse sealed outputs. Unresolved partial directories require explicit engineering review. No default switch, cleanup, commit or push.
