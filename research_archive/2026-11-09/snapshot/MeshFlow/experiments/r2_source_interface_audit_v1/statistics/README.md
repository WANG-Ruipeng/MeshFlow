# Training source arithmetic audit

All outputs are **DERIVED** from the frozen MINI package. No source mesh, model, geometry detector, training log replay, or generated-output scorer was executed. The MINI package remains read-only; its five input hashes match its existing MANIFEST and are unchanged after reading.

| Quantity | COMMON_SMALL32 | EXP_ONLY35 |
| --- | ---: | ---: |
| Independent parents / conditions | 32 / 91 | 35 / 103 |
| Confirmed illegal FF pairs, deduplicated | 721 | 840 |
| UNKNOWN FF pairs, deduplicated | 2 | 3 |
| Parents with at least one exact-duplicate-Y condition | 9 | 15 |
| Source Y angle <5°, parent equal | 25.363459321692897% | 27.337952339563635% |
| Source Y angle <10°, parent equal | 49.28313128762765% | 46.14772798064534% |

The earlier combined counts 723 / 843 equal confirmed illegal + UNKNOWN. UNKNOWN is unresolved, not confirmed illegal. The union of training Y events happens to contain all saved full-source-M problem pairs in these two cohorts; both counts are retained separately in per_parent.csv.

Actual exact-duplicate-Y exposure is 13,500 / 48,000 = 28.125% in USER_SMALL and 17,193 / 48,000 = 35.81875% in USER_EXPANDED_67, an increase of 7.69375 percentage points. This counts a training condition draw once when Y contains any complete saved exact-duplicate group; it does not multiply by duplicate-face count. It measures data exposure, not gradient contribution or the cause of generated defects.

## Definitions

- Confirmed illegal uses saved `proper` or `coplanar_positive_overlap`. UNKNOWN stays separate. Dedup key is parent, retained variant, unordered source face pair; conditions share the same pair only once in parent/cohort totals.
- Source duplicate groups are intersected with Y. At least two members must remain in Y; one member in C and one in Y is not a duplicate-Y group.
- Minimum angles come from saved per-face values. Thresholds are strictly `<5` and `<10`. Each condition divides by all Y faces (`N-K`), matching the prior table. Each parent gives equal weight to its frozen conditions; cohort means give equal weight to parents. Exact rational means are preserved. There are zero missing/nonfinite saved angle occurrences across these 194 conditions.
- A lower `<10°` rate alongside a higher `<5°` rate describes different tails and does not order overall mesh quality. This audit does not certify source meshes as clean.

## Files

- `per_condition.csv`: 194 condition records, ID sets, separate illegal/UNKNOWN events, duplicate groups, angle counts and exact ratios.
- `per_parent.csv`: 67 parent records, deduplicated counts and condition-mean rates.
- `deduplicated_FF_events.csv`: 1,566 source pairs, categories, source identity and condition occurrence lists.
- `exposure_association.csv`: 285 arm/condition records joined to actual saved exposure counts, clean/width draws and free scalar counts.
- `exposure_summary.csv`: arm/role and whole-arm exposure totals, without conflating exposure with gradients.
- `comparison_checks.json`: all 194 saved condition projections match recomputed counts, IDs, duplicate groups and `<10°` rates.
- `independent_projection_check.json`: a second route using the existing condition table verifies combined event unions, parent-equal `<10°` rates and duplicate-Y exposure.
- `summary.json`, `source_hashes.json`, `output_manifest.json`: definitions, results, original/source hashes and output hashes. `delivery_manifest.json` additionally hashes the script, this README and the independent check.

## Recompute

Standard-library Python only; compatible with an extracted MINI at any path. Write to a **new directory**; the script refuses to overwrite result files or write inside MINI.

```powershell
python .\recompute_statistics.py --mini "E:\MeshFlow\deliverables\R2_32_67_ANALYSIS_MINI_V1" --out "E:\MeshFlow\experiments\r2_source_interface_audit_v1\statistics\recomputed"
```

The source root contains only saved CSV/JSON. The script checks their recorded MINI hashes before computing and checks unchanged bytes afterwards. Reported expectations are used solely for claim comparisons, never to fill missing counts.
