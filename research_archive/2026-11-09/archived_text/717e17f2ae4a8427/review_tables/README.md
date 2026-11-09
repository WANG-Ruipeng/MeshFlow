# Frozen image-observation counts

These are descriptive counts of the saved observations, not a new metric or automatic model ranking.
AI_PROXY observations are not human ratings. Reviewer identity, source notes and image hashes remain attached to every row.
Actual receipt status: COMPLETE; viewed rows: 840 / 840.
Numerical ranking seen before review: False.

All template judgment dimensions are included: C_adoption, C_floating, C_replaced_by_other_surface, C_wrong_layer, whole_body_complete, support_missing, fragments_or_folded_layers, visible_join.
Each count table separates NEW32/OLD6, RAW/POST, method, region, original tier and region+tier. Historical MEDIUM is not renamed HALF.
Each category includes explicit zero counts. Counts are output observations, not parent-equal rates; parent-level counts are exported separately.
UNCERTAIN, UNKNOWN and NOT_REVIEWED remain distinct. Review-state-aware tables mark an unviewed row NOT_REVIEWED for every field; original-value tables preserve its untouched UNKNOWN or NOT_REVIEWED values.
No uncertainty is mapped to failure, no missing row is dropped, no binary success index or automatic winner is created.
observations_and_notes_index.csv retains all 840 original rows and exact notes. expanded_vs_small_per_request.csv lists both values and note keys for every paired request/stage/dimension.
The receipt gate and image manifest binding verify recorded identities; this utility does not itself view images or establish blinding.
