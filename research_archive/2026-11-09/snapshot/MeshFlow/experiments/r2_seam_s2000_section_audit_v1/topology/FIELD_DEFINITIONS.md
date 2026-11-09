# Full POST connectivity lookup

`source_raw_slot` identifies the original generated RAW triangle slot. It is not an unknown complete-parent GT face ID. `post_face_index` indexes the full saved POST mesh, before any section clipping.

`edge_*` and `vertex_*` refer to full saved indexed-mesh connectivity, including C. They are not filtered by illegal-contact tests. A shared vertex is weaker than a shared edge; a connected component is not automatically a useful or valid part.

**Area denominator clarification:** `edge_component_free_area_fraction` and `vertex_component_free_area_fraction` inherit the previous audit's `area_fraction_total_free` field. Their numerator is the **whole component area, including C when present**; their denominator is total POST free area. They may exceed 1. They are neither free-only reachable-area fractions nor attachment success metrics. A more literal name is `component_total_area_over_total_free_area`. The saved numeric values and their original hashes are preserved.

Angles and areas are read-only copies of the previous audited per-face table. No new shape scorer, GEO operation, or model call is invoked.

Query example in WSL:

```text
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B /mnt/e/MeshFlow/r2/tools/seam_s2000_section_audit_v1/face_connectivity.py --query 73b369b9_n0 --slot 20
```

For a source slot deleted in the saved GEO stage, the query returns `DELETED_BY_SAVED_GEO_NOT_A_POST_FACE` and its saved reason instead of inventing POST connectivity.
