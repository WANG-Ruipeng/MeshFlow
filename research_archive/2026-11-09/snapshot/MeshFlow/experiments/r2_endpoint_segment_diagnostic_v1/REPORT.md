# 608 / 610 endpoint versus finite-segment diagnosis

Completed eight existing FULL_S6000 requests; CPU only, no geometry changes, binding, model calls or quality rescoring. Distances use the frozen L=9.208138264587333.

**The proposed <2%L corner-near-segment explanation is absent in all eight requests.** No free pre_geo corner is within2%L of any finite C open segment. However, this is NOT equivalent to all free surfaces being far away.

| Request | Nearest corner to segment (%L) | Best aligned edge: both endpoints to segment, maximum (%L) | Best aligned edge: full endpoint assignment (%L) | Nearest free surface to C segment (%L) |
|---|---:|---:|---:|---:|
| 608b46b0_n0 | 2.259 | 2.304 | 4.098 | 0.000 |
| 608b46b0_n1 | 6.673 | 7.812 | 8.298 | 0.000 |
| 608b46b0_n2 | 4.782 | 4.807 | 5.351 | 0.000 |
| 608b46b0_n3 | 2.641 | 3.004 | 3.949 | 0.000 |
| 610b1621_n0 | 4.404 | 4.597 | 7.460 | 0.000 |
| 610b1621_n1 | 7.888 | 8.049 | 8.861 | 0.000 |
| 610b1621_n2 | 11.564 | 11.872 | 16.519 | 9.238 |
| 610b1621_n3 | 5.344 | 6.042 | 11.100 | 3.402 |

The minima in different columns may come from different edges. The aligned-edge domain is the original POST free open edges with abs(cos)>cos30; endpoint displacement uses original pre_geo corners, exactly as the previous engine. Segment distances clamp projection to the finite segment, not an infinite line.

**608 n0/n3 are mixed cases:** an aligned free edge can place both endpoints within3.5%L of the same C segment (2.304/3.004%L), but no aligned full endpoint assignment fits3.5%L (4.098/3.949%L). Therefore the full-endpoint contract does add a restriction at3.5%, but neither case satisfies the proposed2% close-edge explanation. The other six requests still fail3.5% even after replacing C endpoint targets with the same finite C segment.

**Surface/edge interiors are a separate third situation:** six requests have a finite C boundary segment intersecting/touching a free triangle (minimum distance0). Free open edge interiors are also closer than their endpoint corners in several requests. This does not establish compatible alignment, a legal seam, absence of crossing, or adoption. 610 n2 is clearly spatially distant even in surface distance (9.238%L); 610 n3 surface minimum3.402%L still does not provide an aligned open edge with both endpoints in budget.

Do not conclude that the zero-candidate result is purely a long-edge/short-edge artifact, or that all free geometry is beyond3.5%L. Existing corner placement is insufficient at2% for all eight; partial endpoint-contract obstruction at3.5% is demonstrated only for608 n0/n3. A new split/insert-vertex decoder was not tested.

Artifacts: diagnosis.csv; summary.csv; point_distances.csv (all original free corners, with retained/open-edge eligibility); edge_pairs.csv (all open-edge/C-edge pairs); surface_distance_by_C_edge.csv (PRE_GEO and POST); independent_check.json; input_hashes.json; run.py; surface_check.py. Independent NumPy recomputation verifies all eight endpoint/segment candidate minima within1e-11. Surface distances additionally use a feature-exact segment/triangle calculation; no independent intersection classifier was run.

No GPU work, training, generation, mesh mutation, default change, commit or push. Fixed diagnostic completed; no follow-on experiment started.
