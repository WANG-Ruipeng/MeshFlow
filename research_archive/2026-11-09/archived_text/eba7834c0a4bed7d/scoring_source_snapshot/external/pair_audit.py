from collections import Counter
import numpy as np
PROPER="proper_transverse_penetration"
COPLANAR="coplanar_positive_area_overlap"
ILLEGAL=(PROPER,COPLANAR)

def _tri(value):
    value = np.asarray(value)
    if value.ndim == 2 and value.shape[-1] == 9:
        value = value.reshape(-1, 3, 3)
    if value.ndim != 3 or value.shape[1:] != (3, 3):
        raise ValueError("Expected [N,9] or [N,3,3]")
    return value


def pair_scan(first, second, length, *, ledger=None, self_pairs=False,
              affected_faces=None, independent=False, return_disjoint=False):
    """Complete triangle tests; affected subset never silently becomes a full audit."""
    from .intersections import classify_triangle_pair_primary, verified_pair
    a, b = _tri(first), _tri(second)
    affected = None if affected_faces is None else set(int(x) for x in affected_faces)
    rows, counts, tested = [], Counter(), 0
    fn = verified_pair if independent else classify_triangle_pair_primary
    for i in range(len(a)):
        for j in range(i + 1 if self_pairs else 0, len(b)):
            if affected is not None and i not in affected and j not in affected:
                continue
            result = fn(a[i], b[j], length, ledger=ledger)
            label = result.get("classification", "UNRESOLVED")
            if result.get("resolved") is False:
                label = "UNRESOLVED"
            tested += 1
            counts[label] += 1
            if label != "disjoint" or return_disjoint:
                rows.append(dict(pair=[i, j], classification=label, detector=result))
    proper = [row for row in rows if row["classification"] == PROPER]
    affected_proper = sorted({index for row in proper for index in row["pair"]}) if self_pairs else sorted({row["pair"][1] for row in proper})
    unresolved = sum(n for label, n in counts.items() if label not in ILLEGAL and
                     ("unresolved" in label.lower() or "degenerate" in label.lower()))
    return dict(status="PARTIAL_UNRESOLVED" if unresolved else "PASS_STATIC_SCAN",
                scope="all unique free/free pairs" if self_pairs and affected is None else
                      "all C/free pairs" if not self_pairs else "pairs touching changed free faces only",
                tested_pairs=tested, expected_full_pairs=len(a)*(len(a)-1)//2 if self_pairs else len(a)*len(b),
                affected_subset=None if affected is None else sorted(affected),
                classification_counts=dict(counts), proper_pair_count=len(proper),
                proper_affected_faces=affected_proper,
                proper_affected_face_count=len(affected_proper),
                proper_affected_face_denominator=len(a) if self_pairs else len(b),
                proper_affected_face_fraction=len(affected_proper)/(len(a) if self_pairs else len(b)),
                coplanar_positive_area_pairs=counts.get(COPLANAR, 0), unresolved_or_degenerate_pairs=unresolved,
                contacts=rows, independent_verification=independent,
                continuous_collision_safety="NOT_RUN")
