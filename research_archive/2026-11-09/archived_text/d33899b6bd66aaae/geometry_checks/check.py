"""Authorized CPU-only geometry checks; never loads model or Y arrays."""
import hashlib
import json
import os
from pathlib import Path
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from r2.pilots.bench_c_edgeflip_v1.geometry import flip_one, canonical_faces, _illegal_pair
from r2.pilots.bench_four_routes_v1.resources import snapshot

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def write(name, obj):
    with (OUT / name).open("x", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)


def run():
    start = time.perf_counter()
    units = []
    square = np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], np.int64)

    def case(name, v, f, expected, reason=None):
        before = (v.tobytes(), f.tobytes())
        new, audit = flip_one(v, f)
        assert audit["status"] == expected, (name, audit)
        assert (v.tobytes(), f.tobytes()) == before
        assert new.dtype == np.int64 and new.shape == f.shape
        assert audit["directed_boundary_exact"]
        if reason:
            assert reason in [x.get("reason") for x in audit["candidates"]] or audit["reason"] == reason, (name, audit)
        if expected == "PASS":
            model, p = canonical_faces(v, new)
            assert np.array_equal(np.take_along_axis(model, np.argsort(p, axis=1), axis=1), new)
            assert len(audit["changed_face_rows"]) == 2
        else:
            assert np.array_equal(new, f)
        units.append(dict(name=name, status="PASS", expected=expected, audit=audit))

    case("convex_square", square, faces, "PASS")
    case("reversed_shared_winding", square, np.asarray([[0, 1, 2], [0, 3, 2]], np.int64), "BLOCKED", "shared_edge_winding_not_opposite")
    nonplanar = square.copy(); nonplanar[3, 2] = .1
    case("noncoplanar", nonplanar, faces, "BLOCKED", "not_coplanar_within_tolerance")
    concave = square.copy(); concave[2] = [.2, .2, 0]
    case("concave", concave, faces, "BLOCKED")
    case("existing_other_diagonal", np.vstack((square, [[2, 2, 0]])).astype(np.float32), np.vstack((faces, [[1, 3, 4]])).astype(np.int64), "BLOCKED", "alternative_diagonal_already_exists")
    case("single_triangle", square, faces[:1], "BLOCKED", "no_eligible_internal_edge")
    case("degenerate_source", square, np.asarray([[0, 0, 1]], np.int64), "BLOCKED", "source_contains_degenerate_or_duplicate_face")
    case("duplicate_source", square, np.vstack((faces, faces[:1])), "BLOCKED", "source_contains_degenerate_or_duplicate_face")
    cross_v = np.vstack((square, [[.5, .5, -1], [.5, .5, 1], [1.5, .5, 0]])).astype(np.float32)
    case("external_noncoplanar_intersection", cross_v, np.vstack((faces, [[4, 5, 6]])).astype(np.int64), "BLOCKED", "new_patch_has_illegal_triangle_intersection")
    overlap_v = np.vstack((square, [[.2, .2, 0], [.8, .2, 0], [.2, .8, 0]])).astype(np.float32)
    case("external_coplanar_overlap", overlap_v, np.vstack((faces, [[4, 5, 6]])).astype(np.int64), "BLOCKED", "new_patch_has_illegal_triangle_intersection")
    angle_v = np.asarray([[.5, .00001, 0], [0, 0, 0], [.5, -1, 0], [1, 0, 0]], np.float32)
    case("minimum_angle_protection", angle_v, faces, "BLOCKED", "new_minimum_angle_below_frozen_threshold")
    # Coplanar triangles adjacent across IDs must be legal; shared-ID point
    # contact must not hide a transverse intersection farther along the face.
    assert not _illegal_pair(square.astype(np.float64), faces[0], faces[1], 1e-7)
    shared_cross = np.asarray([[0,0,0],[1,0,0],[0,1,0],[.5,.5,-1],[.5,.5,1]], np.float64)
    assert _illegal_pair(shared_cross, np.asarray([0,1,2]), np.asarray([0,3,4]), 1e-7)
    units.append(dict(name="local_intersection_shared_vertex_and_edge", status="PASS"))
    # Two independent patches: first original-ID internal edge must win,
    # regardless of face-row order, and later candidates are not evaluated.
    two_v = np.vstack((square, square + [3,0,0])).astype(np.float32)
    two_f = np.vstack((faces + 4, faces)).astype(np.int64)
    new, audit = flip_one(two_v, two_f)
    assert audit["selected_edge"] == [0,2] and audit["selected_face_rows"] == [2,3]
    assert np.array_equal(new[:2], two_f[:2])
    assert audit["candidates"][-1]["status"] == "NOT_EVALUATED"
    units.append(dict(name="lexicographic_first_and_row_preservation", status="PASS", audit=audit))
    real = []
    reg = json.loads((ROOT / "experiments/r2_bench_c_tessellation_v1/registration.json").read_text())
    sources = sorted(p for p in reg["protected"] if p.endswith("/condition.npz"))
    for source in sources:
        path = Path(source)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == reg["protected"][source]
        with np.load(path, allow_pickle=False) as z:
            v, f = z["C_vertices"].copy(), z["C_faces_original"].copy()
        new, audit = flip_one(v, f)
        again, again_audit = flip_one(v, f)
        assert np.array_equal(new, again) and audit == again_audit
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        name = path.parent.name
        write(name + "_audit.json", audit)
        if audit["status"] == "PASS":
            with (OUT / (name + "_flipped_C.npz")).open("xb") as handle:
                np.savez_compressed(handle, C_vertices=v, C_faces_original=new)
        real.append(dict(condition_id=name, source=source, source_sha256=digest,
                         status=audit["status"], reason=audit["reason"], original_K=len(f),
                         edge=audit.get("selected_edge"), new_edge=audit.get("new_edge"),
                         candidate_reasons=[x["reason"] for x in audit["candidates"]],
                         repeat_exact=True))
    write("unit_checks.json", units)
    elapsed = time.perf_counter() - start
    write("resources_after.json", snapshot())
    report = dict(status="PASS", unit_checks=len(units), real_input_checks=real,
                  CPU_workers=1, CPU_threads_per_worker=1, elapsed_seconds=elapsed,
                  dispatch_reason="Tiny dependent geometry gates; NumPy single-thread CPU work overlaps independent parent work; no GPU dispatch",
                  GPU_calls=0, model_calls=0, Y_arrays_read=0,
                  geometry_sha256=hashlib.sha256((ROOT / "r2/pilots/bench_c_edgeflip_v1/geometry.py").read_bytes()).hexdigest())
    write("complete.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    run()
