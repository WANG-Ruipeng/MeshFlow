"""Deterministic CPU-only checks; no Torch/model/GPU import or invocation."""
from __future__ import annotations

import json
import time
import numpy as np

try:
    from . import corner_contract as C
except ImportError:
    import corner_contract as C


def run_checks():
    started = time.perf_counter()
    results = []

    def checked(name, test):
        if not bool(test):
            raise AssertionError(name)
        results.append(dict(check=name, status="PASS"))

    def rejected(name, call):
        try:
            call()
        except ValueError:
            results.append(dict(check=name, status="PASS"))
            return
        raise AssertionError("Expected rejection: " + name)

    cv = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], np.float64)
    known = np.array([True, False, False, False])
    valid = np.array([True, True, True, False])
    cids = np.full((4, 3), -1, np.int64)
    cids[0] = [0, 1, 2]
    edges = np.array([[-1, -1], [1, 0], [2, 1], [-1, -1]], np.int64)
    occupied = np.array([False, True, False, False])
    ids = C.build_anchor_ids(known, valid, cids, edges, occupied)
    checked("active_two_anchors_null_returns_to_bulk", np.array_equal(ids,
        np.array([[0, 1, 2], [1, 0, -1], [-1, -1, -1], [-1, -1, -1]], np.int64)))
    x = np.zeros((4, 3, 3), np.float64)
    x[0] = cv
    x[1] = [[1., 0., 0.], [0., 0., 0.], [.5, -.5, 0.]]
    x[2] = [[3., 0., 0.], [3., 1., 0.], [3., 0., 1.]]
    original_x = x.copy()
    facts = C.validate_contract(x, known, valid, ids, cv, require_anchors_exact=True)
    checked("free_denominator_3_seam_plus_9_bulk", facts["free_scalar_count"] == 12)
    times = C.corner_conditioning(.3, known, valid, ids)
    checked("mixed_corner_time_1_1_t", np.array_equal(times["corner_time"][1], [1., 1., .3]))
    checked("null_bulk_all_free_time", np.array_equal(times["corner_time"][2], [.3, .3, .3]))
    checked("seam_attention_remains_free_time", times["face_attention_time"][1] == .3)
    checked("padding_no_time_or_role", np.all(times["corner_time"][3] == 0) and np.all(times["corner_role_ids"][3] == 0))

    # Ordinary full C/full F triangles reproduce the old repeated-face masks.
    old_ids = cids.copy()
    old = C.corner_masks(known, valid, old_ids)
    expected = np.repeat((valid & ~known)[:, None], 3, axis=1)
    checked("ordinary_mask_degenerates_to_old_face_contract", np.array_equal(old["free_corner_mask"], expected))
    old_t = C.corner_conditioning(.3, known, valid, old_ids)
    expected_t = np.repeat(np.where(valid, np.where(known, 1., .3), 0.)[:, None], 3, axis=1)
    checked("ordinary_time_degenerates_to_old_face_contract", np.array_equal(old_t["corner_time"], expected_t))
    checked("ordinary_role_degenerates_to_old_face_contract", np.array_equal(old_t["corner_role_ids"], np.repeat(known[:, None], 3, axis=1).astype(np.int64)))
    all_known = C.corner_conditioning(.3, np.array([True]), np.array([True]), cids[:1])
    checked("entirely_known_input_time_one", np.all(all_known["corner_time"] == 1) and not all_known["free_corner_mask"].any())
    all_free = C.corner_conditioning(.3, np.array([False]), np.array([True]), np.full((1, 3), -1, np.int64))
    checked("entirely_free_input_time_t", np.all(all_free["corner_time"] == .3) and all_free["free_corner_mask"].all())

    velocity = np.arange(36, dtype=np.float64).reshape(4, 3, 3) / 10.
    target = np.zeros_like(velocity)
    target[facts["free_corner_mask"]] = .25
    loss = C.masked_velocity_loss(velocity, target, known, valid, ids)
    free = facts["free_corner_mask"]
    checked("loss_has_only_independent_coordinates", loss["loss"] == float(np.square(velocity[free] - target[free]).sum()) / 12.)
    loss_batch = C.masked_velocity_loss(velocity, target, known, valid, ids, denominator=24)
    checked("batch_denominator_scales_microbatch_contribution", loss_batch["loss"] == loss["loss"] / 2.)
    eps = 1.e-5
    numeric = []
    for index in [(1, 2, 0), (1, 0, 0), (2, 1, 2), (3, 0, 0)]:
        plus, minus = velocity.copy(), velocity.copy()
        plus[index] += eps
        minus[index] -= eps
        numerical = (C.masked_velocity_loss(plus, target, known, valid, ids)["loss"] -
                     C.masked_velocity_loss(minus, target, known, valid, ids)["loss"]) / (2 * eps)
        analytical = 2 * (velocity[index] - target[index]) / 12 if free[index[:2]] else 0.
        numeric.append(abs(numerical - analytical))
    checked("finite_difference_loss_gradient_free_only", max(numeric) < 1.e-9)
    advanced = C.euler_step(x, velocity, .02, known, valid, ids, cv)
    checked("euler_anchor_coordinates_bitwise_unchanged", np.array_equal(advanced[~free], x[~free]))
    checked("euler_only_free_corners_integrated", np.array_equal(advanced[free], x[free] + .02 * velocity[free]))
    checked("inputs_not_mutated", np.array_equal(x, original_x))

    z, endpoint = x.copy(), x.copy()
    z[free] += .2
    endpoint[free] -= .1
    u = endpoint - z
    t = .6
    xt = (1. - t) * z + t * endpoint
    xt = C.clamp_coordinates(xt, known, valid, ids, cv)
    estimate = C.straight_endpoint_estimate(xt, u, t, known, valid, ids, cv)
    endpoint_error = float(np.max(np.abs(estimate - endpoint)))
    checked("straight_path_endpoint_recovery", endpoint_error < 1.e-14)
    decoded = C.decode_shared_indices(x, known, valid, ids, cv)
    checked("N_preserved_null_is_real_bulk", decoded["faces"].shape == (3, 3))
    checked("seam_uses_exact_C_vertex_ids", np.array_equal(decoded["faces"][1, :2], [1, 0]))
    checked("decode_roundtrip_coordinates", np.array_equal(decoded["vertices"][decoded["faces"]], x[valid]))
    coincident = x.copy()
    coincident[2, 0] = cv[0]
    independent = C.decode_shared_indices(coincident, known, valid, ids, cv)
    checked("coincident_free_corner_not_silently_welded", independent["faces"][2, 0] >= len(cv))
    checked("padding_IDs_stay_minus_one", np.all(decoded["corner_to_vertex"][3] == -1))

    rejected("reject_integer_bool_mask", lambda: C.corner_masks(known.astype(np.int64), valid, ids))
    bad = ids.copy(); bad[1, 1] = bad[1, 0]
    rejected("reject_duplicate_anchor_ID_on_face", lambda: C.corner_masks(known, valid, bad))
    bad = ids.copy(); bad[3, 0] = 0
    rejected("reject_padding_anchor", lambda: C.corner_masks(known, valid, bad))
    bad = ids.copy(); bad[0, 0] = -1
    rejected("reject_partial_original_C_face", lambda: C.corner_masks(known, valid, bad))
    bad = ids.copy(); bad[1, 0] = 99
    rejected("reject_anchor_ID_out_of_range", lambda: C.validate_contract(x, known, valid, bad, cv))
    bad_x = x.copy(); bad_x[0, 0, 0] += 1.
    rejected("reject_wrong_C_before_integration", lambda: C.euler_step(bad_x, velocity, .02, known, valid, ids, cv))
    bad_x = x.copy(); bad_x[0, 0, 0] = -0.
    rejected("reject_numerically_equal_non_bitwise_anchor", lambda: C.validate_contract(bad_x, known, valid, ids, cv, require_anchors_exact=True))
    restored = C.clamp_coordinates(bad_x, known, valid, ids, cv)
    checked("clamp_restores_C_bits_including_signed_zero", restored[0].tobytes() == cv.tobytes())
    bad_x = x.copy(); bad_x[1, 2, 0] = np.nan
    rejected("reject_nonfinite_state", lambda: C.validate_contract(bad_x, known, valid, ids, cv))
    rejected("reject_precision_conversion_of_C", lambda: C.validate_contract(x, known, valid, ids, cv.astype(np.float32)))
    rejected("reject_mixed_precision_integrator", lambda: C.euler_step(x, velocity.astype(np.float32), .02, known, valid, ids, cv))
    bad_target = target.copy(); bad_target[1, 0, 0] = 1.
    rejected("reject_nonzero_anchor_target_velocity", lambda: C.masked_velocity_loss(velocity, bad_target, known, valid, ids))
    rejected("reject_wrong_scalar_denominator", lambda: C.masked_velocity_loss(velocity, target, known, valid, ids, denominator=11))
    bad_occ = occupied.copy(); bad_occ[3] = True
    rejected("reject_occupied_padding", lambda: C.build_anchor_ids(known, valid, cids, edges, bad_occ))
    bad_edge = edges.copy(); bad_edge[2] = [0, 1]
    rejected("reject_duplicate_C_edge_slots", lambda: C.build_anchor_ids(known, valid, cids, bad_edge, occupied))

    return dict(schema=C.VERSION, status="PASS", checks=results, check_count=len(results),
                derivative_max_error=max(numeric), endpoint_max_error=endpoint_error,
                elapsed_seconds=time.perf_counter() - started, precision="NumPy float64 CPU",
                neural_forwards=0, optimizer_updates=0, GPU_calls=0,
                limitations=["No neural arithmetic/gradient or BF16 execution tested",
                             "Occupancy supplied explicitly; no learned occupancy tested",
                             "Shared C IDs do not certify no intersections or bulk connection"])


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
