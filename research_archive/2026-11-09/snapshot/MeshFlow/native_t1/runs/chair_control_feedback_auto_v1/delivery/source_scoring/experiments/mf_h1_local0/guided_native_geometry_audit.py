"""Read-only CPU geometry auditing and scoring for frozen native correction.

GT is confined to score_output. Neither the original-potential audit nor any
correction direction receives target-free geometry. No model is imported here.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .native_patch_seams import (
    diagnose_patch_seams, extract_boundary_segments, _sample_boundary,
    point_to_triangle_surface_squared,
)
from .native_patch_metrics import evaluate_patch_output, prepare_common_reference


SEEDS = (9201, 9202, 9203, 9204)
STATES = (15, 30, 40, 50)
PROPER = "proper_transverse_penetration"
COPLANAR = "coplanar_positive_area_overlap"
ILLEGAL = (PROPER, COPLANAR)


def _increment(ledger, key, amount=1):
    if ledger is not None:
        ledger[key] = ledger.get(key, 0) + amount


def _tri(value):
    value = np.asarray(value)
    if value.ndim == 2 and value.shape[-1] == 9:
        value = value.reshape(-1, 3, 3)
    if value.ndim != 3 or value.shape[1:] != (3, 3):
        raise ValueError("Expected [N,9] or [N,3,3]")
    return value


def _sha_array(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str((str(value.dtype), value.shape)).encode() + value.tobytes()).hexdigest()


def _case(bank, parent):
    return bank["cases_by_key"][(int(parent), 12)]


def _path(value):
    path = Path(value)
    if path.exists():
        return path
    text = str(value).replace("\\", "/")
    if text.startswith("/mnt/") and len(text) > 7:
        candidate = Path(text[5].upper() + ":/" + text[7:])
        if candidate.exists():
            return candidate
    raise FileNotFoundError(value)


def _sets(contacts):
    return {label: {tuple(row["pair"]) for row in contacts if row["classification"] == label}
            for label in ILLEGAL}


def pair_set_change(before, after):
    old, new = _sets(before["contacts"]), _sets(after["contacts"])
    return {label: dict(before=len(old[label]), after=len(new[label]),
                       removed=[list(x) for x in sorted(old[label] - new[label])],
                       added=[list(x) for x in sorted(new[label] - old[label])]) for label in ILLEGAL}


def pair_scan(first, second, length, *, ledger=None, self_pairs=False,
              affected_faces=None, independent=False, return_disjoint=False):
    """Complete triangle tests; affected subset never silently becomes a full audit."""
    from .guided_native_geometry_geometry import classify_triangle_pair_primary, verified_pair
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


def boundary_metrics(context, free, length, ledger=None):
    segments, boundary = extract_boundary_segments(context)
    points, ids, weights, counts = _sample_boundary(segments, 9919, 512)
    d2 = point_to_triangle_surface_squared(points, free, length)
    _increment(ledger, "boundary_distance_evaluations")
    if d2 is None or len(d2) == 0:
        return dict(status="NO_BOUNDARY_OR_FREE", rms=None, p95=None, maximum=None,
                    no_candidates_is_not_zero=True)
    distances = np.sqrt(d2)
    order = np.argsort(distances, kind="stable")
    cumulative = np.cumsum(weights[order])
    position = min(int(np.searchsorted(cumulative, .95 * cumulative[-1], side="left")), len(order)-1)
    p95 = float(distances[order[position]])
    rms = float(np.sqrt(np.sum(weights*d2)))
    maximum = float(distances.max())
    near = distances < .001 * length
    return dict(status="PASS", length_weighted=True, point_count=len(points), point_seed=9919,
                point_sha256=_sha_array(points), weight_sha256=_sha_array(weights),
                rms=rms, p95=p95, maximum=maximum, rms_bbox_pct=100*rms/length,
                p95_bbox_pct=100*p95/length, max_bbox_pct=100*maximum/length,
                coverage_threshold_bbox_fraction=.001,
                coverage_fraction=float(weights[near].sum()),
                weighted_p95_definition="Smallest sampled distance with cumulative arc-length weights >=95%",
                maximum_scope="Sample maximum; not exact Hausdorff", boundary=boundary,
                sample_counts_per_segment=counts)


def score_output(full, parent, bank, *, ledger=None, prior_metrics=None):
    """One independent endpoint scorer; GT is read only inside this function."""
    tick = time.perf_counter()
    case = _case(bank, parent)
    c, full = _tri(case["constraints"]), _tri(full)
    length = float(case["bbox_diagonal"])
    if full.shape != (112, 3, 3) or full.dtype != np.float32:
        raise ValueError("Require actual FP32 [112,3,3] endpoint")
    if not np.array_equal(full[:12], c):
        raise AssertionError("C was not bitwise preserved")
    free = full[12:].copy()
    seam = diagnose_patch_seams(c, free, length)
    _increment(ledger, "primary_pair_tests", 1200)
    _increment(ledger, "historical_seam_evaluations")
    if prior_metrics is None:
        targets = [_case(bank, p)["free_target"] for p in (0, 1)]
        lengths = [float(_case(bank, p)["bbox_diagonal"]) for p in (0, 1)]
        common = prepare_common_reference(targets, lengths)
        mask = np.arange(112) < 12
        metrics, _ = evaluate_patch_output(full, c, case["full_target"], targets,
            common, lengths, parent, mask, seam_result=seam)
        _increment(ledger, "endpoint_metric_evaluations")
        source = "Recomputed with frozen native_patch_metrics; no GT used in correction"
    else:
        metrics = deepcopy(prior_metrics)
        metrics["seam"] = seam
        _increment(ledger, "endpoint_metric_reuses")
        source = "Archived identical raw output scores reused; seam independently recomputed"
    free_free = pair_scan(free, free, length, ledger=ledger, self_pairs=True, independent=True)
    c_free = pair_scan(c, free, length, ledger=ledger, independent=True)
    free64 = free.astype(np.float64)
    normal = np.cross(free64[:,1]-free64[:,0], free64[:,2]-free64[:,0])
    areas = .5*np.linalg.norm(normal,axis=-1)
    metrics["guided_geometry_audit"] = dict(
        scoring_source=source, boundary=boundary_metrics(c, free, length, ledger),
        C_free=c_free, free_free=free_free, context_bitwise_preserved=True, free_faces=100,
        free_face_area=dict(per_face=areas.tolist(), total=float(areas.sum()),
            minimum=float(areas.min()), median=float(np.median(areas)), normal_vectors=normal.tolist(),
            fixed_bbox_area_threshold=1e-12*length*length,
            degenerate_faces_at_fixed_bbox_threshold=np.flatnonzero(areas<=1e-12*length*length).tolist()),
        actual_coordinate_dtype=str(full.dtype), source_full_sha256=_sha_array(full),
        model_forwards=0, optimizer_updates=0, continuous_collision_safety="NOT_RUN",
        topology_weld_performed=False, seconds=time.perf_counter()-tick)
    return metrics


def score_endpoint(raw, bank, parent, *, ledger=None):
    """Public endpoint API, shared by Raw and all permitted corrected groups."""
    return score_output(raw, parent, bank, ledger=ledger)


def _number_match(actual, expected, name, atol=1e-11):
    if actual is None or expected is None or not np.isclose(actual, expected, rtol=1e-10, atol=atol):
        raise AssertionError(f"Historical metric mismatch {name}: actual={actual} expected={expected}")
    return dict(actual=float(actual), expected=float(expected), absolute_error=abs(float(actual)-float(expected)))


def _energy_json(value):
    return {key: item for key, item in value.items() if key != "gradient_np" and not isinstance(item, np.ndarray)}


def audit_archived_trajectories(bank, prior_results, outputdir, *, epsilon_fraction=None, ledger=None):
    """Stage B, eight archived trajectories/32 states; zero model invocations.

    A None epsilon performs baseline/state detection only. Real original-formula
    auditing requires the explicitly preregistered .005 fraction; no sweep exists.
    All potentials act on current geometry and C, not target surfaces.
    """
    from .guided_native_geometry_geometry import (original_vertex_repel, original_seam_near,
        classify_triangle_pair_independent, classify_triangle_pair_primary)
    if epsilon_fraction is not None and float(epsilon_fraction) != .005:
        raise ValueError("This run preregistered only epsilon=.005 L")
    tick = time.perf_counter()
    out = Path(outputdir)
    if not out.exists():
        raise FileNotFoundError("Root must create the isolated output directory first")
    before_ledger = dict(ledger or {})
    prior = {(int(row["parent"]), int(row["seed"])): row["metrics"]
             for row in prior_results["scores"] if row["model"] == "T1" and row["K"] == 12}
    endpoints, states, candidates, derivatives = [], [], [], {}
    for parent in (0, 1):
        case = _case(bank, parent)
        c, length = _tri(case["constraints"]), float(case["bbox_diagonal"])
        segments, _ = extract_boundary_segments(c)
        for seed in SEEDS:
            path = _path(bank["archived_paths"][(parent, seed)])
            original_bytes_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path, allow_pickle=False) as archive:
                trajectory, endpoint = archive["path"].copy(), archive["output"].copy()
            if trajectory.shape != (51,112,9) or trajectory.dtype != np.float32:
                raise AssertionError("Archived state shape/dtype changed")
            if not np.array_equal(trajectory[-1], endpoint):
                raise AssertionError("Archived final frame differs from endpoint")
            if not np.array_equal(trajectory[:, :12].reshape(51,12,3,3), np.broadcast_to(c,(51,12,3,3))):
                raise AssertionError("Archived C not bitwise present in all51 states")
            metrics = score_output(endpoint, parent, bank, ledger=ledger)
            old, now = prior[(parent,seed)], metrics
            reconcile = {}
            for key in ("symmetric_rms_bbox_pct", "predicted_to_target_rms_bbox_pct", "target_to_predicted_rms_bbox_pct"):
                reconcile["remainder_"+key] = _number_match(now["remainder"]["own"][key], old["remainder"]["own"][key], key)
            reconcile["whole_rms"] = _number_match(now["whole"]["surface"]["symmetric_rms_bbox_pct"],old["whole"]["surface"]["symmetric_rms_bbox_pct"],"whole")
            for key in ("rms_bbox_fraction", "length_weighted_near_surface_fraction"):
                reconcile[key] = _number_match(now["seam"]["boundary_to_free_surface"][key],old["seam"]["boundary_to_free_surface"][key],key)
            old_contacts = old["seam"]["context_free_intersections"]
            now_contacts = now["seam"]["context_free_intersections"]
            old_proper = {(x["context_face"],x["free_face"]) for x in old_contacts["contacts"] if x["classification"]==PROPER}
            now_proper = {(x["context_face"],x["free_face"]) for x in now_contacts["contacts"] if x["classification"]==PROPER}
            if now_proper != old_proper:
                raise AssertionError("Historical proper pair identities changed")
            endpoints.append(dict(parent=parent,seed=seed,source_path=str(path),source_sha256=original_bytes_sha,
                known51_bitwise=True,metrics=metrics,reconciliation=reconcile,
                proper_affected_free_faces=len({x[1] for x in now_proper}),proper_pairs=len(now_proper)))
            for k in STATES:
                free = trajectory[k,12:].reshape(100,3,3).copy()
                scan = pair_scan(c,free,length,ledger=ledger,return_disjoint=True)
                proper = [x for x in scan["contacts"] if x["classification"]==PROPER]
                for pair in scan["contacts"]:
                    ci,fi = pair["pair"]
                    candidates.append(dict(parent=parent,seed=seed,k=k,context_face=ci,free_face=fi,
                        classification=pair["classification"],first=c[ci].copy(),second=free[fi].copy(),length=length))
                compact_scan = dict(scan,contacts=[x for x in scan["contacts"] if x["classification"]!="disjoint"])
                state = dict(parent=parent,seed=seed,k=k,t=k/50,source_free_sha256=_sha_array(free),
                    before=compact_scan, already_proper=bool(proper), before_R_start=k==15,
                    model_calls=0)
                if epsilon_fraction is not None:
                    epsilon = epsilon_fraction*length
                    v = original_vertex_repel(free,c,epsilon,ledger=ledger)
                    s = original_seam_near(free,segments,.005*length,ledger=ledger)
                    gradient = np.asarray(v["gradient_np"],dtype=np.float64)
                    updated64 = free.astype(np.float64) - .02*20*gradient
                    updated = updated64.astype(np.float32)
                    after = pair_scan(c,updated,length,ledger=ledger)
                    affected = np.flatnonzero(np.any(updated != free,axis=(1,2))).tolist()
                    ff_before = pair_scan(free,free,length,ledger=ledger,self_pairs=True,affected_faces=affected,independent=True)
                    ff_after = pair_scan(updated,updated,length,ledger=ledger,self_pairs=True,affected_faces=affected,independent=True)
                    after_v = original_vertex_repel(updated,c,epsilon,ledger=ledger)
                    details = []
                    for pair in proper:
                        ci,fi = pair["pair"]
                        local = dict(energy=float(v["pair_energy"][fi,ci]),
                            active_vertex_count=int(v["pair_active_vertex_count"][fi,ci]),
                            gradient_norm=float(v["pair_gradient_norm"][fi,ci]))
                        after_local = dict(energy=float(after_v["pair_energy"][fi,ci]),
                            active_vertex_count=int(after_v["pair_active_vertex_count"][fi,ci]),
                            gradient_norm=float(after_v["pair_gradient_norm"][fi,ci]))
                        actual = classify_triangle_pair_primary(c[ci],updated[fi],length,ledger=ledger)
                        actual64 = classify_triangle_pair_primary(c[ci],updated64[fi],length,ledger=ledger)
                        grad_nonzero = bool(local["gradient_norm"] > 0)
                        details.append(dict(context_face=ci,free_face=fi,before=pair["detector"],
                            local_original_energy=_energy_json(local),local_original_after_energy=_energy_json(after_local),
                            local_gradient_nonzero=grad_nonzero,
                            local_energy_active=bool(local["energy"] > 0),
                            blind_original_vertex_energy=bool(local["energy"] == 0),
                            all_three_vertex_plane_distances_outside_band=bool(np.all(np.abs(v["signed_distance"][fi,:,ci]) >= epsilon)),
                            vertex_plane_signed_distances=np.asarray(v["signed_distance"][fi,:,ci]).tolist(),
                            inside_projection=np.asarray(v["inside_projection"][fi,:,ci],dtype=bool).tolist(),
                            global_gradient_on_face_norm=float(np.linalg.norm(gradient[fi])),
                            after_actual_fp32=actual,after_working_fp64=actual64,
                            energy_decreased=after_local["energy"] < local["energy"],
                            proper_removed=actual["classification"] != PROPER,
                            update_scope="Single global sum over all 100 free faces and 12 C faces; not isolated pair optimization"))
                    legal_contacts = []
                    for pair in compact_scan["contacts"]:
                        label = pair["classification"]
                        if label in ILLEGAL or "unresolved" in label.lower() or "degenerate" in label.lower():
                            continue
                        ci,fi = pair["pair"]
                        legal_contacts.append(dict(context_face=ci,free_face=fi,before_classification=label,
                            local_energy=float(v["pair_energy"][fi,ci]),
                            local_active_vertex_count=int(v["pair_active_vertex_count"][fi,ci]),
                            local_gradient_norm=float(v["pair_gradient_norm"][fi,ci]),
                            face_changed_by_global_update=fi in affected,
                            after_actual_fp32=classify_triangle_pair_primary(c[ci],updated[fi],length,ledger=ledger)))
                    local_active_faces = sorted({x["free_face"] for x in details if x["local_gradient_nonzero"]})
                    blind_faces = sorted({x["free_face"] for x in details if x["blind_original_vertex_energy"]})
                    state["original_formula_audit"] = dict(epsilon_fraction=epsilon_fraction,epsilon=epsilon,
                        h=.02,lambda_repel=20,working_dtype="float64",actual_writeback_dtype="float32",
                        before_V=_energy_json(v),after_V=_energy_json(after_v),S_observed_only=_energy_json(s),
                        after=after, C_free_pair_set_change=pair_set_change(compact_scan,after),
                        changed_free_faces=affected,changed_free_faces_count=len(affected),
                        affected_free_free_before=ff_before,affected_free_free_after=ff_after,
                        free_free_pair_set_change=pair_set_change(ff_before,ff_after),
                        displacement_rms=float(np.sqrt(np.mean((updated.astype(float)-free)**2))),
                        displacement_max_vertex=float(np.linalg.norm(updated.astype(float)-free,axis=-1).max()),
                        proper_pair_details=details,proper_pair_count=len(details),
                        legal_contact_details=legal_contacts,
                        legal_contact_pairs_with_original_activation=sum(x["local_energy"]>0 for x in legal_contacts),
                        legal_contact_pairs_on_moved_faces=sum(x["face_changed_by_global_update"] for x in legal_contacts),
                        proper_pairs_with_local_gradient=sum(x["local_gradient_nonzero"] for x in details),
                        proper_pairs_blind_to_vertex_energy=sum(x["blind_original_vertex_energy"] for x in details),
                        proper_affected_faces=len({x["free_face"] for x in details}),
                        proper_faces_with_any_local_gradient=local_active_faces,proper_faces_with_blind_pair=blind_faces,
                        energy_decreased_but_proper_remains=sum(x["energy_decreased"] and not x["proper_removed"] for x in details),
                        proper_removed_count=sum(x["proper_removed"] for x in details),
                        continuous_collision_safety="NOT_RUN",not_a_formal_R_S_corrector=True)
                    derivatives[f"P{parent}_seed{seed}_k{k}_originalV_once"] = updated.reshape(100,9)
                states.append(state)
            if hashlib.sha256(path.read_bytes()).hexdigest()!=original_bytes_sha:
                raise AssertionError("Read-only source archive changed")
    summaries=[]
    for parent,expected in ((0,77),(1,55)):
        rows=[x for x in endpoints if x["parent"]==parent]
        numerator=sum(x["proper_affected_free_faces"] for x in rows)
        if numerator!=expected:
            raise AssertionError(f"P{parent} historical unique-free baseline changed: {numerator}/400")
        summaries.append(dict(parent=parent,proper_affected_free_faces=numerator,free_faces_denominator=400,
            proper_face_fraction=numerator/400,proper_pairs=sum(x["proper_pairs"] for x in rows),
            boundary_rms_bbox_pct_mean=float(np.mean([x["metrics"]["guided_geometry_audit"]["boundary"]["rms_bbox_pct"] for x in rows])),
            boundary_coverage_mean=float(np.mean([x["metrics"]["guided_geometry_audit"]["boundary"]["coverage_fraction"] for x in rows]))))
    # Predetermined order prioritizes actual contacts in the nonproper stratum,
    # then disjoint pairs. No corrected outputs are used for case selection.
    key=lambda x:(x["parent"],x["seed"],x["k"],x["context_face"],x["free_face"])
    proper_pool=sorted([x for x in candidates if x["classification"]==PROPER],key=key)
    other_pool=sorted([x for x in candidates if x["classification"]!=PROPER],key=lambda x:(x["classification"]=="disjoint",)+key(x))
    verification=[]
    for stratum,pool in (("proper",proper_pool),("nonproper_contact_first",other_pool)):
        for item in pool[:16]:
            result=classify_triangle_pair_independent(item["first"],item["second"],item["length"],ledger=ledger)
            resolved=result.get("resolved",result.get("classification")!="UNRESOLVED")
            same=result.get("classification")==item["classification"]
            # Legal contact subtype may differ; binary illegal status remains
            # explicit and no disagreement is relabeled as a clean result.
            binary_same=(result.get("illegal") == (item["classification"] in ILLEGAL)) if "illegal" in result else same
            verification.append(dict(stratum=stratum,**{k:v for k,v in item.items() if k not in ("first","second")},
                independent=result,exact_taxonomy_agrees=same,illegal_status_agrees=bool(binary_same),resolved=bool(resolved)))
    derivative_record=None
    if derivatives:
        artifact=out/"original_formula_once_states.npz"
        if artifact.exists():
            raise FileExistsError("Never overwrite an earlier original-formula audit")
        np.savez_compressed(artifact,**derivatives)
        derivative_record=dict(path=str(artifact),sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            arrays=len(derivatives),scope="Offline original V update only; not formal correction sampling")
    result=dict(schema="guided_native_geometry_archive_audit_v1",status="PASS_BASELINE_RECONCILIATION",
        model_calls=0,optimizer_updates=0,model_backwards=0,epsilon_fraction=epsilon_fraction,
        endpoints=endpoints,parent_baselines=summaries,states=states,state_count=len(states),
        source_rollouts=8,original_formula_derivatives=derivative_record,
        independent_selection=dict(order="parent,seed,k,context_face,free_face; contacts before disjoint in nonproper stratum",
            proper_pool=len(proper_pool),nonproper_pool=len(other_pool),
            proper_selected=min(16,len(proper_pool)),nonproper_selected=min(16,len(other_pool))),
        independent_verification=verification,
        independent_all_resolved_and_illegal_agree=all(x["resolved"] and x["illegal_status_agrees"] for x in verification),
        geometry_ledger_delta={k:v-before_ledger.get(k,0) for k,v in (ledger or {}).items()},
        seconds=time.perf_counter()-tick,GT_scope="Only immutable offline surface scoring; never original potential directions",
        continuous_collision_safety="NOT_RUN")
    json.dumps(result,allow_nan=False)
    return result


def run_cpu_tests():
    contacts=dict(contacts=[dict(pair=[0,1],classification=PROPER),dict(pair=[2,1],classification=PROPER)])
    after=dict(contacts=[dict(pair=[0,1],classification=PROPER),dict(pair=[2,2],classification=PROPER)])
    changes=pair_set_change(contacts,after)[PROPER]
    assert changes["removed"]==[[2,1]] and changes["added"]==[[2,2]]
    c=np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]],dtype=np.float32)
    metrics=boundary_metrics(c,c,2.)
    assert metrics["rms"] < 1e-12 and metrics["p95"] < 1e-12 and metrics["coverage_fraction"]>.999999
    assert metrics["weighted_p95_definition"]
    return dict(status="PASS",tests=3,scope="Pair identity changes and length-weighted boundary zero-reference fixture",
                model_calls=0,geometry_autograd_calls=0,optimizer_updates=0)
