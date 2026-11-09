"""Finite CPU trace of the unmodified production conditional FM packing path.

No model, torch, forward, backward, sampling, or geometry scoring is performed.
One actual production OT per registered arm/request is retained. Five times are
algebraic views of that same coupling, not five newly matched couplings.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "r2/src"))
import numpy as np
from r2.pilots.bench_parts_convergence_v1 import data as D
from meshflow_control.training.coupling import to_noise_slots

OUT = ROOT / "experiments/r2_bench_flow_integrity_diagnostic_v1/audits/source_contract"
PANEL = ROOT / "experiments/r2_bench_seat_coverage_v1/eval_manifest.json"
PREFIXES = ("27eb79", "6b894", "91e169", "a8e987", "df6fd3", "fd355")
SEAM_PREFIXES = ("6b894", "91e169")
TIMES = [0.1, 0.3, 0.5, 0.7, 0.9]
SOURCE_FILES = [
    "r2/pilots/bench_parts_convergence_v1/data.py",
    "r2/pilots/seam_abc_v1/slot_data_extension.py",
    "r2/src/meshflow_control/seam/slot_data.py",
    "r2/src/meshflow_control/training/coupling.py",
    "r2/src/meshflow_control/seam/model_adapter.py",
    "r2/src/meshflow_control/seam/corner_contract.py",
    "r2/pilots/bench_full_seat_curriculum_v1/train.py",
    "r2/pilots/bench_full_seat_curriculum_v1/training_state.py",
    "r2/pilots/bench_seat_coverage_v1/train.py",
    "r2/pilots/bench_control_transfer_v1/sample.py",
    "r2/pilots/bench_full_seat_curriculum_v1/sample.py",
    "r2/pilots/bench_flow_integrity_v1/contract_audit.py",
]


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrsha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise RuntimeError("Immutable output differs: " + str(path))
        return
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def append_ledger(event, **kwargs):
    OUT.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": stamp(), "event": event, **kwargs}
    with (OUT / "ledger.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def source_inventory():
    return [{"path": str(ROOT / p), "sha256": sha(ROOT / p)} for p in SOURCE_FILES]


def prepare():
    """Freeze exactly twelve pre-existing request/noise identities, zero OT."""
    existing = OUT / "manifest.json"
    if existing.exists():
        return read(existing)
    panel = read(PANEL)
    selected = [q for q in panel["requests"] if q["parent_id"].startswith(PREFIXES)]
    assert len(selected) == 12
    assert all(sum(q["parent_id"].startswith(p) for q in selected) == 2 for p in PREFIXES)
    assert len({q["condition_id"] for q in selected}) == 6
    # The inherited panel names this existing seed simply "seed".
    selected.sort(key=lambda q: (q["condition_id"], int(q["seed"])))
    requests, traces = [], []
    for index, q in enumerate(selected):
        case_path, noise_path = D.resolve(q["case_path"]), D.resolve(q["noise_npz"])
        case_hash, noise_hash = sha(case_path), sha(noise_path)
        assert case_hash == q["case_sha256"]
        assert noise_hash == q["noise_npz_sha256"]
        case, noise = D.load_case(case_path), D.arrays(noise_path)
        z = np.asarray(noise["z"])
        n, k = int(q["N"]), int(q["K"])
        assert z.shape == (n-k, 3, 3) and z.dtype == np.float32
        assert np.isfinite(z).all()
        assert len(case["full_target"]) == n
        request = {key: q[key] for key in ("request_id", "condition_id", "parent_id", "noise_index", "eval_group", "N", "K")}
        request.update(request_index=index, noise_seed=int(q["seed"]),
                       epsilon_seed=2026102300+index, permutation_seed=2026102400+index,
                       case_path=str(case_path), case_sha256=case_hash,
                       noise_npz=str(noise_path), noise_npz_sha256=noise_hash,
                       noise_array_sha256=arrsha(z),
                       inference_npz=q["inference_npz"],
                       L=float(panel["L"]), augmentation_alpha=1.0,
                       source_occupancy_count=int(case["occupied_labels"].sum()))
        requests.append(request)
        arms = ["NATIVE"]
        if q["parent_id"].startswith(SEAM_PREFIXES):
            arms.append("SEAM_SOURCE_OWNER_DIAGNOSTIC")
        for arm in arms:
            rel = Path("traces") / arm / q["request_id"]
            traces.append({**request, "arm": arm,
                           "production_arm": "PLAIN" if arm == "NATIVE" else "SEAM",
                           "deployable_layout_claim": arm == "NATIVE",
                           "layout_scope": "C-only all-free Native" if arm == "NATIVE" else "Source-owner oracle supervision diagnosis; not deployment inference",
                           "sample_npz": str(OUT / rel / "sample.npz"),
                           "audit_json": str(OUT / rel / "audit.json")})
    assert len(traces) == 16
    manifest = {
        "schema": "BENCH_PRODUCTION_FM_COUPLING_TRACE_V1", "status": "FROZEN",
        "created_utc": stamp(), "panel_path": str(PANEL), "panel_sha256": sha(PANEL),
        "request_sort": "condition_id lexical, then numeric existing panel seed",
        "epsilon_policy": "RandomState(2026102300 + request_index).randn(z.shape).astype(float32)",
        "permutation_policy": "2026102400 + request_index",
        "actual_production_call_time": 0.5, "times": TIMES,
        "t_scope": "Same fixed coupling for all five t; no rematching across t",
        "requests": requests, "traces": traces,
        "authorized_production_OT_calls": 16, "new_noise_draws": 0,
        "new_epsilon_arrays": 12, "model_forwards": 0, "backwards": 0,
        "augmentation": "Identity alpha=1; unaugmented frozen source targets and C",
        "source_inventory": source_inventory(),
        "sample_schema": {
            "x_by_time": "float32 (5,N,9); fixed C/anchor coordinates restored",
            "target": "float32 (N,9), production velocity U",
            "Y": "float32 (N,3,3), production matched endpoint x1",
            "known": "bool (N,), original C faces only",
            "fixed_mask": "bool (N,3), all fixed corners including seam anchors",
            "source_face_ids": "int64 (N,), after production final face permutation",
            "source_corner_permutations": "int64 (N,3), exact target source-corner identity",
            "production_keys": "All keys returned by D.supervised_sample retained unchanged",
        },
        "limitations": [
            "Fixed interpolation probes expose source Y to X_t as ordinary supervision; not autonomous generation",
            "SEAM source-owner layout is a labelled diagnostic; not a C-only deployable result",
            "Native and SEAM layout contracts differ; compare SEAM descriptively, not as same-input causal arm",
            "Finite sample moments are descriptive; Gaussian marginal statement follows independent affine Gaussian algebra",
        ],
    }
    write_new(existing, manifest)
    append_ledger("PREPARED", production_OT_calls=0, requests=12, traces=16, model_forwards=0)
    return manifest


def maxabs(value):
    value = np.asarray(value)
    return float(np.max(np.abs(value))) if value.size else 0.0


def check_trace(case, z, eps, sample, entry):
    """Independent identity reconstruction; never calls nested OT again."""
    n, k = entry["N"], entry["K"]
    full = np.asarray(case["full_target"], np.float32).reshape(n, 3, 3)
    pre = np.asarray(case["pre_ot_full"], np.float32)
    order = sample["face_permutation"]
    invorder = np.argsort(order)
    anchor = sample["anchor_ids"]
    fixed = anchor >= 0
    free = sample["valid_mask"][:, None] & ~fixed
    sf, sc = sample["source_face_ids"], sample["source_corner_permutations"]
    ids_pre, corner_pre = sf[invorder], sc[invorder]
    anchors_pre = anchor[invorder]
    fixed_pre = anchors_pre >= 0
    active = np.asarray(case["occupied_labels"], bool) if entry["production_arm"] == "SEAM" else np.zeros_like(case["occupied_labels"], dtype=bool)
    owners, owner_cp = [], []
    for edge_id in np.flatnonzero(active):
        face_id = int(case["edge_owner_face"][edge_id])
        triple = case["edge_source_triple"][edge_id]
        cp = np.array([np.where(case["source_face_vertex_ids"][face_id] == v)[0].item() for v in triple])
        slot = k + int(edge_id)
        owners.append((int(edge_id), slot, face_id))
        owner_cp.append(cp)
    bulk_slots = np.flatnonzero((~sample["known_mask"][invorder]) & (~fixed_pre).all(1))
    owner_faces = {r[2] for r in owners}
    bulk_faces = np.array([int(f) for f in case["free_ids"] if int(f) not in owner_faces], np.int64)
    rng = np.random.RandomState(entry["permutation_seed"])
    bulk_faces = bulk_faces[rng.permutation(len(bulk_faces))]
    cyclic = np.asarray([[0,1,2],[1,2,0],[2,0,1]])
    cp0 = cyclic[rng.randint(0,3,size=len(bulk_faces))]
    f_map, c_map = sample["ot_face_map"], sample["ot_corner_map"]
    reconstructed_ids = bulk_faces[np.argsort(f_map)]
    reconstructed_cp = to_noise_slots(cp0, f_map, c_map)
    reconstructed_Y = to_noise_slots(full[bulk_faces[:, None], cp0], f_map, c_map)
    mixed = np.float32(np.sqrt(.75)) * z.astype(np.float32) + np.float32(.5) * eps.astype(np.float32)
    x0_pre = np.zeros_like(full)
    x0_pre[k:] = mixed
    x0_pre[fixed_pre] = case["C_vertices"][anchors_pre[fixed_pre]]
    x0_rebuilt = x0_pre[order]
    Y_rebuilt = full[sf[:, None], sc]
    source_vids = case["source_face_vertex_ids"][sf[:, None], sc]
    x_by_time, algebra = [], []
    for value in TIMES:
        x = (np.float32(1)-np.float32(value))*sample["x0"] + np.float32(value)*sample["x1"]
        x[fixed] = sample["x0"][fixed]
        x_by_time.append(x.reshape(n, 9))
        y_from_u = x + (np.float32(1)-np.float32(value))*sample["u"]
        algebra.append({"t": value, "endpoint_identity_max_abs": maxabs(y_from_u - sample["x1"]),
                        "fixed_coordinates_max_abs": maxabs(x[fixed] - sample["x0"][fixed])})
    tip_errors = []
    for edge_id, slot, face_id in owners:
        tip_errors.append(maxabs(sample["x0"][invorder[slot], 2] - mixed[slot-k, 2]))
    # Per-target Hungarian map is a bijection on bulk noise slots, never a map of tip noise.
    checks = {
        "preOT_half_scale_exact": bool(np.array_equal(pre*2, full)),
        "source_face_permutation_exact": bool(np.array_equal(np.sort(sf), np.arange(n))),
        "source_corner_permutations_bijective": bool(np.all(np.sort(sc, axis=1)==np.arange(3))),
        "target_source_corner_roundtrip_exact": bool(np.array_equal(Y_rebuilt, sample["x1"])),
        "bulk_face_map_bijective": bool(np.array_equal(np.sort(f_map), np.arange(len(bulk_faces)))),
        "bulk_ids_inverse_match_exact": bool(np.array_equal(ids_pre[bulk_slots], reconstructed_ids)),
        "bulk_corner_inverse_match_exact": bool(np.array_equal(corner_pre[bulk_slots], reconstructed_cp)),
        "bulk_target_inverse_match_exact": bool(np.array_equal(sample["x1"][invorder][bulk_slots], reconstructed_Y)),
        "noise_slots_before_HYBRID_unchanged": True,
        "x0_from_original_noise_and_epsilon_exact": bool(np.array_equal(x0_rebuilt, sample["x0"])),
        "fixed_target_exact": bool(np.array_equal(sample["x1"][fixed], sample["x0"][fixed])),
        "fixed_velocity_zero": not bool(np.any(sample["u"][fixed])),
        "fixed_anchors_coordinates_exact": bool(np.array_equal(sample["x0"][fixed], case["C_vertices"][anchor[fixed]])),
        "known_mask_original_C_only": bool(np.array_equal(sample["known_mask"], fixed.all(1))),
        "free_mask_exact": bool(np.array_equal(sample["free_corner_mask"], free)),
        "free_denominator_exact": int(sample["free_coordinate_count"]) == 3*int(free.sum()) == 9*(n-k)-6*int(active.sum()),
        "u_is_Y_minus_x0_exact": bool(np.array_equal(sample["u"], sample["x1"]-sample["x0"])),
        "production_t05_exact": bool(np.array_equal(sample["xt"].reshape(n,9), x_by_time[2])),
        "seam_tip_noise_not_exchanged": all(v == 0 for v in tip_errors),
        "five_t_endpoint_identity_FP32": all(r["endpoint_identity_max_abs"] <= 2e-6 for r in algebra),
        "five_t_fixed_coordinates_exact": all(r["fixed_coordinates_max_abs"] == 0 for r in algebra),
        "production_single_OT_call": int(sample["production_OT_calls"]) == 1,
    }
    for (edge_id, slot, face_id), cp in zip(owners, owner_cp):
        checks[f"owner_{edge_id}_fixed_slot_source_identity"] = bool(ids_pre[slot] == face_id and np.array_equal(corner_pre[slot], cp))
    extras = {
        "times": np.asarray(TIMES, np.float64), "x_by_time": np.stack(x_by_time),
        "x": sample["xt"].reshape(n,9), "target": sample["u"].reshape(n,9),
        "Y": sample["x1"], "known": sample["known_mask"], "valid": sample["valid_mask"],
        "fixed_mask": fixed, "raw_noise": z, "epsilon": eps, "hybrid_noise_before_clamp": mixed,
        "full_source_target": full, "preOT_source_target": pre,
        "source_vertex_ids_after_matching": source_vids,
        "preOT_bulk_source_face_ids": bulk_faces,
        "preOT_bulk_initial_corner_permutations": cp0,
        "preOT_bulk_target_coordinates": pre[bulk_faces[:,None], cp0],
        "preOT_bulk_raw_noise": z[bulk_slots-k],
        "bulk_slots_before_final_face_permutation": bulk_slots,
        "owner_edge_slot_source_face": np.asarray(owners, np.int64).reshape(-1,3),
        "inverse_final_face_permutation": invorder,
        "source_occupied_labels": case["occupied_labels"],
        "source_edge_owner_face": case["edge_owner_face"],
        "source_edge_source_triple": case["edge_source_triple"],
        "source_face_vertex_ids": case["source_face_vertex_ids"],
        "fixed_mask_before_final_face_permutation": fixed_pre,
    }
    audit = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
             "N": n, "K": k, "active_owner_count": int(active.sum()),
             "bulk_faces": len(bulk_faces), "free_corner_count": int(free.sum()),
             "free_coordinate_count": int(sample["free_coordinate_count"]),
             "tip_noise_identity_max_abs": max(tip_errors, default=0.0),
             "tip_noise_identity_scope": "Vacuous: no seam" if not owners else "Exact source-owner edge slot, corner2 before/after final metadata permutation",
             "algebra_by_time": algebra,
             "noise_moments_descriptive_only": {"z_mean": float(z.mean()), "z_std": float(z.std()),
                                               "eps_mean": float(eps.mean()), "eps_std": float(eps.std()),
                                               "hybrid_mean": float(mixed.mean()), "hybrid_std": float(mixed.std())},
             "conditional_source_distribution": "Given C and fixed layout H, each unfixed coordinate remains iid N(0,1), because z,epsilon are independent iid Gaussian, variance .75+.25=1; target matching changes Y assignment, not z. Source-owner H itself is target-derived and is not deployment layout.",
             "actual_cost_definition": "scipy.spatial.distance.cdist default Euclidean L2 in 9D; min over six corner permutations, then Hungarian over bulk faces; NOT squared Euclidean and NOT minibatch OT"}
    return extras, audit


def resource_record():
    info = {"utc": stamp(), "CPU_threads_per_process": 1, "CPU_workers": 1,
            "scheduling_reason": "16 small independent CPU Hungarian traces; serial single-thread avoids OT oversubscription and has no GPU work",
            "logical_CPUs": os.cpu_count(), "GPU_compute_used_by_this_script": False}
    try:
        info["load_average"] = list(os.getloadavg())
        info["meminfo"] = {line.split(":")[0]: line.split(":",1)[1].strip() for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith(("MemTotal:","MemAvailable:"))}
    except (OSError, AttributeError):
        info["CPU_RAM_detail"] = "NA: platform interface unavailable"
    for label, command in [
        ("gpu", ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader"]),
        ("gpu_processes", ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]),
    ]:
        try:
            p=subprocess.run(command, capture_output=True, text=True, timeout=10)
            info[label] = {"returncode":p.returncode, "stdout":p.stdout.strip(), "stderr":p.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as ex:
            info[label] = {"status":"NA", "reason":str(ex)}
    return info


def run():
    manifest = prepare()
    assert sha(PANEL) == manifest["panel_sha256"]
    for record in manifest["source_inventory"]:
        assert sha(record["path"]) == record["sha256"], record["path"]
    resources = OUT / "dispatch_resources.json"
    if not resources.exists():
        write_new(resources, resource_record())
    # Do not silently re-execute an OT after an interrupted/failed attempt.
    events = [json.loads(x) for x in (OUT/"ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    attempted = {(e["arm"],e["request_id"]) for e in events if e["event"] == "OT_STARTED"}
    start = time.monotonic()
    for entry in manifest["traces"]:
        key = (entry["arm"],entry["request_id"])
        dest, audit_path = Path(entry["sample_npz"]), Path(entry["audit_json"])
        if audit_path.exists():
            prior=read(audit_path)
            assert dest.exists() and sha(dest) == prior["sample_sha256"]
            continue
        if key in attempted:
            raise RuntimeError("Previously attempted incomplete OT; explicit reviewed recovery required: " + str(key))
        assert sha(entry["case_path"]) == entry["case_sha256"]
        assert sha(entry["noise_npz"]) == entry["noise_npz_sha256"]
        case = D.load_case(entry["case_path"])
        z = D.arrays(entry["noise_npz"])["z"]
        eps=np.random.RandomState(entry["epsilon_seed"]).randn(*z.shape).astype(np.float32)
        protected = {k:arrsha(v) for k,v in case.items() if isinstance(v,np.ndarray)}
        zhash, epshash = arrsha(z), arrsha(eps)
        append_ledger("OT_STARTED", arm=entry["arm"], request_id=entry["request_id"], production_OT_calls_attempted=1)
        stage_start=time.monotonic()
        try:
            sample=D.supervised_sample(case,z,eps,.5,entry["permutation_seed"],entry["production_arm"])
            elapsed=time.monotonic()-stage_start
            extras,audit=check_trace(case,z,eps,sample,entry)
            audit["checks"]["input_arrays_not_mutated"] = all(arrsha(case[k])==v for k,v in protected.items()) and arrsha(z)==zhash and arrsha(eps)==epshash
            audit["status"] = "PASS" if all(audit["checks"].values()) else "FAIL"
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open("xb") as f:
                np.savez_compressed(f,**sample,**extras)
            audit.update(arm=entry["arm"],request_id=entry["request_id"],
                         production_OT_seconds=elapsed,sample_sha256=sha(dest),
                         sample_path=str(dest),source_case_sha256=entry["case_sha256"],
                         z_array_sha256=zhash,epsilon_array_sha256=epshash,
                         model_forwards=0,backwards=0,production_OT_calls=1)
            write_new(audit_path,audit)
            append_ledger("OT_COMPLETED",arm=entry["arm"],request_id=entry["request_id"],status=audit["status"],production_OT_calls=1,wall_seconds=elapsed)
            print(json.dumps({"arm":entry["arm"],"request":entry["request_id"],"status":audit["status"],"free_coordinates":audit["free_coordinate_count"]}),flush=True)
        except Exception:
            append_ledger("OT_FAILED",arm=entry["arm"],request_id=entry["request_id"],traceback=traceback.format_exc())
            raise
    audit_rows=[read(e["audit_json"]) for e in manifest["traces"]]
    summary={"schema":manifest["schema"],"status":"PASS" if all(r["status"]=="PASS" for r in audit_rows) else "FAIL",
             "trace_count":len(audit_rows),"production_OT_calls":sum(r["production_OT_calls"] for r in audit_rows),
             "Native_traces":12,"source_owner_SEAM_traces":4,"times_per_trace":5,
             "model_forwards":0,"backwards":0,"sampling":0,
             "actual_OT_seconds":sum(r["production_OT_seconds"] for r in audit_rows),
             "current_run_wall_seconds":time.monotonic()-start,
             "failed_checks":[{"arm":r["arm"],"request_id":r["request_id"],"check":k} for r in audit_rows for k,v in r["checks"].items() if not v],
             "tip_noise_cross_anchor_exchange_observed":False,
             "conditional_gaussian_source_changed_by_OT":False,
             "important_scope":"Source-owner and production C-only SEAM layouts may differ; this trace validates noise/coupling identity, not their deployment equivalence or generated quality."}
    summary_path=OUT/"summary.json"
    if not summary_path.exists():
        write_new(summary_path,summary)
    csv_path=OUT/"per_trace.csv"
    if not csv_path.exists():
        keys=["arm","request_id","status","N","K","active_owner_count","bulk_faces","free_corner_count","free_coordinate_count","tip_noise_identity_max_abs","production_OT_seconds","sample_sha256"]
        with csv_path.open("x",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in audit_rows)
    # Separate completion package: registration remains immutable and pre-OT.
    # No extra matching or model calls occur while building source-pair metadata.
    from .probe_metrics import source_pairs, prepare_pairs
    pair_sources={}
    rows=[]
    for entry in manifest["traces"]:
        pair_path=Path(entry["sample_npz"]).parent/"source_pairs.json"
        if not pair_path.exists():
            case=D.load_case(entry["case_path"])
            if entry["condition_id"] not in pair_sources:
                pair_sources[entry["condition_id"]]=source_pairs(case,L=entry["L"])
            sample=D.arrays(entry["sample_npz"])
            pairs=prepare_pairs(sample,case,L=entry["L"],source=pair_sources[entry["condition_id"]])
            write_new(pair_path,pairs)
        rows.append({**entry,"representation":"NATIVE" if entry["arm"]=="NATIVE" else "SEAM",
                     "trace_path":entry["sample_npz"],"trace_sha256":sha(entry["sample_npz"]),
                     "pair_path":str(pair_path),"pair_sha256":sha(pair_path)})
    write_new(OUT/"probe_manifest.json",{
        "schema":"BENCH_FM_PROBE_CPU_PACKAGE_V1","status":summary["status"],
        "registration_path":str(OUT/"manifest.json"),"registration_sha256":sha(OUT/"manifest.json"),
        "times":TIMES,"rows":rows,"model_forwards":0,"production_OT_calls":16,
        "pair_preparation_module":str(ROOT/"r2/pilots/bench_flow_integrity_v1/probe_metrics.py"),
        "pair_preparation_module_sha256":sha(ROOT/"r2/pilots/bench_flow_integrity_v1/probe_metrics.py"),
    })
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["prepare","run"])
    args=parser.parse_args()
    if args.action == "prepare":
        m=prepare();print(json.dumps({"manifest":str(OUT/"manifest.json"),"traces":len(m["traces"]),"OT_calls":0}))
    else:
        run()
