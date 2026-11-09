"""Read-only completed-training audit: no Torch/model/OT or new random draws."""
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
from r2.pilots.bench_four_routes_v1.resources import snapshot

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/r2_bench_c_edgeflip_v1"
_hashes = {}


def path(value):
    s = str(value).replace("\\", "/")
    for prefix in ("E:/MeshFlow/", "/mnt/e/MeshFlow/"):
        if s.startswith(prefix):
            return ROOT / s[len(prefix):]
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def sha(value):
    p = path(value)
    if p not in _hashes:
        h = hashlib.sha256()
        with p.open("rb") as stream:
            for b in iter(lambda: stream.read(8 << 20), b""):
                h.update(b)
        _hashes[p] = h.hexdigest()
    return _hashes[p]


def read(p):
    return json.loads(path(p).read_text("utf-8-sig"))


def lines(p):
    return [json.loads(s) for s in path(p).read_text("utf-8-sig").splitlines()]


def require(ok, text):
    if not ok:
        raise ValueError(text)


def verify(book):
    for p, h in book.items():
        require(sha(p) == h, "File changed: " + str(p))


def main():
    start = time.perf_counter()
    result_path = OUT / "training_audit_final.json"
    require(not result_path.exists(), "Audit already exists; review without rewriting")
    resources = snapshot()
    reg = read(OUT / "training_registration.json")
    reg_sha = sha(OUT / "training_registration.json")
    require(reg["arms"] == ["PLAIN200", "FLIP200"] and reg["updates"] == 200, "Budget registration")
    for key in ("training_code_hashes", "gate_hashes", "input_case_hashes"):
        verify(reg[key])
    for key in ("manifest", "eval_manifest", "source_checkpoint", "input_schedule", "coefficient_schedule", "flip_plan", "source_registration"):
        verify({reg[key]: reg[key+"_sha256"]})
    schedule = read(reg["input_schedule"])["steps"]
    coefficients = read(reg["coefficient_schedule"])["steps"]
    plan = read(reg["flip_plan"])["steps"]
    oldreg = read(reg["source_registration"])
    old_spec = oldreg["schedules"]["M_PLUS3000"]
    verify({old_spec["path"]: old_spec["sha256"], oldreg["coefficient_schedule"]: oldreg["coefficient_schedule_sha256"]})
    oldstream = read(old_spec["path"])["steps"][:200]
    require(coefficients == read(oldreg["coefficient_schedule"])["steps"][:200], "Original coefficient replay")
    require(len(schedule) == len(coefficients) == len(plan) == 200, "Schedule length")
    for i, (entry, source) in enumerate(zip(schedule, oldstream), 1):
        require(entry["samples"] == source["samples"] and entry["stage_step"] == i and entry["global_step"] == 25500+i, "Original scientific stream changed")
    case_rows = read(reg["manifest"])["train"]
    cases = {r["condition_id"]: r for r in case_rows}
    free_targets = []
    for row in case_rows:
        with np.load(path(row["case_path"]), allow_pickle=False) as z:
            ids = z["free_ids"]
            free, pre = z["full_target"][ids], z["pre_ot_full"][ids]
            require(free.shape == (int(z["N"])-int(z["K"]), 3, 3), "Free target shape")
            require(np.isfinite(free).all() and np.array_equal(pre*np.float32(2), free), "Free target/pre-OT relation")
            free_targets.append(dict(condition_id=row["condition_id"], case_sha256=sha(row["case_path"]),
                                     free_target_bytes_sha256=hashlib.sha256(free.tobytes()).hexdigest(),
                                     free_pre_ot_bytes_sha256=hashlib.sha256(pre.tobytes()).hexdigest()))
    gate = read(OUT / "training_preparation/runtime_cpu_gate.json")
    require(gate["status"] == "PASS" and gate["actual_CPU_OT_calls"] == 60 and gate["conditions"] == 30, "CPU packing evidence")
    require(all(r["free_x0_x1_xt_u_bitwise_same"] and r["OT_face_corner_maps_bitwise_same"] and r["actual_FM_velocity_target_bitwise_same"] for r in gate["rows"]), "Free/OT gate mismatch")
    exposures, logs, summary, starts = {}, {}, {}, {}
    evidence_files = {str(OUT / "training_registration.json"): reg_sha}
    for arm in reg["arms"]:
        root = OUT / "training" / arm
        status = read(root / "status.json")
        init = read(root / "branch_initialization.json")
        require(status["status"] == "COMPLETE" and status["stage_step"] == 200 and status["global_step"] == 25700 and status["native_total_step"] == 13700, "Endpoint cursor: " + arm)
        require(status["registration_sha256"] == init["registration_sha256"] == reg_sha, "Registration identity")
        require(init["inherited_optimizer"] and init["inherited_rng"] and not init["fresh_AdamW"] and not init["new_random_stream"], "Initialization reset")
        for k in ("source_checkpoint_sha256", "source_generator_sha256", "source_optimizer_sha256", "source_rng_sha256"):
            require(init[k] == reg[k], "Different inherited source: " + k)
        history = lines(root / "checkpoint_history.jsonl")
        require([x["stage_step"] for x in history] == [50, 100, 150, 200], "Unexpected checkpoint/restart history")
        endpoint = status["checkpoint"]
        require(endpoint == read(root / "checkpoint_step200.json") and all(history[-1].get(k) == v for k,v in endpoint.items()), "Endpoint receipt mismatch")
        require(status["inference_export"] == read(root / "generator_step200.json"), "Inference receipt mismatch")
        for key in ("checkpoint", "inference_export"):
            r = status[key]
            verify({r["path"]: r["sha256"]})
            require(path(r["path"]).stat().st_size == r["bytes"] and r["arm"] == arm and r["global_step"] == 25700, "Endpoint file/identity")
            evidence_files[r["path"]] = r["sha256"]
        require(endpoint["generator_state_sha256"] == status["inference_export"]["generator_state_sha256"], "Inference content identity")
        require(endpoint["rng_sha256"] == reg["source_rng_sha256"], "Inherited global RNG changed")
        call_rows = lines(root / "calls.jsonl")
        begin = [r for r in call_rows if r["event"] == "CALL_BEGIN"]
        done = [r for r in call_rows if r["event"] == "CALL_DONE"]
        require(len(begin) == len(done) == 3400 and len(call_rows) == 6800, "Unmatched/failed/repeated calls")
        require([r["call_id"] for r in begin] == [r["call_id"] for r in done] == list(range(1, 3401)), "Call IDs")
        sequence = [(s, kind, mi) for s in range(1,201) for kind, mi in
                    [(x,m) for m in range(8) for x in ("train_forward","train_backward")] + [("optimizer_update",None)]]
        require([(r["context"]["stage_step"],r["kind"],r["context"].get("microbatch")) for r in begin] == sequence, "Actual scientific operation order")
        require(all(a["kind"] == b["kind"] and a["context"] == b["context"] for a,b in zip(begin,done)), "Begin/end context mismatch")
        require(len({r["attempt_id"] for r in call_rows}) == 1, "Multiple scientific attempts")
        counts = dict(Counter(r["kind"] for r in begin))
        require(counts == {"train_forward":1600,"train_backward":1600,"optimizer_update":200}, "Actual call budget")
        require(status["calls"]["attempted"] == status["calls"]["successful"] == counts and not status["calls"]["failed"] and not status["calls"]["pending_call_ids"], "Completion ledger summary")
        exp, log = lines(root / "exposure.jsonl"), lines(root / "training_log.jsonl")
        require(len(exp) == 1600 and len(log) == 200, "Log lengths")
        for s, (entry, coef, fp, step_log) in enumerate(zip(schedule,coefficients,plan,log), 1):
            batch = exp[(s-1)*8:s*8]
            require(step_log["stage_step"] == s and step_log["group_allocation"] == coef, "Coefficient/step mismatch")
            require(math.isfinite(step_log["objective"]) and math.isfinite(step_log["generator_gradient_norm"]), "Nonfinite training statistic")
            for mi,(actual,source,flip) in enumerate(zip(batch,entry["samples"],fp["slots"])):
                require(all(actual[k] == v for k,v in source.items()), "Actual source sample/RNG/augmentation differs")
                require(actual["stage_step"] == s and actual["microbatch"] == mi and actual["global_step"] == 25500+s, "Actual exposure ordering")
                require(actual["flip"] == dict(flip,flip_applied=bool(arm == "FLIP200" and flip["flip_apply"])), "Actual flip differs from plan")
                require(actual["loss_gain"] == coef["micro_gains"][mi] and actual["actual_batch_denominator_D_W"] == coef["D_W"], "Actual objective weighting")
                require(actual["N"] == int(cases[actual["condition_id"]]["N"]) and actual["K"] == int(cases[actual["condition_id"]]["K"]), "Actual N/K differs")
                require(actual["all_free_scalar_count"] == 9*(actual["N"]-actual["K"]), "Actual free scalar count")
            require(sum(x["flip"]["flip_applied"] for x in batch) == step_log["flip_applied"], "Step flip sum")
            require(math.isclose(sum(x["loss_contribution"] for x in batch),step_log["objective"],rel_tol=0,abs_tol=1e-12), "Step objective sum")
        exposures[arm], logs[arm], starts[arm] = exp, log, begin
        summary[arm] = dict(calls=counts, checkpoint=endpoint, inference_export=status["inference_export"],
            actual_flips=sum(r["flip"]["flip_applied"] for r in exp),
            group_exposure=dict(Counter(r["replay_group"] for r in exp)),
            group_flips=dict(Counter(r["replay_group"] for r in exp if r["flip"]["flip_applied"])),
            width_flips=dict(Counter(r["augmentation"] for r in exp if r["flip"]["flip_applied"])),
            last_objective=log[-1]["objective"], first_objective=log[0]["objective"],
            wall_seconds=status["elapsed_seconds"], checkpoint_steps=[50,100,150,200])
        for name in ("calls.jsonl","exposure.jsonl","training_log.jsonl","status.json","branch_initialization.json","checkpoint_history.jsonl","checkpoint_step200.json","generator_step200.json"):
            evidence_files[str(root/name)] = sha(root/name)
    equal_fields = ("condition_id","parent_id","replay_group","augmentation","alpha","scale_xyz","noise_seed","epsilon_seed","time_seed","permutation_seed","t","N","K","loss_gain","actual_batch_denominator_D_W","reference_denominator_D_B","reference_HIGH_ratio_rho","mechanism")
    require(all(all(a[k] == b[k] for k in equal_fields) for a,b in zip(exposures["PLAIN200"],exposures["FLIP200"])), "Paired actual scientific inputs differ beyond registered C intervention")
    first_flip = next(r["stage_step"] for r in exposures["FLIP200"] if r["flip"]["flip_applied"])
    require(all(all(a[k] == b[k] for k in ("objective","FM","generator_gradient_norm")) for a,b in zip(logs["PLAIN200"][:first_flip-1],logs["FLIP200"][:first_flip-1])), "Pre-intervention measured model behavior differs")
    require(summary["PLAIN200"]["actual_flips"] == 0 and summary["FLIP200"]["actual_flips"] == reg["exposure"]["FLIP200_flipped"] == 728, "Flip totals")
    intervals = [(v[0]["time_unix"],v[-1]["time_unix"]) for v in starts.values()]
    overlap = max(0., min(b for a,b in intervals)-max(a for a,b in intervals))
    result = dict(status="PASS", schema="bench_edgeflip_independent_readonly_training_audit_v1",
        training_registration_sha256=reg_sha, arms=summary, total_train_forward=3200,total_train_backward=3200,total_optimizer_updates=400,
        scientific_failures=0,scientific_retries=0,all1600_paired_sample_rows_and_random_identifiers_exact=True,
        all1600_actual_paired_times_N_K_masks_and_coefficients_exact=True,
        first_intervention_update=first_flip, pre_intervention_loss_and_gradient_exact=True,
        free_target_evidence=free_targets, actual_CPU_packing_gate_sha256=sha(OUT/"training_preparation/runtime_cpu_gate.json"),
        free_target_evidence_boundary="All30 protected source free arrays checked; existing60 actual CPU OT packing checks verify original/flipped free x0/x1/xt/u and maps. The trainer code is hash-identical and uses these frozen adapters. Per-update free arrays were not persisted, so no independent all1600 tensor replay claim is made.",
        checkpoint_evidence_boundary="Full checkpoint/export file hashes, bytes and receipts checked; no tensor deserialization or model construction in this audit. State-hash/AdamW cursor validation is recorded by the hash-frozen executed trainer.",
        comparability_issues=[], limitations=["Reuses the first200 earlier EXPOSURE batches/random values; this is a paired development quick screen, not fresh random-stream evidence.",
            "728/1600=45.5% actual flipped exposures; 780 requests and52 blocked fallbacks. CLEAN/X_WIDTH are paired; each has364 flips.",
            "Condition pair tails make the requested rate approximate50%; all30 parents remain in both arms.",
            "Training intervals overlap; per-arm wall time is not an isolated throughput benchmark."],
        observed_training_overlap_seconds=overlap, model_loads=0,model_forwards=0,OT_calls=0,new_random_draws=0,
        evidence_hashes=evidence_files, auditor_source_sha256=sha(__file__), resources_before=resources,
        audit_wall_seconds=time.perf_counter()-start)
    require("torch" not in sys.modules, "Audit imported Torch")
    with result_path.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps({k:result[k] for k in ("status","total_train_forward","total_train_backward","total_optimizer_updates","first_intervention_update","comparability_issues","observed_training_overlap_seconds")}))


if __name__ == "__main__":
    main()
