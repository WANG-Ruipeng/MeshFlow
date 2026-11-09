"""Pure-stdlib fixed-pilot ledger/receipt acceptance; never imports model libraries.

Run after all sampling and scoring: python -B audit/check_call_closure.py --final --write
Without --final, missing/in-progress artifacts remain INCOMPLETE. No retries, repairs,
weight loads, raw geometry recomputation, or GPU calls are performed.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

LABELS = ("START", "A_FM_1000", "B_FREE_ALIGN_1000", "C_KNOWN_ALIGN_1000")
CONTRACTS = ("C_output_bitwise", "N_preserved", "finite", "C_all51_bitwise",
             "actual_Gaussian_bitwise", "NPZ_C_bitwise", "PLY_C_bitwise")
EXPECTED = dict(train_forward=24000, train_backward=24000, optimizer_update=3000,
                aligner_forward=8000, aligner_backward=8000,
                gradient_observation_backward=12, rollout_attempt=160,
                sampling_forward=8000, teacher_forward=8, teacher_image=128)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def rows(value):
    return value if isinstance(value, list) else value["rows"]

def external(path):
    p = str(path)
    if len(p) > 2 and p[1] == ":" and Path("/mnt").exists():
        return Path("/mnt/" + p[0].lower() + "/" + p[3:].replace("\\", "/"))
    return Path(p)

def check(root):
    root = Path(root).resolve()
    errors, missing, checks = [], [], []
    def require(ok, name, detail=None, pending=False):
        entry = dict(check=name, detail=detail)
        if not ok:
            (missing if pending else errors).append(entry)
        else:
            checks.append(name)
    def read(path, optional=False):
        path = Path(path)
        if not path.exists():
            if not optional: missing.append(dict(check="file_missing", path=str(path)))
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except (ValueError, OSError) as e:
            errors.append(dict(check="invalid_json", path=str(path), error=str(e)))
            return None
    def lines(path):
        path = Path(path)
        if not path.exists():
            missing.append(dict(check="file_missing", path=str(path)))
            return []
        content = path.read_bytes()
        result = []
        for i, raw in enumerate(content.splitlines(), 1):
            if not raw.strip(): continue
            try: result.append(dict(json.loads(raw), _line=i))
            except ValueError as e:
                incomplete_tail = not content.endswith(b"\n") and i == len(content.splitlines())
                (missing if incomplete_tail else errors).append(dict(check="invalid_jsonl", path=str(path), line=i, error=str(e)))
        return result
    protocol = read(root / "protocol.json") or {}
    limits = protocol.get("budget", {})
    calls = lines(root / "audit/budget/calls.jsonl")
    counts, reservations, active, terminated = defaultdict(Counter), {}, {}, {}
    by_kind = defaultdict(list)
    for event in calls:
        et = event.get("event"); kind = event.get("kind")
        if et == "ROLLOUT_COMPLETE": kind = "rollout_attempt"
        context = event.get("context", {})
        require(isinstance(context, dict), "context_is_object", event["_line"])
        if not isinstance(context, dict): continue
        key = (kind, canonical(context))
        counts[str(kind)][et] += 1
        stamp = event.get("time_unix")
        require(isinstance(stamp, (int, float)) and math.isfinite(stamp), "finite_call_timestamp", event["_line"])
        if et == "RESERVE":
            require(kind in limits, "registered_call_kind", dict(line=event["_line"], kind=kind))
            require(key not in reservations, "unique_call_context", dict(line=event["_line"], kind=kind, context=context))
            ordinal = event.get("ordinal")
            require(type(ordinal) is int and ordinal == counts[str(kind)]["RESERVE"], "contiguous_kind_ordinal", dict(line=event["_line"], ordinal=ordinal, kind=kind))
            if key not in reservations:
                reservations[key] = event; active[key] = event; by_kind[kind].append(event)
        elif et in ("RETURN", "ERROR", "ROLLOUT_COMPLETE"):
            require(key in active, "terminal_has_unique_prior_reservation", dict(line=event["_line"], kind=kind, context=context))
            if key in active:
                require(event["_line"] > active[key]["_line"], "reserve_before_terminal", event["_line"])
                active.pop(key); terminated[key] = event
            if et == "ERROR": errors.append(dict(check="actual_call_error", line=event["_line"], kind=kind, context=context, error=event.get("error")))
            if et == "ROLLOUT_COMPLETE":
                require(event.get("nfe") == 50 and event.get("known_all51_bitwise") is True, "completed_rollout_50_and_C51", context)
        else:
            errors.append(dict(check="unrecognized_call_event", event=et, line=event["_line"]))
    for key, event in active.items():
        missing.append(dict(check="pending_call", kind=key[0], context=event["context"], line=event["_line"]))
    for kind, limit in limits.items():
        if kind == "gpu_owner_seconds": continue
        actual = counts[kind]["RESERVE"]
        require(actual <= limit, "within_fixed_budget_" + kind, dict(actual=actual, maximum=limit))
    for kind, expected in EXPECTED.items():
        actual = counts[kind]["RESERVE"]
        require(actual == expected, "normal_completion_count_" + kind,
                dict(actual=actual, expected=expected), pending=actual < expected)
    # Complete formal training coordinates; no checkpoint tensors are read here.
    for kind in ("train_forward", "train_backward", "aligner_forward", "aligner_backward"):
        found = Counter()
        for e in by_kind[kind]:
            c = e["context"]; key = (c.get("arm"), c.get("step"), c.get("microbatch"))
            found[key] += 1
            legal = key[0] in ("A","B","C") and type(key[1]) is int and 1 <= key[1] <= 1000 and type(key[2]) is int and 0 <= key[2] < 8
            require(legal, "training_call_registered_coordinates", dict(kind=kind, context=c))
            if legal: require(c.get("sample_index") == (key[1] - 1) * 8 + key[2], "training_call_sample_cursor", dict(kind=kind, context=c))
        require(all(v == 1 for v in found.values()), "one_call_per_training_coordinate_" + kind)
        for arm in "ABC":
            desired = (8000 if kind.startswith("train_") else (0 if arm == "A" else 4000))
            actual = sum(v for k, v in found.items() if k[0] == arm)
            require(actual == desired, kind + "_per_arm_" + arm, dict(actual=actual, expected=desired), pending=actual < desired)
    for kind in ("train_forward", "train_backward"):
        actual = {(e["context"].get("arm"), e["context"].get("step"), e["context"].get("microbatch")) for e in by_kind[kind]}
        target = {(a, s, m) for a in "ABC" for s in range(1, 1001) for m in range(8)}
        require(actual == target, "all_1000x8_coordinates_" + kind, dict(missing=len(target-actual), extra=len(actual-target)), pending=not (actual-target))
    updates = Counter((e["context"].get("arm"), e["context"].get("step")) for e in by_kind["optimizer_update"])
    target_updates = {(a, s) for a in "ABC" for s in range(1, 1001)}
    require(set(updates) == target_updates and all(v == 1 for v in updates.values()), "each_arm_1000_unique_updates",
            dict(counts={a:sum(v for k,v in updates.items() if k[0] == a) for a in "ABC"}), pending=not(set(updates)-target_updates) and all(v == 1 for v in updates.values()))
    observations = Counter((e["context"].get("arm"), e["context"].get("step"), e["context"].get("objective")) for e in by_kind["gradient_observation_backward"])
    target_obs = {(a, s, objective) for a in "BC" for s in (1, 100, 1000) for objective in ("FM", "weighted_aux")}
    require(set(observations) == target_obs and all(v == 1 for v in observations.values()), "exact_12_registered_gradient_observations", pending=not(set(observations)-target_obs) and all(v == 1 for v in observations.values()))
    # The head backward is participation in the same combined autograd call.
    head_b = {(e["context"].get("arm"), e["context"].get("step"), e["context"].get("microbatch")):e for e in by_kind["aligner_backward"]}
    train_b = {(e["context"].get("arm"), e["context"].get("step"), e["context"].get("microbatch")):e for e in by_kind["train_backward"]}
    head_f = {(e["context"].get("arm"), e["context"].get("step"), e["context"].get("microbatch")) for e in by_kind["aligner_forward"]}
    require(set(head_b) == head_f, "head_F_B_coordinates_equal", pending=bool(head_f-set(head_b)))
    for key,e in head_b.items():
        c=e["context"]; b=train_b.get(key)
        require(c.get("combined_backward") is True and c.get("generator_call_shared") is True,
                "head_backward_explicitly_shared", key)
        require(b is not None and b["context"].get("combined_backward") is True,
                "head_backward_has_combined_generator_backward", key, pending=b is None)
        if b is not None:
            head_end=terminated.get(("aligner_backward",canonical(c)))
            train_end=terminated.get(("train_backward",canonical(b["context"])))
            if head_end is not None and train_end is not None:
                require(e["_line"] < b["_line"] < train_end["_line"] < head_end["_line"],
                        "head_generator_same_nested_combined_backward", key)
    # GPU owners are unique, nonoverlapping and measured over their whole lifetime.
    occupancy = lines(root / "audit/budget/gpu_occupancy.jsonl")
    owners, ended, live, phase_seconds = {}, {}, set(), defaultdict(float)
    for e in occupancy:
        ident=e.get("id"); et=e.get("event")
        if et == "START":
            require(ident not in owners, "unique_GPU_owner_id", ident)
            require(not live, "single_GPU_owner_no_overlap", sorted(live))
            owners[ident]=e;live.add(ident)
        elif et == "END":
            require(ident in live and ident not in ended, "GPU_END_matches_one_START", ident)
            if ident in owners:require(e.get("phase") == owners[ident].get("phase"), "GPU_phase_identity", ident)
            seconds=e.get("seconds")
            require(isinstance(seconds,(int,float)) and math.isfinite(seconds) and seconds >= 0, "valid_GPU_duration", ident)
            if isinstance(seconds,(int,float)) and math.isfinite(seconds):phase_seconds[e.get("phase")]+=seconds
            ended[ident]=e;live.discard(ident)
        else:errors.append(dict(check="unknown_GPU_event", event=et))
    require(not live, "all_GPU_owners_closed", sorted(live), pending=True)
    gpu_seconds=sum(phase_seconds.values())
    require(gpu_seconds <= limits.get("gpu_owner_seconds",21600), "GPU_owner_time_budget", gpu_seconds)
    # Fixed requests and actual completed matrix. Do not inspect quality values.
    reqdoc=read(root / "data/evaluation/requests.json")
    requests=[] if reqdoc is None else rows(reqdoc)
    rid=[q["request_id"] for q in requests]
    require(len(rid)==48 and len(set(rid))==48, "48_unique_registered_requests", len(rid), pending=not requests)
    qmap={q["request_id"]:q for q in requests}
    cache_doc=read(root / "data/evaluation/start_cached_outputs.json")
    cached=[] if cache_doc is None else rows(cache_doc)
    cache_keys={(q["label"],q["request_id"]) for q in cached}
    require(len(cached)==len(cache_keys)==32 and all(k[0]=="START" for k in cache_keys), "32_unique_START_cache_cells", len(cached), pending=not cached)
    expect={(label,r) for label in LABELS for r in rid}; expected_new=expect-cache_keys
    manifest=read(root / "outputs/manifest.json")
    completed=[] if manifest is None else rows(manifest)
    cmap={(r["label"],r["request_id"]):r for r in completed}
    require(len(completed)==len(cmap), "no_duplicate_output_cells", len(completed))
    require(set(cmap)==expect and len(cmap)==192, "192_unique_final_output_matrix",
            dict(actual=len(cmap),missing=len(expect-set(cmap)),extra=len(set(cmap)-expect)), pending=not(set(cmap)-expect))
    sampled={(e["context"].get("label"),e["context"].get("request_id")):e for e in by_kind["rollout_attempt"]}
    require(set(sampled)==expected_new and len(sampled)==160, "160_new_rollouts_exactly_uncached_cells",
            dict(actual=len(sampled),missing=len(expected_new-set(sampled)),extra=len(set(sampled)-expected_new)), pending=not(set(sampled)-expected_new))
    sample_steps=defaultdict(list)
    for e in by_kind["sampling_forward"]:
        c=e["context"];key=(c.get("label"),c.get("request_id"));sample_steps[key].append(e)
        require(key in expected_new, "sampling_forward_registered_uncached_cell", c)
    for key,e in sampled.items():
        values=sample_steps[key];steps=[v["context"].get("euler_step") for v in values]
        require(steps==list(range(50)), "each_rollout_50_ordered_unique_NFE", dict(cell=key,steps=steps), pending=len(steps)<50 and steps==list(range(len(steps))))
        terminal=terminated.get(("rollout_attempt",canonical(e["context"])))
        if terminal:
            require(all(e["_line"]<v["_line"]<terminal["_line"] for v in values), "rollout_encloses_its_forward_reservations", key)
            require(all(terminated.get(("sampling_forward",canonical(v["context"])),{}).get("_line",math.inf)<terminal["_line"] for v in values), "all_velocity_returns_before_rollout_complete", key)
    # Score/run receipt contract checks are independent of score.status==PASS.
    contracts=[]
    def export_flag(run,key):
        for obj in (run,run.get("contracts",{}),run.get("export",{})):
            if key in obj:return obj[key]
        return run.get("export_npz_ply_bitwise")
    for key,row in sorted(cmap.items()):
        label,request_id=key;q=qmap.get(request_id)
        if q is None:continue
        for field in ("panel","cohort","condition_id","parent_id","candidate","N","K","input_sha256","reference_sha256"):
            require(row.get(field)==q.get(field), "output_frozen_request_field", dict(cell=key,field=field))
        is_cache=key in cache_keys
        require(row.get("generation_status")==("VERIFIED_CACHE" if is_cache else "GENERATED"), "output_cache_vs_new_status", key)
        if is_cache:
            original=next(r for r in cached if (r["label"],r["request_id"])==key)
            for field in ("raw_path","raw_sha256","result_path","result_sha256","checkpoint_sha256"):
                require(row.get(field)==original.get(field), "START_cache_pointer_identity", dict(cell=key,field=field))
        run_path=external(row["result_path"]);run=read(run_path)
        score_path=root/"evaluation/rows"/label/(request_id+".json");score=read(score_path)
        if run is not None and row.get("result_sha256"):
            require(digest(run_path)==row["result_sha256"], "run_receipt_hash", key)
        if run is not None:
            require(run.get("known_all51_bitwise") is True, "run_C51_exact_true", key)
            require(run.get("known_coordinates_bitwise") is True, "run_C_final_exact_true", key)
            for field in ("NPZ_C_bitwise","PLY_C_bitwise"):
                require(export_flag(run,field) is True, "run_export_exact_true_"+field, key)
            if not is_cache:require(run.get("nfe")==50, "run_new_nfe50", key)
        flags={} if score is None else score.get("contracts",{})
        if score is not None:
            require(score.get("status")=="PASS", "score_completed", dict(cell=key,status=score.get("status")), pending=score.get("status") in ("PENDING","NOT_RUN"))
            for field in CONTRACTS:require(flags.get(field) is True, "score_contract_exact_true_"+field, dict(cell=key,value=flags.get(field)))
            for field in ("raw_sha256","input_sha256","reference_sha256"):
                require(score.get("identity",{}).get(field)==row.get(field), "score_receipt_asset_identity", dict(cell=key,field=field))
            require(score.get("label")==label and score.get("request_id")==request_id, "score_cell_identity", key)
        contracts.append(dict(label=label,request_id=request_id,cached=is_cache,score_path=str(score_path),run_path=str(run_path),
                              all_seven_exact_true=score is not None and all(flags.get(k) is True for k in CONTRACTS),contracts=flags))
    require(len(contracts)==192 and all(x["all_seven_exact_true"] for x in contracts), "all_192_outputs_have_7_exact_true_contracts", dict(checked=len(contracts),passed=sum(x["all_seven_exact_true"] for x in contracts)), pending=len(contracts)<192 or any(not Path(x["score_path"]).exists() for x in contracts))
    # Provenance receipts for successful preflight and independent exports.
    preflight=read(root/"preflight/preflight.json")
    if preflight is not None:
        require(preflight.get("status")=="PASS" and preflight.get("optimizer_updates")==0, "preflight_PASS_zero_updates")
    for arm in "ABC":
        proof=read(root/"audit"/(arm+"_export_capture_check.json"))
        if proof is not None:
            require(proof.get("status")=="PASS" and proof.get("capture_on_off_bitwise") is True and proof.get("no_remaining_hooks") is True,
                    "independent_export_capture_check_"+arm)
    # Expected declared teacher batch/image counts must match actual reservations.
    teacher=read(root/"teacher_manifest.json")
    if teacher is not None:
        require(teacher.get("status")=="COMPLETE" and teacher.get("images")==128 and teacher.get("teacher_batches")==8, "teacher_cache_actual_128_images_8_batches")
    report=dict(schema="contextual_alignment_call_closure_v1",status="FAIL" if errors else "INCOMPLETE" if missing else "PASS",
        time_unix=time.time(),counts={k:dict(v) for k,v in sorted(counts.items())},normal_expected=EXPECTED,
        head_preflight_forward=None if preflight is None else preflight.get("auxiliary_head_preflight_forward"),
        head_preflight_backward_participation=None if preflight is None else preflight.get("auxiliary_head_preflight_backward"),
        auxiliary_gradient_observations_through_head=sum(1 for e in by_kind["gradient_observation_backward"] if e["context"].get("objective")=="weighted_aux"),
        check_count=len(checks)+len(errors)+len(missing),passed_checks=len(checks),errors=errors,incomplete=missing,
        actual_model_call_failures=sum(v["ERROR"] for k,v in counts.items() if k not in ("teacher_image","rollout_attempt")),
        all_error_events=sum(v["ERROR"] for v in counts.values()),
        gpu_owner_seconds=gpu_seconds,gpu_phase_seconds=dict(phase_seconds),gpu_pending_ids=sorted(live),
        expected_output_cells=192,completed_output_cells=len(cmap),expected_new_rollouts=160,START_cache_cells=len(cache_keys),
        output_contracts=contracts,ledger_sha256=digest(root/"audit/budget/calls.jsonl") if calls else None,
        audit_scope="Stdlib ledger/context and existing immutable score/run receipts only. No weight loading, raw geometry rescore or new numerical model computation. Seven flags require bool True, never None/truthiness.",
        accounting_note="Head backward participates in the same combined generator backward; never sum as extra independent backward. Rollout uses ROLLOUT_COMPLETE, not RETURN. Sparse autograd observations and teacher batches/images are separately counted.",
        actual_model_calls_by_this_script=0,actual_scoring_calls_by_this_script=0)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument("--final",action="store_true");p.add_argument("--write",action="store_true")
    a=p.parse_args();v=check(a.root)
    if a.write:
        path=a.root/"audit"/("call_closure_final.json" if a.final else "call_closure_live.json")
        path.write_text(json.dumps(v,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps({k:v[k] for k in ("status","check_count","passed_checks","counts","gpu_owner_seconds","completed_output_cells","actual_model_call_failures")}))
    if v["errors"]:print(json.dumps({"errors":v["errors"][:10]}))
    if v["incomplete"]:print(json.dumps({"incomplete_count":len(v["incomplete"]),"first_incomplete":v["incomplete"][:5]}))
    return 0 if v["status"]=="PASS" or (not a.final and v["status"]=="INCOMPLETE") else 1

if __name__=="__main__":raise SystemExit(main())