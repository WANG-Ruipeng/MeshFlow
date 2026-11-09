"""Independent CPU audit of final A/B/C pairing; never imports model/trainer/OT code.

Run only after the registered training has finished. Incomplete arms are PENDING.
--self-test checks the audit logic on synthetic records and never reads run data.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import time

ARMS = ("A", "B", "C")
HASH_FIELDS = ("shared_input_sha256", "input_bytehash", "label_bytehash", "ot_cache_key")
BATCH_FIELDS = {"shared_batch_sha256": "shared_input_sha256",
                "input_batch_sha256": "input_bytehash", "label_batch_sha256": "label_bytehash"}
EVENTS = ("INPUT_BEGIN", "INPUT_DONE", "UPDATE_BEGIN", "UPDATE_DONE")
HEX = re.compile(r"^[0-9a-f]{64}$")


class AuditFailure(ValueError):
    pass


def require(condition, message, **context):
    if not condition:
        raise AuditFailure(message + (": " + json.dumps(context, sort_keys=True) if context else ""))


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def integer(value):
    number = float(value)
    require(math.isfinite(number) and number.is_integer(), "Noninteger numeric field", value=str(value))
    return int(number)


def number(value):
    result = float(value)
    require(math.isfinite(result), "Nonfinite logged field", value=str(value))
    return result


def index_unique(rows, field, expected, label):
    values = [integer(r[field]) for r in rows]
    require(values == list(expected), "Missing/duplicate/reordered registered rows", table=label,
            expected_count=len(expected), actual_count=len(values),
            first_values=values[:12], last_values=values[-12:])
    return {integer(r[field]): r for r in rows}


def validate_arm(arm, exposure, logs, events, plan, paired, parents, *, updates=1000, checkpoints=(500, 1000)):
    """Pure list/dict audit, independently reproducing only identity arithmetic."""
    count = 8 * updates
    require(len(parents) == 32 and len(set(parents)) == 32, "Require original 32 distinct parents")
    pm = index_unique(plan, "sample_index", range(count), "frozen_plan")
    im = index_unique(paired, "sample_index", range(count), "frozen_paired_inputs")
    em = index_unique(exposure, "sample_index", range(count), arm + "_exposure")
    lm = index_unique(logs, "step", range(1, updates + 1), arm + "_training_log")
    counts = Counter()
    parent_rows = {uid: dict(arm=arm, uid=uid, exposures=0, alpha1=0, alpha_uniform=0,
                            ratio20=0, ratio40=0, free_coordinates=0) for uid in parents}
    normalized = []
    grouped = defaultdict(list)
    for index, row in em.items():
        registered = pm[index]
        step = index // 8 + 1
        require(row["arm"] == arm and integer(row["step"]) == step, "Exposure arm/step differs", arm=arm, sample_index=index)
        require(row["uid"] in parent_rows, "Unregistered training parent", uid=row["uid"])
        for key in ("uid", "task_id"):
            require(row[key] == registered[key], "Exposure differs from frozen plan", arm=arm, sample_index=index, field=key)
        for key in ("step", "N", "K"):
            require(integer(row[key]) == integer(registered[key]), "Exposure numeric identity differs", arm=arm, sample_index=index, field=key)
        for key in ("alpha", "ratio"):
            require(number(row[key]) == number(registered[key]), "Exposure augmentation/task differs", arm=arm, sample_index=index, field=key)
        require(.9 <= number(row["alpha"]) <= 1.1, "Alpha outside registered range", index=index)
        require(0 <= number(row["t"]) <= 1, "Invalid time", index=index)
        n, k = integer(row["N"]), integer(row["K"])
        require(128 <= n <= 256 and 0 < k < n, "Invalid N/K", index=index)
        free = integer(row["free_coordinates"])
        require(free == 9 * (n-k) == integer(im[index]["free_coordinate_count"]), "Free coordinate count differs", index=index)
        require(row["task_id"] == im[index]["task_id"], "Paired task identity differs", index=index)
        for key in HASH_FIELDS:
            require(HEX.fullmatch(row[key]) is not None, "Malformed input hash", index=index, field=key)
            require(row[key] == im[index][key], "Actual input/label/OT differs from frozen input", arm=arm, sample_index=index, field=key)
        require(row["ot_cache_status"] == "HIT", "Training did not reuse precomputed registered map", arm=arm, sample_index=index, value=row["ot_cache_status"])
        counts[row["uid"]] += 1
        target = parent_rows[row["uid"]]
        target["exposures"] += 1
        target["alpha1" if number(row["alpha"]) == 1. else "alpha_uniform"] += 1
        ratio = number(row["ratio"])
        require(ratio in (.2, .4), "Unexpected context ratio", index=index)
        target["ratio20" if ratio == .2 else "ratio40"] += 1
        target["free_coordinates"] += free
        grouped[step].append(row)
        normalized.append(dict(sample_index=index, uid=row["uid"], task_id=row["task_id"],
            N=n, K=k, alpha=number(row["alpha"]), ratio=ratio, t=number(row["t"]),
            free_coordinates=free, **{key: row[key] for key in HASH_FIELDS}))
    require(count % 32 == 0 and counts == Counter({uid: count//32 for uid in parents}),
            "Training parent exposure imbalance", counts=dict(counts))
    normalized_batches = []
    for step, log in lm.items():
        rows = grouped[step]
        require(len(rows) == 8 and len({r["uid"] for r in rows}) == 8, "Batch does not have eight distinct parents", arm=arm, step=step)
        require(sum(number(r["alpha"]) == 1. for r in rows) == 4, "Batch does not have four exact alpha1", arm=arm, step=step)
        require(log["arm"] == arm and integer(log["conditional_total_step"]) == 5000+step, "Training log arm/total cursor differs", step=step)
        require(integer(log["old_adamw_step"]) == 5000+step, "Old AdamW cursor differs", arm=arm, step=step)
        new = log["new_adamw_step"]
        require(new in (None, "") if arm == "A" else integer(new) == step, "New AdamW cursor differs", arm=arm, step=step)
        require(integer(log["unused_parameters"]) == 0, "Unused trainable parameters", arm=arm, step=step)
        require(number(log["FM"]) >= 0., "Invalid FM", arm=arm, step=step)
        denominator = sum(integer(r["free_coordinates"]) for r in rows)
        require(integer(log["free_coordinate_denominator"]) == denominator, "FM batch denominator differs", arm=arm, step=step)
        hashes = {key: json_hash([r[source] for r in rows]) for key, source in BATCH_FIELDS.items()}
        for key, expected in hashes.items():
            require(log[key] == expected, "Batch hash differs from ordered actual examples", arm=arm, step=step, field=key)
        normalized_batches.append(dict(step=step, denominator=denominator, **hashes))
    # Durable events prove complete registered input/update attempts, not just matching CSV summaries.
    kinds = defaultdict(list)
    for ordinal, event in enumerate(events):
        kind = event.get("event")
        require(kind in (*EVENTS, "CHECKPOINT_BEGIN", "CHECKPOINT_DONE"), "Unknown/failure training event", arm=arm, ordinal=ordinal, event=kind)
        require(event.get("arm") == arm, "Event arm differs", ordinal=ordinal)
        kinds[kind].append((ordinal, event))
    for kind in EVENTS:
        require([integer(e["step"]) for _, e in kinds[kind]] == list(range(1, updates+1)),
                "Missing/duplicate/skipped attempt or update event", arm=arm, event=kind)
    for step in range(1, updates+1):
        positions = [kinds[kind][step-1][0] for kind in EVENTS]
        require(positions == sorted(positions), "Input/update event ordering differs", arm=arm, step=step)
        if step > 1:
            require(kinds["UPDATE_DONE"][step-2][0] < positions[0], "Overlapping/reordered update attempts", step=step)
        input_done = kinds["INPUT_DONE"][step-1][1]
        update_begin = kinds["UPDATE_BEGIN"][step-1][1]
        done = kinds["UPDATE_DONE"][step-1][1]
        for key in BATCH_FIELDS:
            require(input_done[key] == lm[step][key], "INPUT_DONE hash differs from CSV", arm=arm, step=step, field=key)
        for event in (update_begin, done):
            require(event["shared_batch_sha256"] == lm[step]["shared_batch_sha256"], "Update event batch identity differs", step=step)
        require(integer(input_done["effective_free_coordinates"]) == integer(lm[step]["free_coordinate_denominator"]), "Input event denominator differs", step=step)
        require(integer(input_done["batch_index"]) == step, "Input cursor differs", step=step)
        require(integer(done["old_adamw_step"]) == 5000+step, "Event old AdamW cursor differs", step=step)
        require(done["new_adamw_step"] is None if arm == "A" else integer(done["new_adamw_step"]) == step, "Event new AdamW cursor differs", step=step)
        require((integer(done["train_forward"]), integer(done["train_backward"]), integer(done["optimizer_updates"])) == (8,8,1),
                "Update event call counts differ", step=step)
    cp_begin = kinds["CHECKPOINT_BEGIN"]
    cp_done = kinds["CHECKPOINT_DONE"]
    require(len(cp_begin) == len(cp_done), "Unfinished checkpoint save")
    checkpoint_receipts = []
    for (begin_index, begin), (done_index, done) in zip(cp_begin, cp_done):
        step = integer(done["step"])
        require(integer(begin["step"]) == step and begin_index < done_index, "Checkpoint begin/done mismatch")
        require(1 <= step <= updates and kinds["UPDATE_DONE"][step-1][0] < begin_index, "Checkpoint before completed update", step=step)
        ck = done["checkpoint"]
        require(integer(ck["pilot_step"]) == step and HEX.fullmatch(ck["sha256"]) and HEX.fullmatch(ck["model_state_sha256"]),
                "Checkpoint metadata/hash missing", step=step)
        checkpoint_receipts.append(dict(step=step, **ck))
    require(set(checkpoints) <= {integer(r["step"]) for r in checkpoint_receipts}, "Fixed 500/1000 checkpoint event missing")
    return dict(status="PASS",arm=arm,updates=updates,exposure_rows=count,parents=32,
        per_parent_exposures=count//32,all_batches_eight_distinct_parents=True,all_batches_four_alpha1=True,
        frozen_input_label_OT_match=True,ordered_batch_hashes_recomputed=True,
        FM_denominators_match=True,unused_parameters=0,
        old_adamw_start_recorded=5001,old_adamw_final_recorded=5000+updates,
        new_adamw_start_recorded=None if arm=="A" else 1,
        new_adamw_final_recorded=None if arm=="A" else updates,
        event_counts=dict(Counter(e["event"] for e in events)),
        event_recorded_calls=dict(train_forward=8*updates,train_backward=8*updates,optimizer_update=updates),
        no_missing_duplicate_skipped_input_or_update_events=True,
        checkpoint_receipts=checkpoint_receipts,
        ordered_example_identity_sha256=json_hash(normalized),
        ordered_batch_identity_sha256=json_hash(normalized_batches)), normalized, normalized_batches, list(parent_rows.values())


def compare_arms(samples, batches):
    first = ARMS[0]
    for arm in ARMS[1:]:
        require(samples[arm] == samples[first], "Cross-arm actual example/augmentation/hash mismatch", arm=arm)
        require(batches[arm] == batches[first], "Cross-arm actual batch/hash/denominator mismatch", arm=arm)
    return dict(status="PASS", reference_arm=first, compared_arms=list(ARMS),
        matched_examples_per_arm=len(samples[first]), matched_batches_per_arm=len(batches[first]),
        includes=list(HASH_FIELDS)+["uid","task_id","N","K","alpha","ratio","t","free_coordinates"])


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    tmp.replace(path)


def run(artifacts, out=None):
    root = Path(artifacts).resolve()
    target = Path(out).resolve() if out else root/"audit/training_pairing.json"
    source = Path(__file__).resolve()
    report = dict(schema="native_direct_context_training_pairing_audit_v1", status="PENDING",
        source=dict(path=str(source),sha256=digest_bytes(source.read_bytes())),
        started_unix=time.time(), model_calls=0, CUDA_calls=0, OT_calls=0, checkpoint_tensor_loads=0,
        scope="Final actual exposure/log/events pairing; checkpoint save metadata and existence only, no checkpoint-byte rehash or tensor load. Event call counts are recorded claims, not an independent GPU-call ledger recount.",
        expected=dict(arms=list(ARMS),updates_per_arm=1000,examples_per_arm=8000,parents=32,exposures_per_parent=250),
        inputs=[],arms={},errors=[])
    def read(path, kind="json"):
        path = Path(path)
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),
                "Input was changing during final audit", path=str(path))
        report["inputs"].append(dict(path=str(path),sha256=digest_bytes(raw),bytes=len(raw)))
        text = raw.decode("utf-8-sig")
        if kind == "csv": return list(csv.DictReader(io.StringIO(text)))
        if kind == "jsonl": return [json.loads(s) for s in text.splitlines() if s.strip()]
        return json.loads(text)
    try:
        statuses = {}
        for arm in ARMS:
            path = root/"training"/arm/"status.json"
            statuses[arm] = read(path) if path.exists() else dict(status="MISSING")
        report["observed_training_states"] = {arm:{k:v for k,v in d.items() if k in ("status","step","stop_step","conditional_total_step")} for arm,d in statuses.items()}
        require(not any(d.get("status") in ("BLOCKED","FAILED","FAIL") for d in statuses.values()),
                "A training arm reports failure; no final pairing PASS", states=report["observed_training_states"])
        if not all(d.get("status")=="COMPLETE" and d.get("step")==1000 for d in statuses.values()):
            report["reason"]="All three actual training arms must finish 1000 updates before this final audit."
            return report
        manifest = read(root/"data/train_manifest.json")
        plan_doc = read(root/"training_stream/train_plan.json")
        paired_doc = read(root/"training_stream/paired_inputs.json")
        manifest_sha = digest_bytes((root/"data/train_manifest.json").read_bytes())
        plan_sha = digest_bytes((root/"training_stream/train_plan.json").read_bytes())
        require(plan_doc["data_manifest_sha256"]==manifest_sha==paired_doc["data_manifest_sha256"], "Frozen data identity differs")
        require(paired_doc["plan_sha256"]==plan_sha, "Frozen paired plan identity differs")
        parents = [p["uid"] for p in manifest["parents"]]
        require(plan_doc["effective_batch_count"]==1000 and plan_doc["batch_size"]==8 and plan_doc["microbatch_size"]==1, "Frozen plan budget differs")
        require(plan_doc["object_counts"]=={uid:250 for uid in parents}, "Frozen parent exposure target differs")
        all_samples, all_batches, parent_rows = {}, {}, []
        for arm in ARMS:
            folder = root/"training"/arm
            result, samples, batches, rows = validate_arm(arm,
                read(folder/"exposure.csv","csv"), read(folder/"training_log.csv","csv"),
                read(folder/"events.jsonl","jsonl"), plan_doc["records"],paired_doc["records"],parents)
            for ck in result["checkpoint_receipts"]:
                file = Path(ck["path"])
                require(file.resolve().parent == (root/"checkpoints").resolve(), "Checkpoint outside expected artifact directory", path=str(file))
                require(file.name == arm+"_step"+str(ck["step"]).zfill(4)+".pt", "Checkpoint arm/step filename differs", path=str(file))
                require(file.is_file() and file.stat().st_size>0, "Checkpoint file missing/empty", path=str(file))
                ck["file_bytes"]=file.stat().st_size
                ck["file_sha256_recomputed"]=False
            state=statuses[arm]
            require(state["data_manifest_sha256"]==manifest_sha and state["plan_sha256"]==plan_sha, "Final training status identity differs", arm=arm)
            require(state["conditional_total_step"]==6000, "Final total cursor differs", arm=arm)
            latest=read(folder/"latest.json")
            final=[x for x in result["checkpoint_receipts"] if x["step"]==1000][-1]
            for key in ("path","pilot_step","sha256","model_state_sha256"):
                require(latest[key]==state["checkpoint"][key]==final[key], "Latest/final checkpoint receipts disagree", arm=arm,field=key)
            report["arms"][arm]=result
            all_samples[arm],all_batches[arm]=samples,batches
            parent_rows.extend(rows)
        report["cross_arm"]=compare_arms(all_samples,all_batches)
        target.parent.mkdir(parents=True, exist_ok=True)
        parent_csv=target.with_name("training_parent_exposure.csv")
        with parent_csv.open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(parent_rows[0]));writer.writeheader();writer.writerows(parent_rows)
        report["per_parent_csv"]=dict(path=str(parent_csv),rows=len(parent_rows),sha256=digest_bytes(parent_csv.read_bytes()))
        report["status"]="PASS"
    except (AuditFailure,KeyError,ValueError,OSError,TypeError) as exc:
        report["status"]="FAIL";report["errors"].append(dict(type=type(exc).__name__,message=str(exc)))
    finally:
        report["finished_unix"]=time.time()
        write_json(target,report)
    return report


def self_test():
    """Four synthetic batches; reject specific corruptions, no experiment assets."""
    updates=4;parents=["parent_"+str(i) for i in range(32)]
    plan=[];paired=[]
    for i in range(32):
        plan.append(dict(sample_index=i,uid=parents[i],task_id="task_"+str(i),step=i//8+1,
                         N=128,K=26,alpha=1. if i%8<4 else .94+.01*(i%4),ratio=.2))
        paired.append(dict(sample_index=i,task_id="task_"+str(i),free_coordinate_count=918,
                           **{key:json_hash([key,i]) for key in HASH_FIELDS}))
    def fixture(arm):
        exposure=[];logs=[];events=[]
        for p, q in zip(plan,paired):
            exposure.append(dict(p,arm=arm,t=.375,free_coordinates=918,ot_cache_status="HIT",
                                 **{key:q[key] for key in HASH_FIELDS}))
        for step in range(1,5):
            row=exposure[(step-1)*8:step*8]
            hashes={key:json_hash([r[source] for r in row]) for key,source in BATCH_FIELDS.items()}
            logs.append(dict(arm=arm,step=step,conditional_total_step=5000+step,old_adamw_step=5000+step,
                new_adamw_step="" if arm=="A" else step,unused_parameters=0,FM=.2,free_coordinate_denominator=7344,**hashes))
            base=dict(arm=arm,step=step)
            events.extend([dict(base,event="INPUT_BEGIN"),
                dict(base,event="INPUT_DONE",batch_index=step,effective_free_coordinates=7344,**hashes),
                dict(base,event="UPDATE_BEGIN",shared_batch_sha256=hashes["shared_batch_sha256"]),
                dict(base,event="UPDATE_DONE",shared_batch_sha256=hashes["shared_batch_sha256"],
                     old_adamw_step=5000+step,new_adamw_step=None if arm=="A" else step,train_forward=8,train_backward=8,optimizer_updates=1)])
            if step in (2,4):
                events.extend([dict(base,event="CHECKPOINT_BEGIN"),
                    dict(base,event="CHECKPOINT_DONE",checkpoint=dict(pilot_step=step,path=arm+"_step"+str(step)+".pt",
                        sha256=json_hash([arm,step]),model_state_sha256=json_hash([arm,step,"state"])))])
        return exposure,logs,events
    def check(arm,values):
        return validate_arm(arm,*values,plan,paired,parents,updates=4,checkpoints=(2,4))
    outputs={a:check(a,fixture(a)) for a in ARMS}
    compare_arms({a:q[1] for a,q in outputs.items()},{a:q[2] for a,q in outputs.items()})
    checks=["valid_three_arm_synthetic_pairing"]
    modifications=[
        ("duplicate_sample_index",lambda v:v[0][1].update(sample_index=0)),
        ("altered_input_hash",lambda v:v[0][0].update(input_bytehash="0"*64)),
        ("altered_batch_hash",lambda v:v[1][0].update(shared_batch_sha256="0"*64)),
        ("wrong_alpha",lambda v:v[0][0].update(alpha=.97)),
        ("wrong_FM_denominator",lambda v:v[1][0].update(free_coordinate_denominator=7345)),
        ("wrong_old_AdamW",lambda v:v[1][0].update(old_adamw_step=1)),
        ("wrong_new_AdamW",lambda v:v[1][0].update(new_adamw_step=5001)),
        ("skipped_update_event",lambda v:v[2].pop(3)),
        ("missing_checkpoint_event",lambda v:v[2].pop()),
    ]
    for name,change in modifications:
        values=copy.deepcopy(fixture("B"));change(values)
        try:check("B",values)
        except AuditFailure:checks.append(name+"_rejected")
        else:raise AssertionError("Corruption accepted: "+name)
    samples={a:copy.deepcopy(q[1]) for a,q in outputs.items()}
    samples["C"][0]["t"]=.625
    try:compare_arms(samples,{a:q[2] for a,q in outputs.items()})
    except AuditFailure:checks.append("cross_arm_time_mismatch_rejected")
    else:raise AssertionError("Cross-arm mismatch accepted")
    assert "torch" not in sys.modules
    return dict(status="PASS",scope="Finite synthetic audit-logic self-test only; not real experiment audit",
                checks=checks,model_calls=0,optimizer_updates=0,OT_calls=0,torch_imported=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts");parser.add_argument("--out")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args(argv)
    if args.self_test:
        result=self_test()
        if args.out:write_json(args.out,result)
    else:
        if not args.artifacts:parser.error("--artifacts is required unless --self-test")
        result=run(args.artifacts,args.out)
    print(json.dumps(result if args.self_test else {k:result[k] for k in ("status","errors","arms")},indent=2))
    if result["status"]=="FAIL":raise SystemExit(1)
    if result["status"]=="PENDING":raise SystemExit(2)


if __name__=="__main__":
    main()
