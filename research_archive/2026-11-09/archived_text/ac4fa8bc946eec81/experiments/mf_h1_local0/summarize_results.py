"""Summarize a completed, bounded Phase 2 run without an advantage claim.

Only consumes completed local artifacts. Incomplete runs produce NOT_READY and
no summary files. Base score reuse is never counted as additional generation.
Run: python -m experiments.mf_h1_local0.summarize_results --run-dir RUN_DIR
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from .evaluate import sample_mesh_surface, surface_distance_squared

ARMS = ("Q_center", "G_closest")
KINDS = ("single_edge", "single_face", "mixed_2edges_1face")
SEEDS = (4101, 4102)
MODES = ("correct", "cyclic_mismatch")
PARENTS = tuple(range(4))
METRICS = ("guide_actual_rms_bbox_pct", "guide_original_rms_bbox_pct",
           "completion_symmetric_rms_bbox_pct", "min_angle_lt5_fraction",
           "degenerate_fraction", "angles_p05")


class IncompleteRun(RuntimeError):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    if not Path(path).is_file():
        raise IncompleteRun("Missing required artifact: " + str(path))
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise IncompleteRun(message)


def finite_float(value):
    if value in ("", None, "None", "null"):
        return None
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("Nonfinite metric cannot support a numerical conclusion")
    return number


def read_rows(path):
    if not Path(path).is_file():
        raise IncompleteRun("per_output.csv is not yet complete")
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            for key in ("parent_index", "noise_seed", "condition_parent_index", "raw_faces", "nan_count", "inf_count", "nfe"):
                row[key] = int(row[key])
            for key in METRICS + ("sampling_seconds", "peak_cuda_allocated"):
                row[key] = finite_float(row[key])
            rows.append(row)
    return rows


def row_key(row):
    return (row["arm"], row["parent_index"], row["kind"], row["noise_seed"], row["condition_mode"])


def index_rows(rows):
    indexed = {}
    for row in rows:
        key = row_key(row)
        if key in indexed:
            raise ValueError("Duplicate per-output identity: " + repr(key))
        indexed[key] = row
    expected = {(arm, parent, kind, seed, mode) for arm in ARMS for parent in PARENTS
                for kind in KINDS for seed in SEEDS for mode in MODES}
    expected |= {("base", parent, kind, seed, "none") for parent in PARENTS for kind in KINDS for seed in SEEDS}
    missing, extra = expected-set(indexed), set(indexed)-expected
    require(not missing, "Missing output identities: " + repr(sorted(missing)))
    if extra:
        raise ValueError("Unexpected output identities: " + repr(sorted(extra)))
    if len(rows) != 120:
        raise ValueError("Expected 96 adapted rows plus 24 scores of 2 cached base outputs")
    return indexed


def load_complete(run_dir):
    run_dir = Path(run_dir)
    execution = read_json(run_dir/"phase2_execution.json")
    require(execution.get("status") == "PASS", "Phase 2 is not complete and PASS: " + str(execution.get("status")))
    require(execution.get("optimizer_updates") == 400, "Exactly 400 completed Phase 2 updates required")
    require(execution.get("complete_rollouts") == 98 and execution.get("complete_sampler_calls") == 98,
            "Expected exactly 96 adapted plus 2 independent base completed rollouts")
    require(execution.get("base_hash_unchanged") is True, "Frozen backbone hash check is missing or failed")
    require(execution.get("paired_input_hashes_equal") is True and execution.get("paired_tensor_hashes_equal") is True,
            "Paired Q/G training streams have not been verified")
    if execution.get("phase3") != "NOT_AUTHORIZED":
        raise ValueError("Unexpected Phase 3 authorization state")
    training = {arm: read_json(run_dir/arm/"training_report.json") for arm in ARMS}
    for arm, report in training.items():
        require(report.get("status") == "PASS" and report.get("completed_updates_total") == 200,
                arm + " has not completed exactly 200 updates")
        require(report.get("base_before") == report.get("base_after"), arm + " backbone state changed")
    require(training[ARMS[0]]["initial_state_sha256"] == training[ARMS[1]]["initial_state_sha256"],
            "Q/G initialization hashes differ")
    require(training[ARMS[0]]["input_hashes"] == training[ARMS[1]]["input_hashes"], "Q/G raw input sequences differ")
    require(training[ARMS[0]]["tensor_hashes"] == training[ARMS[1]]["tensor_hashes"], "Q/G input tensor sequences differ")
    split = read_json(run_dir/"split_manifest.json")
    require(len(split["train"]) == 16 and len({v.get("parent_uid",v["uid"]) for v in split["train"]}) == 16,
            "Training manifest must contain exactly 16 distinct parents")
    manifest = read_json(run_dir/"prompt_manifest.json")
    require(manifest.get("status") == "PASS", "Prompt manifest did not pass")
    prompts = {(record["parent_index"], record["kind"]): record for record in manifest["records"]}
    require(len(prompts) == 12 and set(prompts) == {(p,k) for p in PARENTS for k in KINDS},
            "Expected all 4 parents by 3 prompt types")
    require(len({record["parent_uid"] for record in prompts.values()}) == 4, "Exactly four diagnostic parent identities required")
    rows = read_rows(run_dir/"per_output.csv")
    indexed = index_rows(rows)
    expected_names = set()
    for key, row in indexed.items():
        arm, parent, kind, seed, mode = key
        expected_parent = parent if mode in ("correct", "none") else (parent+1)%4
        if row["condition_parent_index"] != expected_parent or row["parent_uid"] != prompts[(parent,kind)]["parent_uid"]:
            raise ValueError("Prompt/parent identity mismatch for " + repr(key))
        expected_name = (f"base_bf16_seed{seed}" if arm == "base" else
                         f"{arm}_parent{parent}_{kind}_seed{seed}_{mode}")
        if row["name"] != expected_name:
            raise ValueError("Unexpected raw name for " + repr(key))
        expected_names.add(expected_name)
        require((run_dir/"raw"/(expected_name+".npz")).is_file(), "Missing raw output " + expected_name)
        require(row["raw_faces"] == 512 and row["nfe"] == 50, "Invalid raw face or NFE count")
        require(row["nan_count"] == 0 and row["inf_count"] == 0, "Nonfinite raw output cannot support response summary")
    if len(expected_names) != 98:
        raise ValueError("Raw identities do not equal 98 saved completed outputs")
    return execution, training, prompts, indexed


def average(values):
    valid = [value for value in values if value is not None]
    return float(np.mean(valid)) if len(valid) == len(values) and valid else None


def paired_metric(indexed, metric, mode, kinds):
    parents = []
    for parent in PARENTS:
        q = average([indexed[("Q_center",parent,kind,seed,mode)][metric] for kind in kinds for seed in SEEDS])
        g = average([indexed[("G_closest",parent,kind,seed,mode)][metric] for kind in kinds for seed in SEEDS])
        delta = g-q if g is not None and q is not None else None
        # A zero baseline has no defined relative change; do not insert epsilon.
        relative = delta/q if delta is not None and q != 0 else None
        parents.append(dict(parent_index=parent, parent_uid=indexed[("Q_center",parent,kinds[0],4101,mode)]["parent_uid"],
                            q_mean=q, g_mean=g, g_minus_q=delta, relative_change_g_minus_q_over_q=relative))
    q_mean, g_mean = average([p["q_mean"] for p in parents]), average([p["g_mean"] for p in parents])
    delta = g_mean-q_mean if g_mean is not None and q_mean is not None else None
    valid = [p for p in parents if p["g_minus_q"] is not None]
    return dict(metric=metric, parent_count=4, noises_per_parent=2, prompt_types=list(kinds),
                q_mean=q_mean, g_mean=g_mean, g_minus_q=delta,
                relative_change_of_means=delta/q_mean if delta is not None and q_mean != 0 else None,
                mean_parent_relative_change=average([p["relative_change_g_minus_q_over_q"] for p in parents]),
                fraction_parents_g_lower=float(np.mean([p["g_minus_q"] < 0 for p in valid])) if valid else None,
                interpretation="Lower is preferable except angles_p05; descriptive n=4, one training seed only.",
                parents=parents)


def metric_summary(indexed, arm, mode, kinds):
    output = {}
    for metric in METRICS:
        parent_means = [average([indexed[(arm,parent,kind,seed,mode)][metric] for kind in kinds for seed in SEEDS])
                        for parent in PARENTS]
        output[metric] = dict(mean=average(parent_means), per_parent_means=parent_means)
    return output


def training_summary(run_dir, training):
    output = {}
    for arm in ARMS:
        log_path = run_dir/arm/"train.jsonl"
        if not log_path.is_file():
            raise IncompleteRun("Missing complete training log: " + str(log_path))
        records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        if [row["step"] for row in records] != list(range(1,201)):
            raise ValueError("Training log does not record exactly the ordered 200 updates: " + arm)
        report = training[arm]
        output[arm] = dict(updates=report["completed_updates_total"],
            loss_first=records[0]["loss"], loss_last=records[-1]["loss"],
            loss_mean_first20=average([row["loss"] for row in records[:20]]),
            loss_mean_last20=average([row["loss"] for row in records[-20:]]),
            teacher_mse_before=report["teacher_mse_before"], teacher_mse_after=report["teacher_mse_after"],
            initial_state_sha256=report["initial_state_sha256"],
            input_sequence_sha256=report["input_sequence_sha256"],
            base_before=report["base_before"], base_after=report["base_after"],
            counts=report["counts"], phase_forward_counts=report["phase_forward_counts"],
            peak_cuda_allocated=report["peak_cuda_allocated"], seconds=report["seconds"],
            training_loss_is_not_a_generation_quality_result=True)
    return output


def raw_arrays(run_dir, row):
    with np.load(run_dir/"raw"/(row["name"]+".npz"), allow_pickle=False) as loaded:
        return {key:loaded[key].copy() for key in loaded.files}


def prompt_arrays(record):
    path = Path(record["path"])
    require(path.is_file(), "Missing fixed prompt " + str(path))
    if sha256(path) != record["sha256"]:
        raise ValueError("Fixed prompt hash changed: " + str(path))
    with np.load(path, allow_pickle=False) as loaded:
        return {key:loaded[key].copy() for key in loaded.files}


def condition_pairs(run_dir, indexed, prompts):
    records = []
    for arm in ARMS:
        for parent in PARENTS:
            for kind in KINDS:
                for seed in SEEDS:
                    correct_row = indexed[(arm,parent,kind,seed,"correct")]
                    wrong_row = indexed[(arm,parent,kind,seed,"cyclic_mismatch")]
                    correct = raw_arrays(run_dir, correct_row)
                    wrong = raw_arrays(run_dir, wrong_row)
                    if not np.array_equal(correct["initial_noise"], wrong["initial_noise"]):
                        raise ValueError("Condition-response pair has different initial noise")
                    for array, index in ((correct,parent),(wrong,(parent+1)%4)):
                        record = prompts[(index,kind)]
                        expected = prompt_arrays(record)
                        for raw_key, prompt_key in (("prompt_vertices","vertices"),("prompt_arity","arity"),
                                                    ("prompt_points","points"),("point_primitive","point_primitive"),
                                                    ("check_points","check_points")):
                            if not np.array_equal(array[raw_key], expected[prompt_key]):
                                raise ValueError("Saved raw condition differs from input prompt: " + raw_key)
                        if str(array["condition_parent_uid"].item()) != record["parent_uid"]:
                            raise ValueError("Saved raw condition parent UID mismatch")
                    difference = correct["tokens"].astype(np.float64)-wrong["tokens"].astype(np.float64)
                    if difference.shape != (512,9) or not np.isfinite(difference).all():
                        raise ValueError("Invalid raw triangle soup in response pair")
                    bbox = float(prompts[(parent,kind)]["bbox_diagonal"])
                    records.append(dict(arm=arm,parent_index=parent,kind=kind,noise_seed=seed,
                        same_initial_noise=True, raw_conditions_match_manifests=True,
                        token_max_abs=float(np.abs(difference).max()), token_rms=float(np.sqrt(np.mean(difference**2))),
                        token_rms_original_bbox_pct=float(100*np.sqrt(np.mean(difference**2))/bbox),
                        correct_actual_guide=correct_row["guide_actual_rms_bbox_pct"],
                        mismatch_actual_guide=wrong_row["guide_actual_rms_bbox_pct"],
                        mismatch_minus_correct_actual_guide=wrong_row["guide_actual_rms_bbox_pct"]-correct_row["guide_actual_rms_bbox_pct"],
                        correct_original_guide=correct_row["guide_original_rms_bbox_pct"],
                        mismatch_original_guide=wrong_row["guide_original_rms_bbox_pct"],
                        mismatch_minus_correct_original_guide=wrong_row["guide_original_rms_bbox_pct"]-correct_row["guide_original_rms_bbox_pct"]))
    summaries = {}
    numeric = ("token_max_abs","token_rms","token_rms_original_bbox_pct","correct_actual_guide","mismatch_actual_guide",
               "mismatch_minus_correct_actual_guide","correct_original_guide","mismatch_original_guide",
               "mismatch_minus_correct_original_guide")
    for arm in ARMS:
        summaries[arm] = {}
        for label,kinds in [(kind,(kind,)) for kind in KINDS]+[("equal_type_overall",KINDS)]:
            parent_means = []
            for parent in PARENTS:
                values = [r for r in records if r["arm"]==arm and r["parent_index"]==parent and r["kind"] in kinds]
                parent_means.append(dict(parent_index=parent,**{key:average([r[key] for r in values]) for key in numeric}))
            summaries[arm][label] = dict(parent_count=4,per_parent=parent_means,
                mean={key:average([r[key] for r in parent_means]) for key in numeric},
                fraction_parents_correct_improves_original_guide=float(np.mean(
                    [r["mismatch_minus_correct_original_guide"]>0 for r in parent_means])))
    return records,summaries


def exploratory_diversity(run_dir,indexed,prompts,count=256):
    records=[]
    cache={}
    measured_inputs={}
    for arm in ARMS:
        for parent in PARENTS:
            for kind in KINDS:
                for mode in MODES:
                    input_parent=parent if mode=="correct" else (parent+1)%4
                    unique_input=(arm,input_parent,kind)
                    arrays=[]
                    for seed in SEEDS:
                        row=indexed[(arm,parent,kind,seed,mode)]
                        arrays.append(raw_arrays(run_dir,row)["tokens"].reshape(-1,3,3))
                    raw_pair_hash=hashlib.sha256(b"".join(value.tobytes() for value in arrays)).hexdigest()
                    if unique_input in measured_inputs and measured_inputs[unique_input]["raw_pair_sha256"]==raw_pair_hash:
                        records.append(dict(measured_inputs[unique_input],parent_index=parent,condition_mode=mode,
                                            reused_same_conditioning_pair=True))
                        continue
                    input_parent=parent if mode=="correct" else (parent+1)%4
                    key=(input_parent,kind)
                    if key not in cache:
                        cache[key]=prompt_arrays(prompts[key])
                    checks=cache[key]["check_points"]
                    if checks.ndim!=3 or checks.shape[1:]!=(128,3):
                        raise ValueError("Expected 128 independent checks per primitive")
                    checks=checks.reshape(-1,3)
                    bbox=float(prompts[key]["bbox_diagonal"])
                    tag=f"{arm}|{input_parent}|{kind}|diversity_v1"
                    sampling_seed=int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4],"little")
                    rng=np.random.RandomState(sampling_seed)
                    sampled=[sample_mesh_surface(mesh,count,rng) for mesh in arrays]
                    base=dict(arm=arm,parent_index=parent,kind=kind,condition_mode=mode,
                              condition_parent_index=input_parent,noise_seeds=list(SEEDS),sampling_seed=sampling_seed,
                              area_samples_per_direction=count,exclusion_distance_bbox_fraction=.01)
                    if any(value is None for value in sampled):
                        records.append(dict(**base,status="NOT_COMPUTABLE_ZERO_AREA_SURFACE"))
                        continue
                    kept=[points[np.min(np.sum((points[:,None]-checks[None])**2,axis=-1),axis=1)>(.01*bbox)**2]
                          for points in sampled]
                    base["retained_surface_points"]=[len(value) for value in kept]
                    if not all(len(value)>0 for value in kept):
                        records.append(dict(**base,status="NOT_COMPUTABLE_EMPTY_UNGUIDED_REGION"))
                        continue
                    d2=[surface_distance_squared(kept[0],arrays[1]),surface_distance_squared(kept[1],arrays[0])]
                    if not all(np.isfinite(value).all() for value in d2):
                        records.append(dict(**base,status="NOT_COMPUTABLE_NONFINITE_SURFACE_DISTANCE"))
                        continue
                    rms=float(np.sqrt((d2[0].mean()+d2[1].mean())/2))
                    result=dict(**base,status="PASS",unguided_surface_symmetric_rms=rms,
                                unguided_surface_symmetric_rms_bbox_pct=100*rms/bbox,
                                reused_same_conditioning_pair=False,raw_pair_sha256=raw_pair_hash)
                    records.append(result)
                    measured_inputs[unique_input]=result
    summary={}
    for arm in ARMS:
        summary[arm]={}
        for mode in MODES:
            summary[arm][mode]={}
            for label,kinds in [(kind,(kind,)) for kind in KINDS]+[("equal_type_overall",KINDS)]:
                values=[r for r in records if r["arm"]==arm and r["condition_mode"]==mode and r["kind"] in kinds]
                parents=[]
                for parent in PARENTS:
                    selected=[r for r in values if r["parent_index"]==parent]
                    parents.append(average([r.get("unguided_surface_symmetric_rms_bbox_pct") for r in selected]))
                summary[arm][mode][label]=dict(mean_bbox_pct=average(parents),per_parent_bbox_pct=parents,
                    computed_pairs=sum(r["status"]=="PASS" for r in values),expected_pairs=4*len(kinds))
    return dict(status="EXPLORATORY_ONLY",unique_arm_condition_two_noise_pairs=24,condition_scoring_records=48,
        source_samples="deterministic area sampling, 256 per surface; repeated conditioning pairs reuse one measurement",
        distance_oracle="evaluate.surface_distance_squared uses independent trimesh closest-point implementation",
        region="Source surface samples farther than 1% input-target bbox from every supplied primitive's 128 independent checkpoints",
        destination="Full other-noise raw triangle surface; selection applies to source samples in each direction",
        limitation="Checkpoint-based exclusion approximates the guide region. This geometric variation can be noise or fragments and does not establish plausible diversity or mesh quality.",
        statistical_unit="parent, with prompt types kept separate then equal-weight averaged",records=records,summary=summary)


def summarize(run_dir,with_diversity=True):
    run_dir=Path(run_dir)
    for name in ("paired_summary.json","condition_response.json"):
        if (run_dir/name).exists():
            raise FileExistsError("Refusing to overwrite existing summary: "+str(run_dir/name))
    execution,training,prompts,indexed=load_complete(run_dir)
    train_summary=training_summary(run_dir,training)
    scope=dict(phase=2,training_seed=10,train_parent_count=16,evaluation_parent_count=4,
        prompt_types=list(KINDS),noise_seeds=list(SEEDS),updates_per_arm=200,total_updates=400,
        adapted_rollouts_executed=96,adapted_unique_arm_prompt_noise_inputs=48,
        adapted_unique_inputs_per_arm=24,base_rollouts_executed=2,base_unique_noise_inputs=2,base_scoring_rows=24,
        statistical_unit="parent: first average the two noise outcomes, then compare paired parents",
        phase3="NOT_AUTHORIZED",architecture_advantage="NOT_ESTABLISHED_PHASE2_ONLY",
        uncertainty="Four fixed diagnostic parents and one training seed do not establish the preregistered Phase 3 advantage criterion.")
    gq={}
    descriptives={}
    for mode in MODES:
        gq[mode]={}
        for label,kinds in [(kind,(kind,)) for kind in KINDS]+[("equal_type_overall",KINDS)]:
            gq[mode][label]={metric:paired_metric(indexed,metric,mode,kinds) for metric in METRICS}
    for arm in (*ARMS,"base"):
        descriptives[arm]={}
        for mode in (("none",) if arm=="base" else MODES):
            descriptives[arm][mode]={label:metric_summary(indexed,arm,mode,kinds)
                for label,kinds in [(kind,(kind,)) for kind in KINDS]+[("equal_type_overall",KINDS)]}
    response_records,response_summary=condition_pairs(run_dir,indexed,prompts)
    repeat_records=[]
    for arm in ARMS:
        for parent in PARENTS:
            for kind in KINDS:
                for seed in SEEDS:
                    wrong=raw_arrays(run_dir,indexed[(arm,parent,kind,seed,"cyclic_mismatch")])
                    matching=raw_arrays(run_dir,indexed[(arm,(parent+1)%4,kind,seed,"correct")])
                    if not np.array_equal(wrong["initial_noise"],matching["initial_noise"]):
                        raise ValueError("Cyclic repeated-input noise mismatch")
                    a,b=wrong["tokens"],matching["tokens"]
                    delta=a.astype(np.float64)-b.astype(np.float64)
                    wrong_row=indexed[(arm,parent,kind,seed,"cyclic_mismatch")]
                    matching_row=indexed[(arm,(parent+1)%4,kind,seed,"correct")]
                    repeat_records.append(dict(arm=arm,parent_index=parent,condition_parent_index=(parent+1)%4,
                        kind=kind,noise_seed=seed,bitwise_equal=bool(np.array_equal(a,b)),
                        max_abs=float(np.max(np.abs(delta))),token_rms=float(np.sqrt(np.mean(delta**2))),
                        token_rms_input_bbox_pct=float(100*np.sqrt(np.mean(delta**2))/prompts[((parent+1)%4,kind)]["bbox_diagonal"]),
                        actual_guide_difference=wrong_row["guide_actual_rms_bbox_pct"]-matching_row["guide_actual_rms_bbox_pct"]))
    repeat_summary=dict(repeated_input_pairs=len(repeat_records),
        bitwise_equal_pairs=sum(r["bitwise_equal"] for r in repeat_records),
        bitwise_equal_fraction=float(np.mean([r["bitwise_equal"] for r in repeat_records])),
        max_abs=max(r["max_abs"] for r in repeat_records),records=repeat_records,
        explanation="At fixed N=512 and noise, cyclic mismatch at parent i has exactly the same model input as correct at parent (i+1)%4. The extra calls are repetitions, not independent inputs.")
    repeated_by_condition={(r["arm"],r["condition_parent_index"],r["kind"],r["noise_seed"]):r for r in repeat_records}
    for row in response_records:
        arm,parent,kind,seed=row["arm"],row["parent_index"],row["kind"],row["noise_seed"]
        first=repeated_by_condition[(arm,parent,kind,seed)]
        second=repeated_by_condition[(arm,(parent+1)%4,kind,seed)]
        reference=first["token_rms"]+second["token_rms"]
        row.update(same_input_first_condition_repeat_rms=first["token_rms"],
                   same_input_second_condition_repeat_rms=second["token_rms"],
                   two_endpoint_repeat_rms_sum=reference,
                   condition_change_rms_over_repeat_rms_sum=row["token_rms"]/reference if reference>0 else None,
                   condition_change_exceeds_observed_repeat_sum=row["token_rms"]>reference)
    variation={}
    for arm in ARMS:
        variation[arm]={}
        for label,kinds in [(kind,(kind,)) for kind in KINDS]+[("equal_type_overall",KINDS)]:
            selected=[r for r in response_records if r["arm"]==arm and r["kind"] in kinds]
            selected_repeat=[r for r in repeat_records if r["arm"]==arm and r["kind"] in kinds]
            parents=[]
            for parent in PARENTS:
                one=[r for r in selected if r["parent_index"]==parent]
                parents.append(dict(parent_index=parent,condition_change_rms_mean=average([r["token_rms"] for r in one]),
                    two_endpoint_repeat_rms_sum_mean=average([r["two_endpoint_repeat_rms_sum"] for r in one]),
                    mean_ratio=average([r["condition_change_rms_over_repeat_rms_sum"] for r in one])))
            ratios=[r["condition_change_rms_over_repeat_rms_sum"] for r in selected if r["condition_change_rms_over_repeat_rms_sum"] is not None]
            variation[arm][label]=dict(parent_count=4,parents=parents,
                condition_change_rms_mean=average([r["condition_change_rms_mean"] for r in parents]),
                same_input_repeat_rms_mean=average([r["token_rms"] for r in selected_repeat]),
                two_endpoint_repeat_rms_sum_mean=average([r["two_endpoint_repeat_rms_sum_mean"] for r in parents]),
                repeat_actual_guide_abs_difference_mean=average([abs(r["actual_guide_difference"]) for r in selected_repeat]),
                ratio_min=float(np.min(ratios)) if ratios else None,
                ratio_median=float(np.median(ratios)) if ratios else None,
                ratio_max=float(np.max(ratios)) if ratios else None,
                pairs_change_exceeds_observed_repeat_sum=sum(r["condition_change_exceeds_observed_repeat_sum"] for r in selected),
                comparison_pairs=len(selected),
                same_input_bitwise_equal_pairs=sum(r["bitwise_equal"] for r in selected_repeat))
    observed=any(row["token_max_abs"]>0 for row in response_records)
    repeat_variation=any(not row["bitwise_equal"] for row in repeat_records)
    response_status=("RESPONSE_NOT_SEPARATED_FROM_NUMERICAL_VARIATION" if repeat_variation else
                     "CONDITION_RESPONSE_ONLY" if observed else "NEEDS_LEARNING_DIAGNOSIS")
    print(json.dumps(dict(stage="repeat_input_audit",bitwise_equal_pairs=repeat_summary["bitwise_equal_pairs"],
        repeated_pairs=len(repeat_records),same_input_max_abs=repeat_summary["max_abs"],response_status=response_status,
        per_arm={arm:variation[arm]["equal_type_overall"] for arm in ARMS})),flush=True)
    quality_distinct={}
    for arm in (*ARMS,"base"):
        # A cyclic mismatch repeats the next parent's correct prompt with the
        # same noise/N. Keep exactly one output per unique conditioning input.
        distinct={row["name"]:row for row in indexed.values() if row["arm"]==arm and
                  (arm=="base" or row["condition_mode"]=="correct")}
        quality_distinct[arm]=dict(unique_conditioning_inputs=len(distinct),
            actually_executed_rollouts=2 if arm=="base" else 48,
            nonfinite_outputs=sum(row["nan_count"]>0 or row["inf_count"]>0 for row in distinct.values()),
            mean_min_angle_lt5_fraction=average([row["min_angle_lt5_fraction"] for row in distinct.values()]),
            mean_degenerate_fraction=average([row["degenerate_fraction"] for row in distinct.values()]),
            self_intersection_status="NOT_RUN",watertight_status="NOT_RUN",manifold_status="NOT_RUN")
    sampling=execution["sampling"]
    timing={}
    for arm in (*ARMS,"base"):
        values=[row["seconds"] for row in sampling if row.get("arm")==arm]
        timing[arm]=dict(completed_outputs=len(values),seconds_mean=float(np.mean(values)),
                        seconds_p50=float(np.quantile(values,.5)),seconds_p95=float(np.quantile(values,.95)),
                        limitation="Per-output sampling includes encoding and backbone; component timing and independently warmed benchmark were not separately instrumented.")
    evidence={name:sha256(run_dir/name) for name in ("per_output.csv","phase2_execution.json","prompt_manifest.json","split_manifest.json")}
    paired=dict(schema_version=1,status="PHASE2_DIAGNOSTIC_COMPLETE",scope=scope,evidence_sha256=evidence,
        training=train_summary,g_minus_q=gq,descriptives=descriptives,quality_unique_outputs=quality_distinct,
        timing=timing,actual_execution_counts={key:execution[key] for key in
            ("optimizer_updates","actual_backbone_forwards","actual_backbone_examples","complete_rollouts","complete_sampler_calls")},
        official_source_numerical_limitation=execution.get("numerical_reference_limitation"),
        claims=dict(implementation="Use completed precheck and training evidence; this summary is only Phase 2",
                    condition_response=response_status,causal_response="NOT_ESTABLISHED_FROM_CURRENT_GENERATION_COMPARISON",
                    architecture_advantage="NOT_ESTABLISHED",
                    G_vs_P="NOT_RUN_PHASE2",S_static="NOT_TRAINED_PHASE2",phase3_started=False),
        bootstrap_status="NOT_RUN: this is a four-parent single-seed wiring diagnostic, not the Phase 3 mechanism pilot")
    response=dict(schema_version=1,status=response_status,scope=scope,evidence_sha256=evidence,
        response_measure="Same initial noise and face slots; indexed raw token coordinates compared between correct and cyclically mismatched prompts",
        zero_threshold_note="Nonzero changed-condition differences alone do not establish causal response when the same input repeated also changes",
        causal_response="NOT_ESTABLISHED_FROM_CURRENT_GENERATION_COMPARISON",
        numerical_variation_comparison=variation,
        variation_comparison_caveat="The observed sum of two endpoint repeat RMS differences is a descriptive numerical-variation scale, not a confidence bound or a preregistered effect-size threshold. Ratios above it do not establish causality.",
        possible_cause="BF16 condition-encoder reductions and trajectory sensitivity are plausible sources; this CPU audit does not identify the responsible CUDA operation.",
        actual_guide_note="Each output is scored against its actual input prompt and that input parent's bbox; correct/mismatch actual scores reference different conditions",
        original_guide_note="Both correct and mismatch outputs are separately scored against the original parent's fixed prompt and bbox",
        quality_caveat="Condition sensitivity and lower guide distance do not establish good shapes, plausible diversity, or a geometry-specific architecture advantage",
        records=response_records,summary=response_summary,repeated_condition_inputs=repeat_summary,
        diversity=exploratory_diversity(run_dir,indexed,prompts) if with_diversity else {"status":"NOT_RUN_EXPLICITLY_DISABLED"})
    # Everything above must complete before either conclusion file is created.
    for name,value in (("paired_summary.json",paired),("condition_response.json",response)):
        with (run_dir/name).open("x",encoding="utf-8") as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write("\n")
    return paired,response


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",required=True,type=Path)
    parser.add_argument("--skip-diversity",action="store_true")
    args=parser.parse_args()
    try:
        paired,response=summarize(args.run_dir,not args.skip_diversity)
    except IncompleteRun as error:
        print(json.dumps(dict(status="NOT_READY",reason=str(error),summary_files_written=False)),file=sys.stderr)
        return 2
    print(json.dumps(dict(status=paired["status"],condition_response=response["status"],
                         architecture_advantage="NOT_ESTABLISHED_PHASE2_ONLY",phase3="NOT_AUTHORIZED")))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
