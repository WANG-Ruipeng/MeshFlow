"""Read saved teacher tables only. Standard library; no scientific imports/calls."""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
from collections import defaultdict, Counter
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import time

ROOT = Path("/mnt/e/MeshFlow")
ARMS = ("T1", "Geo", "Full")
SPLITS = ("train", "validation")
TIMES = (0.2, 0.5, 0.8)
PHASES = (0, 1000, 2500, 3000)
METRICS = ("FM_error", "D_edge", "L5")
PCT_EPS = 1e-12
NA = "NA"

def sha(path):
    with path.open("rb") as f:
        h = hashlib.sha256()
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
        return h.hexdigest()

def csv_read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def number(x):
    try:
        z = float(x)
        return z if math.isfinite(z) else None
    except (ValueError, TypeError):
        return None

def mean(values):
    good = [x for x in values if x is not None]
    return statistics.fmean(good) if good else None

def pct(start, end):
    if start is None or end is None:
        return None, "MISSING_VALUE"
    if abs(start) <= PCT_EPS:
        return None, "NEAR_ZERO_BASELINE"
    return 100.0 * (end - start) / start, "OK"

def sign(x):
    if x is None:
        return "NA"
    return "UP" if x > 0 else ("DOWN" if x < 0 else "FLAT")

def write_csv(path, rows):
    if not rows:
        raise ValueError("Empty output: " + str(path))
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: (NA if row.get(k) is None else row.get(k, NA)) for k in fields})

def parent_mean(rows, metric):
    by = defaultdict(list)
    for r in rows:
        if r["status"] == "PASS":
            z = number(r[metric])
            if z is not None:
                by[r["uid"]].append(z)
    pm = {uid: mean(v) for uid, v in by.items()}
    return mean(pm.values()), len(pm), sum(len(v) for v in by.values())

def matched_summary(rows):
    # Cases/arms are nested within parent; parents receive equal weight.
    by = defaultdict(list)
    for r in rows:
        if r["start_value"] is not None and r["end_value"] is not None:
            by[r["uid"]].append(r)
    pv = []
    for uid, records in by.items():
        start = mean([r["start_value"] for r in records])
        end = mean([r["end_value"] for r in records])
        pc, reason = pct(start, end)
        pv.append(dict(uid=uid, start=start, end=end, delta=end-start, pct=pc, reason=reason))
    start = mean([r["start"] for r in pv])
    end = mean([r["end"] for r in pv])
    pc, reason = pct(start, end)
    deltas = [r["delta"] for r in pv]
    return dict(
        requested_case_pairs=len(rows), valid_case_pairs=sum(len(v) for v in by.values()),
        requested_parents=len({r["uid"] for r in rows}), valid_paired_parents=len(pv),
        start_parent_mean=start, end_parent_mean=end,
        paired_mean_delta=mean(deltas),
        paired_median_delta=statistics.median(deltas) if deltas else None,
        percent_change_of_parent_mean=pc, percent_status=reason,
        mean_parent_percent_change=mean([r["pct"] for r in pv]),
        valid_parent_percent_denominator=sum(r["pct"] is not None for r in pv),
        parents_decreased=sum(x < 0 for x in deltas),
        parents_increased=sum(x > 0 for x in deltas),
        parents_unchanged=sum(x == 0 for x in deltas),
    )

def run(source, out, feedback):
    start_time = time.time()
    out.mkdir(parents=True, exist_ok=True)
    assert not out.resolve().is_relative_to(source.resolve())
    source_paths = [source/n for n in (
        "teacher_per_case.csv", "teacher_summary.csv", "shared_input_generation_curves.csv",
        "protocol.json", "panels.json")]
    source_paths += [feedback, ROOT/"experiments/mf_h1_local0/chair_domain_control_v1/reporting.py"]
    panels = json.loads((source/"panels.json").read_text())
    protocol = json.loads((source/"protocol.json").read_text())
    source_paths += [source/r["input_path"] for r in panels["teacher"]]
    source_paths = sorted(set(source_paths))
    source_before = {str(p): dict(sha256=sha(p), bytes=p.stat().st_size) for p in source_paths}
    assert sha(source/"panels.json") == protocol["frozen_files"]["panels.json"]
    teachers = csv_read(source/"teacher_per_case.csv")
    old_summary = csv_read(source/"teacher_summary.csv")
    generations = csv_read(source/"shared_input_generation_curves.csv")
    assert len(teachers) == 288 and len(old_summary) == 72 and len(generations) == 18
    panel_t = {r["case_id"]: r for r in panels["teacher"]}
    assert len(panel_t) == 24
    series = defaultdict(dict)
    for r in teachers:
        r["phase_step"] = int(r["phase_step"])
        r["conditional_total_step"] = int(r["conditional_total_step"])
        r["t"] = float(r["t"])
        assert r["arm"] in ARMS and r["split"] in SPLITS and r["t"] in TIMES
        assert r["phase_step"] in PHASES and r["conditional_total_step"] == 1000+r["phase_step"]
        p = panel_t[r["case_id"]]
        assert (r["uid"], r["split"], r["t"], r["input_sha256"]) == (p["uid"], p["split"], p["t"], p["input_sha256"])
        assert sha(source/p["input_path"]) == r["input_sha256"]
        key = (r["arm"], r["split"], r["t"], r["uid"], r["case_id"], r["input_sha256"])
        assert r["phase_step"] not in series[key], "Duplicate teacher observation"
        series[key][r["phase_step"]] = r
    assert len(series) == 72
    for key, rows in series.items():
        assert set(rows) == set(PHASES), "Missing checkpoint for exact input key"
    groups = {}
    for arm, split, t, phase in itertools.product(ARMS, SPLITS, TIMES, PHASES):
        rr = [r for r in teachers if (r["arm"],r["split"],r["t"],r["phase_step"]) == (arm,split,t,phase)]
        assert len(rr) == 4 and len({r["uid"] for r in rr}) == 4
        groups[arm,split,t,phase] = rr

    reconciled = []
    curves = []
    for r in old_summary:
        key = r["arm"],r["split"],float(r["t"]),int(r["phase_step"])
        rows = groups[key]
        for metric in (*METRICS, "Yhat_free_area"):
            value, parents, valid_cases = parent_mean(rows, metric)
            reported = number(r[metric])
            ok = (value is None and reported is None) or (
                value is not None and reported is not None and
                math.isclose(value, reported, rel_tol=1e-12, abs_tol=1e-15))
            counts_ok = int(r[metric+"_valid_parents"]) == parents and int(r[metric+"_valid_cases"]) == valid_cases
            reconciled.append(dict(arm=key[0], split=key[1], t=key[2], phase_step=key[3],
                metric=metric, recomputed_parent_mean=value, reported_parent_mean=reported,
                numeric_difference=None if value is None or reported is None else value-reported,
                valid_parents=parents, valid_cases=valid_cases, status="PASS" if ok and counts_ok else "FAIL"))
        assert int(r["requested_parents"]) == 4 and int(r["requested_cases"]) == 4
        assert int(r["completed_generations"]) == 0
        assert int(r["completed_cases"]) == sum(x["status"] == "PASS" for x in rows)
    assert all(r["status"] == "PASS" for r in reconciled), "Teacher summary mismatch"
    for arm,split,t,phase,metric in itertools.product((*ARMS,"ALL_ARMS"),SPLITS,TIMES,PHASES,METRICS):
        rows = [r for r in teachers if (arm=="ALL_ARMS" or r["arm"]==arm) and
                (r["split"],r["t"],r["phase_step"])==(split,t,phase)]
        val, parents, cases = parent_mean(rows,metric)
        curves.append(dict(arm=arm,split=split,t=t,phase_step=phase,conditional_total_step=phase+1000,
            metric=metric,parent_equal_mean=val,requested_parents=len({r["uid"] for r in rows}),
            valid_parents=parents,requested_cases=len(rows),valid_cases=cases,
            aggregation="equal parents; within parent equal registered cases/arms"))

    intervals = tuple(itertools.combinations(PHASES,2))
    pairs = []
    for key, observations in sorted(series.items()):
        arm,split,t,uid,case_id,input_sha = key
        for a,b in intervals:
            ra,rb = observations[a],observations[b]
            for metric in METRICS:
                va = number(ra[metric]) if ra["status"]=="PASS" else None
                vb = number(rb[metric]) if rb["status"]=="PASS" else None
                pc, reason = pct(va,vb)
                pairs.append(dict(arm=arm,split=split,t=t,uid=uid,case_id=case_id,input_sha256=input_sha,
                    metric=metric,from_phase=a,to_phase=b,from_total=a+1000,to_total=b+1000,
                    start_status=ra["status"],end_status=rb["status"],
                    start_value=va,end_value=vb,paired_delta=None if va is None or vb is None else vb-va,
                    percent_change=pc,percent_status=reason))
    changes = []
    for arm,split,t,metric,(a,b) in itertools.product((*ARMS,"ALL_ARMS"),SPLITS,TIMES,METRICS,intervals):
        rows = [r for r in pairs if (arm=="ALL_ARMS" or r["arm"]==arm) and
                (r["split"],r["t"],r["metric"],r["from_phase"],r["to_phase"])==(split,t,metric,a,b)]
        changes.append(dict(arm=arm,split=split,t=t,metric=metric,from_phase=a,to_phase=b,
            from_total=a+1000,to_total=b+1000,**matched_summary(rows)))
    change_index = {(r["arm"],r["split"],r["t"],r["metric"],r["from_phase"],r["to_phase"]):r for r in changes}
    trajectory = []
    for arm,split,t,metric in itertools.product((*ARMS,"ALL_ARMS"),SPLITS,TIMES,METRICS):
        cc = [next(r for r in curves if (r["arm"],r["split"],r["t"],r["phase_step"],r["metric"])==
                   (arm,split,t,p,metric)) for p in PHASES]
        vv = [r["parent_equal_mean"] for r in cc]
        dd = [b-a if a is not None and b is not None else None for a,b in zip(vv,vv[1:])]
        signs = [sign(d) for d in dd]
        nonmono = "UP" in signs and "DOWN" in signs
        one = dict(arm=arm,split=split,t=t,metric=metric,
            adjacent_direction="/".join(signs),nonmonotonic=nonmono,
            interpretation="direction reversal across saved checkpoints; no significance test")
        one.update({"mean_phase"+str(p):v for p,v in zip(PHASES,vv)})
        one.update({"delta_"+str(a)+"_"+str(b):d for (a,b),d in zip(zip(PHASES,PHASES[1:]),dd)})
        # Per-parent curves are kept separate from group-mean reversal.
        parent_trajectories = []
        for uid in sorted({r["uid"] for r in teachers if r["split"]==split}):
            parent_values = []
            for p in PHASES:
                rr = [r for r in teachers if (arm=="ALL_ARMS" or r["arm"]==arm) and
                      (r["split"],r["t"],r["phase_step"],r["uid"])==(split,t,p,uid) and r["status"]=="PASS"]
                parent_values.append(mean([number(r[metric]) for r in rr]))
            if all(v is not None for v in parent_values):
                ps = {sign(b-a) for a,b in zip(parent_values,parent_values[1:])}
                parent_trajectories.append("UP" in ps and "DOWN" in ps)
        one["valid_parent_trajectories"] = len(parent_trajectories)
        one["nonmonotonic_parent_trajectories"] = sum(parent_trajectories)
        trajectory.append(one)

    claims = []
    for arm,metric in itertools.product((*ARMS,"ALL_ARMS"),METRICS):
        for t,train_sign,val_sign in ((0.2,"DOWN","UP"),(0.8,"UP","DOWN")):
            tr=change_index[arm,"train",t,metric,0,3000]
            va=change_index[arm,"validation",t,metric,0,3000]
            claims.append(dict(arm=arm,metric=metric,t=t,split="train_and_validation",
                claim="endpoint training "+train_sign+" and validation "+val_sign,
                observed_train=sign(tr["paired_mean_delta"]),observed_validation=sign(va["paired_mean_delta"]),
                train_percent_change=tr["percent_change_of_parent_mean"],
                validation_percent_change=va["percent_change_of_parent_mean"],
                claim_holds=sign(tr["paired_mean_delta"])==train_sign and sign(va["paired_mean_delta"])==val_sign,
                scope="start versus endpoint; not every consecutive interval"))
        for split in SPLITS:
            tr=next(r for r in trajectory if (r["arm"],r["split"],r["t"],r["metric"])==(arm,split,0.5,metric))
            claims.append(dict(arm=arm,metric=metric,t=0.5,split=split,
                claim="both upward and downward adjacent changes",claim_holds=tr["nonmonotonic"],
                adjacent_direction=tr["adjacent_direction"],scope="four saved checkpoint means"))

    initial_A = [r for r in panels["A"] if 0 in r["phase_steps"]]
    assert all(r["ratio"]==0.2 and r["noise_index"]==0 for r in initial_A)
    membership = []
    overlaps = []
    for split in SPLITS:
        ts={r["uid"] for r in panels["teacher"] if r["split"]==split}
        gs={r["uid"] for r in initial_A if r["split"]==split and r["status"]=="READY"}
        bs={r["uid"] for r in panels["B"] if r["split"]==split and r["status"]=="READY"}
        teacher_hash={r["input_sha256"] for r in panels["teacher"] if r["split"]==split}
        generation_hash={r["input_sha256"] for r in initial_A if r["split"]==split and r["status"]=="READY"}
        for uid in sorted(ts|gs|bs):
            membership.append(dict(split=split,uid=uid,teacher_parent=uid in ts,shared_A20_generation_parent=uid in gs,
                B_width_parent=uid in bs,
                teacher_task_ids=";".join(sorted({r["task_id"] for r in panels["teacher"] if r["uid"]==uid})),
                shared_generation_task_ids=";".join(sorted({r["task_id"] for r in initial_A if r["uid"]==uid}))))
        overlaps.append(dict(split=split,teacher_parents=len(ts),shared_generation_parents=len(gs),
            common_parent_count=len(ts&gs),teacher_only_parents=sorted(ts-gs),generation_only_parents=sorted(gs-ts),
            parent_sets_identical=ts==gs,teacher_parents_subset_of_generation=ts<=gs,
            teacher_B_parent_sets_identical=ts==bs,shared_full_input_hashes=len(teacher_hash&generation_hash),
            teacher_condition_tasks=sorted({r["task_id"] for r in panels["teacher"] if r["split"]==split}),
            generation_condition_tasks=sorted({r["task_id"] for r in initial_A if r["split"]==split})))
    generation_schedule=[]
    for arm,split,phase in itertools.product(ARMS,SPLITS,PHASES):
        rr=[r for r in generations if (r["arm"],r["split"],int(r["phase_step"]))==(arm,split,phase)]
        assert len(rr)<=1
        generation_schedule.append(dict(arm=arm,split=split,phase_step=phase,conditional_total_step=phase+1000,
            measurement_available=bool(rr),requested_parents=rr[0]["requested_parents"] if rr else None,
            valid_parents=rr[0]["valid_parents"] if rr else None,
            explanation="measured shared A20 first-noise requests" if rr else "not measured; never interpolate as observations"))
    assert all(len({r["input_sha256"] for r in teachers if r["case_id"]==case})==1 for case in panel_t)

    outputs={
        "teacher_summary_reconciliation.csv":reconciled,
        "teacher_group_curves.csv":curves,
        "teacher_paired_changes_per_case.csv":pairs,
        "teacher_paired_changes_summary.csv":changes,
        "teacher_endpoint_changes.csv":[r for r in changes if r["from_phase"]==0 and r["to_phase"]==3000],
        "teacher_trajectory_directions.csv":trajectory,
        "cloud_claim_checks.csv":claims,
        "teacher_generation_parent_membership.csv":membership,
        "generation_measurement_schedule.csv":generation_schedule,
    }
    for name, rows in outputs.items(): write_csv(out/name,rows)
    def pp(v):
        return NA if v is None else f"{v:+.6f}%"
    report=[
        "# Teacher 数值核对",
        "",
        "本次只分析已有 CSV、协议、输入登记和输入文件哈希。没有训练、模型前向、生成、OT、评分或绘图。",
        "",
        "## 数据与配对",
        "",
        "288 条 teacher 记录 = 3 臂 × 4 检查点 × 2 划分 × 4 父对象 × 3 个 t。所有记录均按 arm、split、t、uid、case_id、input_sha256 精确配对；72 条输入轨迹均完整。24 个 teacher 输入文件的实际 SHA256 与原冻结登记一致。",
        "检查点 phase 0/1000/2500/3000 对应累计 1000/2000/3500/4000 步。逐父等权重算的 FM、Dedge、L5 和面积共 288 项汇总均与原 teacher_summary.csv 一致。",
        "每条配对先做终点减起点，再按父对象等权汇总；配对差中位数由逐父差计算。ALL_ARMS 是同一批父对象上三臂的描述性平均，仍只有每 split 4 个独立父对象，不能计作 12 个独立对象。",
        "百分比为 100×(后值−前值)/前值。基值绝对值 ≤ 1e−12 或缺失时写 NA，同时保留绝对差和分母状态。该阈值只是此次除法保护，不改变原实验指标。CSV 保留 Python 双精度数值，下面百分比仅为展示而四舍五入。",
        "",
        "## 起终点变化",
        "",
        "原 fixed_teacher_curves.png 的纵轴是 FM_error，不是三项 loss 的合成值。三项必须分开解释；负变化表示该诊断值降低。",
    ]
    for metric in METRICS:
        report += ["",f"### {metric}","","| 臂 | t=.2 训练 | t=.2 验证 | t=.5 训练 | t=.5 验证 | t=.8 训练 | t=.8 验证 |",
                   "|---|---:|---:|---:|---:|---:|---:|"]
        for arm in (*ARMS,"ALL_ARMS"):
            vals=[pp(change_index[arm,s,t,metric,0,3000]["percent_change_of_parent_mean"]) for t in TIMES for s in SPLITS]
            report.append("| "+arm+" | "+" | ".join(vals)+" |")
    report += [
        "",
        "## 云端三项判断",
        "",
        "FM：t=.2 训练下降、验证上升，t=.8 训练上升、验证下降，在 T1、Geo、Full 和三臂等权均值上都成立。t=.5 的训练与验证均值在各臂及三臂均值上都发生方向反转；这确认离散检查点上的非单调性，不构成统计显著性判断。",
        "Dedge：t=.2 同样是训练下降、验证上升；但 t=.8 三臂训练和验证都下降，所以不能将 FM 的反向模式推广到 Dedge。t=.5 各臂两种划分均非单调。",
        "L5：t=.2 的方向与 FM 相同。t=.8 三臂训练都上升，但验证 T1 上升约 1.199873%，Geo 下降约 4.942799%，Full 下降约 0.387968%；三臂均值验证下降。因此 t=.8 的反向模式仅对 Geo、Full 和三臂均值成立，不对 T1 成立。t=.5 各臂两种划分均非单调。",
        "以上成立范围是固定父集合的均值，并非每个父对象都同向。以 ALL_ARMS 的 FM t=.8 为例，训练和验证各自都是 2 个父对象上升、2 个下降；均值仍分别上升和下降。百分比表是均值之比的变化，不是逐父百分比的平均。上述方向主要指 phase0 到 phase3000。中间区间不能省略：例如 FM 在 t=.2 的训练均值最后一个区间三臂都回升，验证均值最后一个区间三臂都回落。起终点方向不能写成全程单调。",
        "",
        "### t=.5 的 FM 原始均值",
        "",
        "| 臂/划分 | phase0 | phase1000 | phase2500 | phase3000 | 相邻变化 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for arm,split in itertools.product((*ARMS,"ALL_ARMS"),SPLITS):
        r=next(x for x in trajectory if (x["arm"],x["split"],x["t"],x["metric"])==(arm,split,0.5,"FM_error"))
        report.append("| "+arm+"/"+split+" | "+" | ".join(format(r["mean_phase"+str(p)],".12g") for p in PHASES)+" | "+r["adjacent_direction"]+" |")
    report += [
        "",
        "## Teacher 与完整生成不能作逐例因果匹配",
        "",
        "teacher 每个划分 4 个父对象，共有输入生成曲线每个划分 8 个父对象。两种划分中 teacher 的 4 个父对象都是生成 8 父集合的子集，集合并不相同。与 B 宽度面板的父集合相同，也不代表输入状态相同。",
        "训练 teacher 使用保留的 train20_0 条件，而共有生成曲线使用新 A20 条件；验证 teacher 使用其对应父对象的同一 A20 条件任务，但 teacher 是含 GT 的固定插值状态并经过原配对，生成从完整高斯状态积分。teacher 和共有生成请求没有相同的完整 input_sha256。相同 UID 不能消除条件、噪声、状态和父集合差异，因此本报告不计算二者逐例相关性，更不推断因果。",
        "生成曲线训练父只测 phase0/3000，phase1000/2500 为 NA；验证父才有全部四个点。训练图连线不是中间测量，也不能用 teacher 曲线补造中间生成点。",
        "",
        "## 解释边界与复现",
        "",
        "本轮协议的 Full ramp 恒为 1，原辅助项门控为 FP32 t>=0.5，生成器使用固定 Euler50。teacher 在三个 t 上记录诊断值，不应把 t=.2 的辅助诊断值误认成当时参与训练的加权 loss。这里只核对已保存数值，不从曲线方向认定门控、时间权重或求解器是病因，也不提出由本分析自动触发的新实验。",
        "运行：OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' /home/warpwang/miniforge3/envs/meshflow/bin/python -B teacher_audit.py。可用 --source、--out、--feedback 显式指定路径。",
        "精确值见 teacher_paired_changes_per_case.csv、teacher_paired_changes_summary.csv 和 teacher_endpoint_changes.csv；全部相邻反转见 teacher_trajectory_directions.csv；判断适用范围见 cloud_claim_checks.csv；父对象与输入差异见 teacher_generation_parent_membership.csv；原表重算见 teacher_summary_reconciliation.csv。",
        "audit.json 保存输入 SHA256、运行时长、行数、除法保护计数和零实验调用声明。原文件运行前后哈希保持一致。",
    ]
    (out/"TEACHER_AUDIT_REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    source_after={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in source_paths}
    assert source_before==source_after,"Source files changed"
    execution=dict(model_loads=0,model_forwards=0,model_backwards=0,optimizer_updates=0,
                   generations=0,OT_calls=0,scoring_calls=0,GPU_calls=0)
    audit=dict(schema="chair_control_teacher_followup_v1",status="PASS",
        completed_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.time()-start_time,
        script_sha256=sha(Path(__file__)),source_files=source_before,source_hashes_unchanged=True,
        teacher_records=len(teachers),teacher_statuses=dict(Counter(r["status"] for r in teachers)),
        exact_input_trajectories=len(series),unique_teacher_inputs=len(panel_t),
        all_trajectories_have_all_four_phases=True,summary_numeric_checks=len(reconciled),
        summary_numeric_check_failures=sum(r["status"]!="PASS" for r in reconciled),
        percentage_definition="100*(end-start)/start; no finite input or abs(start)<=1e-12 => NA",
        near_zero_threshold=PCT_EPS,
        per_case_percent_status_counts=dict(Counter(r["percent_status"] for r in pairs)),
        parent_weighting="equal parents; within parent equal registered cases and (ALL_ARMS only) arms",
        parent_overlaps=overlaps,thread_environment={k:os.environ[k] for k in (
            "OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","CUDA_VISIBLE_DEVICES")},
        execution_counts=execution,output_rows={name:len(rows) for name,rows in outputs.items()},
        claim_checks=claims,
        artifacts=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir())
                   if p.is_file() and p.name!="audit.json"])
    (out/"audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(status="PASS",teacher_records=len(teachers),summary_checks=len(reconciled),
                         output_rows=audit["output_rows"],execution_counts=execution,
                         source_hashes_unchanged=True,wall_seconds=audit["wall_seconds"]),indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROOT/"native_t1/runs/chair_domain_control_v1")
    parser.add_argument("--out",type=Path,default=ROOT/"native_t1/analysis/chair_control_followup_v1/teacher_audit")
    parser.add_argument("--feedback",type=Path,default=ROOT/"native_t1/analysis/chair_control_followup_v1/cloud_feedback.txt")
    args=parser.parse_args()
    run(args.source,args.out,args.feedback)
