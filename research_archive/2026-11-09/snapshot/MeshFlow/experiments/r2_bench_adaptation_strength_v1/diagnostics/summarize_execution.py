"""Read existing small ledgers/receipts only. No model, mesh or scoring imports."""
from pathlib import Path
import collections
import datetime as dt
import hashlib
import json

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "diagnostics"
SOURCES = {}


def read(path):
    path = Path(path)
    data = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def lines(path):
    path = Path(path)
    data = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return [json.loads(x) for x in data.splitlines() if x.strip()]


def utc(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat()


def instant(row):
    if "unix" in row:
        return row["unix"]
    if "time_unix" in row:
        return row["time_unix"]
    return dt.datetime.fromisoformat(row["utc"].replace("Z", "+00:00")).timestamp()


def gpu(snapshot):
    raw = snapshot.get("gpu", {}).get("stdout", "").strip()
    try:
        name, total, used, free, util = [x.strip() for x in raw.split(",")]
        return dict(name=name, total_mib=float(total.split()[0]),
                    used_mib=float(used.split()[0]), free_mib=float(free.split()[0]),
                    utilization_percent=float(util.split()[0]))
    except (ValueError, IndexError):
        return {"raw": raw}


def peak_overlap(intervals):
    events = [(a, 1) for a, b in intervals] + [(b, -1) for a, b in intervals]
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    return peak


accounting = read(ROOT / "training_summary/20261020T113914_478591Z/accounting.json")
assert accounting["status"] == "PASS"
resource_initial = read(ROOT / "resources_before_preparation.json")
training = []
train_intervals = []
for name, arm in accounting["arms"].items():
    status = arm["status_snapshot"]
    events = lines(ROOT / "training" / name / "calls.jsonl")
    start, end = min(map(instant, events)), max(map(instant, events))
    train_intervals.append((start, end))
    resource = arm["timing_and_resources"]
    training.append({
        "arm": name, "local_updates": arm["local_updates_observed"],
        "history_updates_before_this_run": arm["source_global_step"],
        "cumulative_updates": arm["cumulative_generator_updates_observed"],
        "fresh_AdamW": status["fresh_optimizer"],
        "calls": arm["call_accounting"], "head_calls": status["head_calls"],
        "elapsed_seconds": status["elapsed_seconds"],
        "model_call_span_start_utc": utc(start), "model_call_span_end_utc": utc(end),
        "model_call_span_seconds": end-start,
        "updates_per_second_including_probe_and_checkpoint": 1500/status["elapsed_seconds"],
        "resources": resource,
        "completed_sample_exposures": arm["completed_exposure_rows"],
    })

labels = ["COMMON_S0", "OFFICIAL_S0", "A_S500", "B_S500", "C_S500",
          "A_S1500", "B_S1500", "C_S1500"]
label_rows = []
raw_intervals, d0_intervals = [], []
raw_counts, d0_counts = collections.Counter(), collections.Counter()
raw_alloc, raw_reserved, d0_alloc, d0_reserved = [], [], [], []
eval_snapshots = []
for label in labels:
    base = ROOT / "evaluation" / label
    subset = "full" if label.endswith("1500") else "process"
    raw = read(base / f"raw_complete_{subset}.json")
    post = read(base / f"post_complete_{subset}.json")
    raw_receipts = sorted(base.glob("*/raw/receipt.json"))
    post_receipts = sorted(base.glob("*/post/proposal_receipt.json"))
    assert len(raw_receipts) == raw["outputs"] == len(post_receipts) == post["outputs"]
    times_raw, times_d0 = [], []
    for p in raw_receipts:
        receipt = read(p)
        assert receipt["generator_forwards"] == 50 and receipt["head_forwards"] == 0
        ev = lines(p.parent / "calls.jsonl")
        counts = collections.Counter(x["event"] for x in ev)
        assert counts["FORWARD_STARTED"] == counts["FORWARD_COMPLETE"] == 50
        assert set(counts) == {"FORWARD_STARTED", "FORWARD_COMPLETE"}
        raw_counts.update(counts)
        times_raw.extend(map(instant, ev))
        raw_alloc.append(receipt["audit"]["peak_allocated_bytes"])
        raw_reserved.append(receipt["audit"]["peak_reserved_bytes"])
    for p in post_receipts:
        receipt = read(p)
        assert receipt["D0_forwards"] == 1
        ev = lines(p.parent / "calls.jsonl")
        counts = collections.Counter(x["event"] for x in ev)
        assert counts["D0_FORWARD_STARTED"] == counts["D0_FORWARD_RETURNED"] == 1
        assert set(counts) == {"D0_FORWARD_STARTED", "D0_FORWARD_RETURNED"}
        d0_counts.update(counts)
        times_d0.extend(map(instant, ev))
        d0_alloc.append(receipt["peak_allocated_bytes"])
        d0_reserved.append(receipt["peak_reserved_bytes"])
    raw_intervals.append((min(times_raw), max(times_raw)))
    d0_intervals.append((min(times_d0), max(times_d0)))
    cpu_raw = read(ROOT / "evaluation/completions" / f"raw__{label}.json")
    cpu_post = read(ROOT / "evaluation/completions" / f"post__{label}.json")
    label_rows.append({
        "label": label, "outputs": raw["outputs"], "generator_forwards": raw["new_forwards"],
        "D0_forwards": post["new_D0_forwards"],
        "GEO_calls": post["new_GEO_calls"], "raw_seconds": raw["seconds"],
        "D0_GEO_seconds": post["seconds"],
        "raw_score_seconds": cpu_raw["seconds"], "post_chain_and_scores_seconds": cpu_post["seconds"],
        "raw_score_workers": cpu_raw["workers"], "post_workers": cpu_post["workers"],
        "raw_forward_start_utc": utc(min(times_raw)), "raw_forward_end_utc": utc(max(times_raw)),
        "D0_forward_start_utc": utc(min(times_d0)), "D0_forward_end_utc": utc(max(times_d0)),
    })
    for attempt in sorted(base.glob("*attempt*")):
        if attempt.is_dir():
            for p in sorted(attempt.glob("resources*.json")):
                snapshot = read(p)
                eval_snapshots.append({"source": str(p.relative_to(ROOT)), "gpu": gpu(snapshot)})

dispatch = []
all_snapshots = [resource_initial]
for p in sorted((ROOT / "dispatch").glob("job_*.json")):
    job = read(p)
    snap = job["resources"]
    all_snapshots.append(snap)
    dispatch.append({
        "stage": job["stage"], "identity": job["identity"],
        "pid": job["pid"], "process_created_utc": utc(job["process_created"]),
        "estimated_process_peak_mib": job.get("estimated_process_peak_mib"),
        "headroom_mib": job.get("headroom_mib"),
        "incremental_slots_estimate": job.get("incremental_slots_estimate"),
        "actual_owned_GPU_workers_at_dispatch": job.get("actual_owned_GPU_workers"),
        "gpu": gpu(snap), "CPU_cores": snap.get("CPU_cores"),
        "RAM_available_bytes": snap.get("RAM_available"),
        "automatic_retry": job.get("automatic_retry"),
    })
extensions = []
for p in sorted((ROOT / "workflow").glob("cpu_parallel_extension_*.json")):
    ext = read(p)
    all_snapshots.append(ext["resources"])
    extensions.append({
        "source": str(p.relative_to(ROOT)), "utc": ext["utc"], "label": ext["target_label"],
        "workers": ext["workers"], "total_pool_limit": ext["total_pool_limit"],
        "total_worker_cap": ext["total_worker_cap"],
        "active_other_pools_at_dispatch": len(ext["active_owned_pools"]),
        "reason": ext["reason"], "model_calls": ext["model_calls"],
        "replay": ext["replay"], "new_samples": ext["new_samples"],
    })
cpu_jobs = []
cpu_intervals = []
for p in sorted((ROOT / "workflow/jobs").glob("job_*.json")):
    job = read(p)
    all_snapshots.append(job["resources"])
    stage, label = job["stage"], job["identity"]
    if stage in ("cpu_raw", "cpu_post"):
        receipt = ROOT / "evaluation/completions" / f"{stage[4:]}__{label}.json"
        done = read(receipt)
        # Explicitly approximate: dispatch start includes interpreter/import overhead,
        # receipt mtime is the completion-write timestamp, not process exit.
        cpu_intervals.append((job["process_created"], receipt.stat().st_mtime))
        cpu_jobs.append({"stage": stage, "label": label, "workers": done["workers"],
                         "seconds": done["seconds"], "dispatch_utc": utc(job["process_created"]),
                         "receipt_write_utc": utc(receipt.stat().st_mtime)})
workflow = lines(ROOT / "workflow/ledger.jsonl")
gallery = read(ROOT / "gallery/completion.json")
visual = read(ROOT / "visual_review/completion.json")
v2 = read(ROOT / "results_v2/completion.json")
revision = read(ROOT / "diagnostics/collector_v2_revision.json")
assert sum(x["outputs"] for x in label_rows) == 224
assert raw_counts["FORWARD_COMPLETE"] == 11200
assert d0_counts["D0_FORWARD_RETURNED"] == 224
assert v2["rows"] == 672 and gallery["sheets"] == 144
all_gpu = [gpu(x) for x in all_snapshots] + [x["gpu"] for x in eval_snapshots]
head_calls = sum(x["head_calls"] for x in training)
training_calls = {key: sum(x["calls"]["completed"].get(key, 0) for x in training)
                  for key in ("train_forward", "train_backward", "optimizer_update", "probe_forward")}
compute_done = [x for x in workflow if x["event"] == "COMPUTE_COMPLETE"][0]
max_device = max(
    [x["resources"]["measured_device_used_peak_mib"] for x in training] +
    [x["used_mib"] for x in all_gpu if "used_mib" in x])
summary = {
    "schema": "BENCH_ADAPTATION_EXECUTION_SUMMARY_V1", "status": "PASS",
    "scope": "Read-only bookkeeping from existing receipts; no extra model or score calls",
    "initial_resources": {"utc": resource_initial["utc"], "gpu": gpu(resource_initial),
                          "CPU_cores": resource_initial["CPU_cores"],
                          "RAM_total_bytes": resource_initial["RAM_total"],
                          "RAM_available_bytes": resource_initial["RAM_available"]},
    "training": training,
    "budget": {**training_calls, "generation_rollouts": 224, "generator_sampling_forwards": 11200,
               "single_D0_forwards": 224, "GEO_outputs": 224, "scored_cells": v2["rows"],
               "rendered_sheets": gallery["sheets"], "head_forwards": head_calls,
               "total_neural_forwards": training_calls["train_forward"] +
                  training_calls["probe_forward"] + 11200 + 224,
               "failed_or_unresolved_training_calls": sum(len(x["calls"]["failed"]) +
                  len(x["calls"]["pending_call_ids"]) for x in training),
               "RAW_ledger_events": dict(raw_counts), "D0_ledger_events": dict(d0_counts)},
    "labels": label_rows, "GPU_dispatch_records": dispatch, "CPU_jobs": cpu_jobs,
    "CPU_parallel_extensions": extensions,
    "concurrency": {
        "max_training_model_call_span_overlap": peak_overlap(train_intervals),
        "max_RAW_model_call_span_overlap": peak_overlap(raw_intervals),
        "max_D0_model_call_span_overlap": peak_overlap(d0_intervals),
        "max_all_neural_job_call_span_overlap": peak_overlap(train_intervals+raw_intervals+d0_intervals),
        "max_owned_GPU_workers_recorded_at_dispatch": max(x["actual_owned_GPU_workers_at_dispatch"] or 0 for x in dispatch),
        "max_CPU_pool_overlap_dispatch_to_receipt_approx": peak_overlap(cpu_intervals),
        "CPU_pool_worker_max": max(x["workers"] for x in cpu_jobs),
        "CPU_extension_total_worker_cap": max(x["total_worker_cap"] for x in extensions),
        "scope": "Call-span overlap is active-job overlap, not simultaneous CUDA-kernel occupancy.",
    },
    "timing": {
        "first_training_call_utc": utc(min(x[0] for x in train_intervals)),
        "last_training_call_utc": utc(max(x[1] for x in train_intervals)),
        "training_union_span_seconds": max(x[1] for x in train_intervals)-min(x[0] for x in train_intervals),
        "sum_training_arm_elapsed_seconds": sum(x["elapsed_seconds"] for x in training),
        "sum_RAW_label_seconds": sum(x["raw_seconds"] for x in label_rows),
        "sum_D0_GEO_label_seconds": sum(x["D0_GEO_seconds"] for x in label_rows),
        "sum_CPU_RAW_pool_seconds": sum(x["raw_score_seconds"] for x in label_rows),
        "sum_CPU_POST_pool_seconds": sum(x["post_chain_and_scores_seconds"] for x in label_rows),
        "gallery_seconds": gallery["seconds"], "gallery_workers": gallery["workers"],
        "compute_complete_utc": compute_done["utc"],
        "preparation_snapshot_to_compute_complete_seconds": instant(compute_done)-instant(resource_initial),
        "scope": "Elapsed times include concurrent contention; summing job times is not total experiment wall time. Preparation-to-compute excludes later actual image review and reporting.",
    },
    "memory": {
        "measured_per_process_peak_mib": None,
        "process_peak_status": "NA: WSL/WDDM nvidia-smi returned used_memory [N/A]",
        "max_total_device_used_observed_mib": max_device,
        "training_allocator_peak_allocated_bytes": max(x["resources"]["max_formal_batch_allocated_bytes"] for x in training),
        "training_allocator_peak_reserved_bytes": max(x["resources"]["max_formal_batch_reserved_bytes"] for x in training),
        "RAW_allocator_peak_allocated_bytes": max(raw_alloc),
        "RAW_allocator_peak_reserved_bytes": max(raw_reserved),
        "D0_allocator_peak_allocated_bytes": max(d0_alloc),
        "D0_allocator_peak_reserved_bytes": max(d0_reserved),
        "scope": "Training peaks reset per real formal update including backward/AdamW; RAW per real rollout; D0 per single real forward with resident model. Allocator figures exclude CUDA context/external allocations. Device memory includes other concurrent workers/background processes. Estimated dispatch peaks are planning bounds, not measured process peaks.",
    },
    "all_RAW_before_D0": max(x[1] for x in raw_intervals) < min(x[0] for x in d0_intervals),
    "visual_review_completion_snapshot": {k: v for k, v in visual.items()
                                 if isinstance(v, (str, int, float, bool))},
    "visual_review_scope": "This is a snapshot only. The root agent separately collects and validates actual reviews; this execution summary does not certify review completion.",
    "collector_v2_revision": revision,
    "scientific_winner": "NOT_ASSESSED",
    "additional_model_calls": 0, "additional_scoring_calls": 0,
    "source_sha256": SOURCES,
}
OUT.joinpath("execution_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

def mib(x): return f"{x/1048576:.1f}"
text = [
    "# Bench 三臂执行与资源小结",
    "",
    "本文件只读取既有账本与收据；没有启动训练、推理或重评分，也不据此选择科学胜者。",
    "",
    "## 实际预算",
    "",
    "| 项目 | 注册预算 | 完成量 |",
    "|---|---:|---:|",
    "| 优化器更新 | 4500 | 4500（每臂1500） |",
    "| 训练逐样本前向 / 反向 | 36000 / 36000 | 36000 / 36000 |",
    "| 固定探针前向 | 1056 | 1056 |",
    "| Euler50生成 / 生成器前向 | 224 / 11200 | 224 / 11200 |",
    "| 单次官方D0前向 | 224 | 224 |",
    "| 原定义逐输出评分 | 672 | 672（RAW、BASE、FINAL各224） |",
    "| 图页 | 144 | 144；渲染与实际图审分开记录 |",
    "| 占用头前向 | 0 | 0 |",
    "",
    f"总神经网络前向为 **{summary['budget']['total_neural_forwards']:,}**；另外有36,000次反向。三臂训练调用均无失败、未决调用或恢复重算。每臂12,000次样本曝光；训练汇总已核验相同冻结数据流。A/B共享COMMON零步的96次探针，B没有重复执行；A/B/C实际probe分别384/288/384。三臂均为fresh AdamW，C没有COMMON训练历史。",
    "",
    "| 臂 | 本地步数 | 累计身份 | 实际臂耗时（秒） | 更新/秒 |",
    "|---|---:|---:|---:|---:|",
]
for row in training:
    text.append(f"| {row['arm']} | {row['local_updates']} | {row['cumulative_updates']} | {row['elapsed_seconds']:.2f} | {row['updates_per_second_including_probe_and_checkpoint']:.3f} |")
text += [
    "",
    "## 并行与资源",
    "",
    f"初始实测RTX 5080，整卡16,303 MiB，已用2,511 MiB、可用13,469 MiB，GPU利用率0%；WSL可用24核、RAM {resource_initial['RAM_available']/2**30:.2f} GiB。每轮派发前均保存了GPU/CPU/RAM快照。起始保留2,048 MiB额外余量，新增进程按1.2倍保守显存估计核算。",
    "",
    f"A先启动，B在实测后加入；C等待显存余量后接替完成的训练臂。训练最多{summary['concurrency']['max_training_model_call_span_overlap']}臂重叠；包括采样在内，神经作业调用时段最多{summary['concurrency']['max_all_neural_job_call_span_overlap']}个重叠。各采样模型拥有独立输出目录；不能把这里的并发耗时当作隔离速度基准。",
    "",
    f"CPU评分先与GPU任务重叠，RAW评分使用4或6个worker。训练结束后以{len(extensions)}次资源复测扩展派发独立POST标签：先2个池，再短时3个池，随后回到2个池；每池6 worker，最大总上限{summary['concurrency']['CPU_extension_total_worker_cap']}。派发到完成收据时段也核出最大{summary['concurrency']['max_CPU_pool_overlap_dispatch_to_receipt_approx']}池重叠。未停止原控制器或正在运行的作业。图库使用6 worker。扩展只改变调度，不改变输入、评分器、样本、随机流或神经预算。",
    "",
    "## 显存统计范围",
    "",
    f"**单进程实测显存峰值为NA**：WSL/WDDM的`nvidia-smi`对进程显存返回`[N/A]`。不以模型文件大小或分配器峰值冒充进程显存。所有观测中的整卡使用峰值为{max_device:.0f} MiB，包含同时运行的作业及背景进程。",
    "",
    "| 阶段 | allocator allocated峰值(MiB) | allocator reserved峰值(MiB) | 作用域 |",
    "|---|---:|---:|---|",
    f"| 训练 | {mib(summary['memory']['training_allocator_peak_allocated_bytes'])} | {mib(summary['memory']['training_allocator_peak_reserved_bytes'])} | 每次真实更新前重置，含前向/反向/AdamW；非进程峰值 |",
    f"| RAW生成 | {mib(max(raw_alloc))} | {mib(max(raw_reserved))} | 每次真实Euler50 rollout；非进程峰值 |",
    f"| D0 | {mib(max(d0_alloc))} | {mib(max(d0_reserved))} | 单次真实D0前向，含常驻模型；非进程峰值 |",
    "",
    "派发估计训练4,500 MiB、RAW2,800 MiB、D0约3,500 MiB属于保守容量规划值，不是本轮测得的进程峰值；逐作业实际设置保留在JSON中。",
    "",
    "## 阶段耗时",
    "",
    "| 标签 | 输出数 | RAW秒 | D0+GEO秒 | CPU RAW评分秒 | CPU POST链/评分秒 |",
    "|---|---:|---:|---:|---:|---:|",
]
for row in label_rows:
    text.append(f"| {row['label']} | {row['outputs']} | {row['raw_seconds']:.2f} | {row['D0_GEO_seconds']:.2f} | {row['raw_score_seconds']:.2f} | {row['post_chain_and_scores_seconds']:.2f} |")
tim = summary["timing"]
text += [
    "",
    f"三臂各自耗时合计{tim['sum_training_arm_elapsed_seconds']:.2f}秒；首个到最后一个训练调用横跨{tim['training_union_span_seconds']:.2f}秒。RAW标签时间合计{tim['sum_RAW_label_seconds']:.2f}秒，D0+GEO合计{tim['sum_D0_GEO_label_seconds']:.2f}秒；这些有重叠，不能相加作为总墙钟。图库渲染{gallery['seconds']:.2f}秒。",
    "",
    f"从准备资源快照（{resource_initial['utc']}）到计算完成（{compute_done['utc']}）为{tim['preparation_snapshot_to_compute_complete_seconds']/60:.2f}分钟；该范围含准备、训练、评分与渲染，不含随后实际图审与报告。已验证所有224份RAW完成后才开始D0。CPU作业跨度以派发到完成收据落盘估算，不等同精确进程退出时刻。",
    "",
    "## Collector V2 工程修正",
    "",
    "原汇总器把原评分器的全网格`boundary_edges`覆盖为输入C边数。原672条评分JSON、冻结源码及旧`results/`保留不改；新增`collect_v2.py`生成独立`results_v2/`，保持原`boundary_edges`，用新元数据列`C_boundary_edges`存C边数，并纳入原边界字段的聚合与配对。该修正没有新增模型调用、几何修改或重评分。正式数值汇总应引用V2，详见`collector_v2_revision.json`。",
    "",
    "所有数值来源、逐标签时段、派发快照摘要和调用计数保存在相邻`execution_summary.json`；输入小文件hash随附。原件只读，本文件没有重算几何指标。",
]
OUT.joinpath("execution_summary.md").write_text("\n".join(text)+"\n", encoding="utf-8")
print(json.dumps({"status": summary["status"], "budget": summary["budget"],
                  "concurrency": summary["concurrency"], "timing": summary["timing"],
                  "memory": summary["memory"]}, ensure_ascii=False, indent=2))
