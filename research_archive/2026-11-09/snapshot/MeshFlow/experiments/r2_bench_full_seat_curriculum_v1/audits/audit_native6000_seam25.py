"""One-shot stdlib audit of saved metadata/logs; no tensor/weight/GPU reads."""
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
HASH_KEYS = ("generator_state_sha256", "optimizer_state_sha256", "rng_sha256", "head_state_sha256")


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def sha(p):
    p = Path(p)
    if p.suffix in (".pt", ".pth", ".npz"):
        raise ValueError("This audit never hashes tensor/weight assets")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def records(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            # Complete prefixes only. A concurrent suffix is irrelevant to the
            # SEAM first-25 audit, and the reader stops at the first later step.
            if line.strip():
                yield json.loads(line)


def cursor(row, stage, step):
    offset = 0 if stage == "NATIVE" else 6000
    require(row["stage"] == stage and row["stage_step"] == row["local_step"] == step,
            "Stage/local cursor differs")
    require(row["curriculum_step"] == offset + step and row["global_step"] == 12000 + offset + step,
            "Global/curriculum cursor differs")
    require(row["native_total_step"] == (step if stage == "NATIVE" else 6000)
            and row["seam_total_step"] == (step if stage == "SEAM" else 0), "Cumulative step differs")


def audit_stage(stage, count, schedule):
    folder = OUT / "training" / stage
    prefix = 0 if stage == "NATIVE" else 6000
    expected = dict(train_forward=8 * count, train_backward=8 * count, optimizer_update=count)
    starts, dones = collections.Counter(), collections.Counter()
    pending, attempts = {}, set()
    failure_count = 0
    for r in records(folder / "calls.jsonl"):
        ctx = r["context"]
        if ctx["stage_step"] > count:
            require(stage == "SEAM", "Native exceeded its authorized endpoint")
            break
        cursor(ctx, stage, ctx["stage_step"])
        key = (r["attempt_id"], r["call_id"])
        attempts.add(r["attempt_id"])
        kind = r["kind"]
        require(kind in expected, "Unbudgeted head/probe/unknown training call")
        if r["event"] == "CALL_BEGIN":
            require(key not in pending, "Duplicate pending call")
            position = starts[kind]
            step = position // 8 + 1 if kind != "optimizer_update" else position + 1
            mi = position % 8 if kind != "optimizer_update" else None
            require((ctx["stage_step"], ctx.get("microbatch")) == (step, mi), "Call sequence differs")
            if mi is not None:
                require(ctx["condition_id"] == schedule[prefix + step - 1]["samples"][mi]["condition_id"],
                        "Call condition differs from frozen stream")
            starts[kind] += 1
            pending[key] = (kind, ctx)
        elif r["event"] == "CALL_DONE":
            require(pending.pop(key, None) == (kind, ctx), "Call completion lacks exact matching start")
            require(math.isfinite(r["seconds"]) and r["seconds"] >= 0, "Call time invalid")
            dones[kind] += 1
        else:
            failure_count += 1
    require(dict(starts) == expected and dict(dones) == expected and not pending and failure_count == 0,
            "Call counts/failures/pending differ")
    require(len(attempts) == 1, "Unexpected training attempt/restart")
    logs = []
    for r in records(folder / "training_log.jsonl"):
        if r["stage_step"] > count:
            require(stage == "SEAM", "Native log exceeds budget")
            break
        cursor(r, stage, len(logs) + 1)
        for k in ("FM", "objective", "generator_gradient_norm", "all_free_velocity_sse", "all_free_endpoint_sse"):
            require(math.isfinite(r[k]), "Nonfinite training diagnostic")
        require(r["generator_gradient_norm"] > 0 and r["head_calls"] == r["head_step"] == 0,
                "Gradient/head activity invalid")
        require(r["free_coordinate_denominator"] == r["all_free_scalar_count"] > 0, "Batch scalar denominator differs")
        require(math.isclose(r["FM"], r["all_free_velocity_sse"] / r["all_free_scalar_count"], rel_tol=4e-6, abs_tol=1e-7),
                "Batch FM != original SSE/free-scalar count")
        require(r["objective"] == r["FM"], "Additional objective detected")
        logs.append(r)
    require(len(logs) == count, "Training log cursor incomplete")
    parent_counts, condition_counts, aug_counts = collections.Counter(), collections.Counter(), collections.Counter()
    batch = collections.defaultdict(collections.Counter)
    owners, active, exposures = 0, 0, 0
    role_keys = [f"{part}_{stat}" for part in ("all_free", "seam_tip", "bulk")
                 for stat in ("scalar_count", "velocity_sse", "endpoint_sse")]
    for r in records(folder / "exposure.jsonl"):
        if r["stage_step"] > count:
            require(stage == "SEAM", "Native exposure exceeds budget")
            break
        step, mi = exposures // 8 + 1, exposures % 8
        cursor(r, stage, step)
        require(r["microbatch"] == r["sample_index"] == mi, "Exposure microbatch order differs")
        frozen = schedule[prefix + step - 1]["samples"][mi]
        require(all(r.get(k) == v for k, v in frozen.items()), "Exposure differs from frozen schedule")
        require(r["head_forward_called"] is False, "Training called occupancy head")
        m = r["mechanism"]
        n, k, source = r["N"], r["K"], m["source_owner_count"]
        wanted = 0 if stage == "NATIVE" else source
        require(m["active_seam_slots"] == wanted, "Active layout is not the required owner policy")
        require(m["policy"] == ("NATIVE_ALL_FREE_OT" if stage == "NATIVE" else "SOURCE_OWNER_SEAM_OT"),
                "Representation/OT policy differs")
        require(m["free_coordinate_count"] == r["all_free_scalar_count"] == 9 * (n - k) - 6 * wanted,
                "Sample free scalar denominator differs")
        require(r["seam_tip_scalar_count"] == 3 * wanted and r["bulk_scalar_count"] == 9 * (n - k - wanted),
                "Tip/bulk free scalar roles differ")
        require(r["all_free_scalar_count"] == r["seam_tip_scalar_count"] + r["bulk_scalar_count"], "Roles do not partition free count")
        for suffix in ("velocity_sse", "endpoint_sse"):
            require(math.isclose(r["all_free_" + suffix], r["seam_tip_" + suffix] + r["bulk_" + suffix], rel_tol=1e-10, abs_tol=1e-8),
                    "Role SSE does not sum to full free SSE")
        require(math.isclose(r["loss_contribution"], r["all_free_velocity_sse"] / logs[step - 1]["free_coordinate_denominator"],
                             rel_tol=4e-6, abs_tol=1e-7), "Exposure contribution uses another denominator")
        require(0 <= r["t"] <= 1 and all(math.isfinite(r[x]) for x in role_keys), "Nonfinite exposure scalar")
        for key in role_keys:
            batch[step][key] += r[key]
        batch[step]["loss_contribution"] += r["loss_contribution"]
        parent_counts[r["parent_id"]] += 1
        condition_counts[r["condition_id"]] += 1
        aug_counts[r["augmentation"]] += 1
        owners += source; active += wanted; exposures += 1
    require(exposures == 8 * count, "Exposure count differs")
    require(aug_counts == {"CLEAN": 4 * count, "X_WIDTH": 4 * count}, "Identity/width exposure balance differs")
    for step, sums in batch.items():
        for key in role_keys:
            require(math.isclose(sums[key], logs[step - 1][key], rel_tol=1e-10, abs_tol=1e-7), "Exposure/log sums differ")
        require(math.isclose(sums["loss_contribution"], logs[step - 1]["FM"], rel_tol=1e-10, abs_tol=1e-9), "FM contributions differ")
    def window(a, b):
        rr = logs[a - 1:b]
        den = sum(r["all_free_scalar_count"] for r in rr)
        return dict(first_step=a, last_step=b, velocity_sse=sum(r["all_free_velocity_sse"] for r in rr),
                    scalar_count=den, velocity_mse=sum(r["all_free_velocity_sse"] for r in rr) / den)
    return dict(stage=stage, audited_updates=count, global_start=12000 + prefix, global_end=12000 + prefix + count,
                calls=dict(attempted=dict(starts), successful=dict(dones), failures=failure_count, pending=0,
                           attempt_ids=sorted(attempts), head=0, probe=0),
                exposure=dict(total=exposures, parents=len(parent_counts), conditions=len(condition_counts),
                              augmentation=dict(aug_counts), parent_counts=dict(sorted(parent_counts.items())),
                              condition_counts=dict(sorted(condition_counts.items())), source_owner_slots=owners,
                              active_seam_slots=active, frozen_stream_match=True),
                roles={k: sum(r[k] for r in logs) for k in role_keys},
                loss_diagnostics=[window(1, min(100, count)), window(max(1, count - 99), count)],
                gradient_norm_min=min(r["generator_gradient_norm"] for r in logs),
                gradient_norm_max=max(r["generator_gradient_norm"] for r in logs),
                input_seconds=sum(r["input_seconds"] for r in logs), update_seconds=sum(r["update_seconds"] for r in logs),
                peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in logs),
                peak_reserved_bytes=max(r["peak_reserved_bytes"] for r in logs),
                peak_scope=logs[0]["peak_scope"], no_nonfinite_scalars=True)


def run():
    native = read(OUT / "training/NATIVE/status.json")
    workflow = read(OUT / "workflow/native/status.json")
    seam = read(OUT / "training/SEAM/status.json")
    require(native["status"] == workflow["status"] == "COMPLETE" and native["stage_step"] == 6000,
            "Native finalization/workflow not complete")
    require(seam["status"] in ("TRAINING", "COMPLETE") and seam["stage_step"] >= 25, "SEAM first 25 not complete")
    reg = read(OUT / "registration.json")
    require(reg["stage_updates"] == {"NATIVE": 6000, "SEAM": 3000} and reg["final_global_step"] == 21000,
            "Registered curriculum budget differs")
    require((reg["generator_training_forwards"], reg["generator_backwards"], reg["optimizer_updates"],
             reg["training_head_forwards"], reg["training_head_updates"], reg["probe_forwards"])
            == (72000, 72000, 9000, 0, 0, 0), "Registered call budgets differ")
    regsha = sha(OUT / "registration.json")
    require(native["registration_sha256"] == seam["registration_sha256"] == regsha, "Stage registration identity differs")
    schedule_path = OUT / "data/schedule.json"
    require(sha(schedule_path) == reg["input_schedule_sha256"], "Frozen stream changed")
    stream = read(schedule_path)
    require(len(stream["steps"]) == 9000 and stream["native_updates"] == 6000 and stream["seam_updates"] == 3000,
            "Frozen schedule budget differs")
    np = OUT / "training/NATIVE/checkpoint_step6000.json"
    sp = OUT / "training/SEAM/checkpoint_step0.json"
    nr, sr = read(np), read(sp)
    require(native["checkpoint"] == nr and nr["stage_step"] == 6000 and sr["stage_step"] == 0,
            "Endpoint/SEAM0 receipt identity differs")
    require(nr["global_step"] == sr["global_step"] == 18000 and nr["curriculum_step"] == sr["curriculum_step"] == 6000,
            "Transition global/stream cursor differs")
    require(all(nr[k] == sr[k] for k in HASH_KEYS), "Transition generator/AdamW/RNG/head identity differs")
    require(nr["head_state_sha256"] == reg["head_state_sha256"], "Frozen head changed")
    nadir = OUT / "training/NATIVE/attempts" / native["attempt"]
    sadir = OUT / "training/SEAM/attempts" / seam["attempt"]
    require(read(nadir / "final_status.json") == native, "Native finalization receipt/status differ")
    init = read(sadir / "initialization.json")
    transition = init["stage_transition"]
    require(init["inherited_optimizer"] is True and init["inherited_rng"] is True and init["fresh_AdamW"] is False,
            "Original COMMON initialization policy differs")
    require(transition["optimizer_reset"] is False and transition["head_frozen"] is True
            and transition["source_global_step"] == 18000, "Stage inheritance/reset policy differs")
    require(transition["source_sha256"] == nr["sha256"] and transition["same_manifest_sha256"] == reg["manifest_sha256"],
            "Transition source file identity/data pool differs")
    require(all(transition[k] == nr[k] for k in HASH_KEYS[:3]), "Initialization receipt states differ")
    export = read(OUT / "training/NATIVE/generator_step6000.json")
    require(native["inference_export"] == export and export["global_step"] == 18000
            and export["generator_state_sha256"] == nr["generator_state_sha256"]
            and export["head_state_sha256"] == nr["head_state_sha256"], "Native export receipt differs")
    na = audit_stage("NATIVE", 6000, stream["steps"])
    sa = audit_stage("SEAM", 25, stream["steps"])
    for key in ("attempted", "successful"):
        require(all(native["calls"][key].get(k, 0) == v for k, v in na["calls"][key].items()), "Native final call snapshot differs")
    require(not native["calls"]["pending_call_ids"] and not native["calls"]["failed"], "Native final ledger has pending/failure")
    files = [OUT / "registration.json", np, sp, nadir / "final_status.json", sadir / "initialization.json",
             OUT / "training/NATIVE/generator_step6000.json", Path(__file__)]
    result = dict(schema="BENCH_FULL_SEAT_NATIVE6000_SEAM25_TRANSITION_AUDIT_V1", status="PASS",
                  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  evidence_hashes={str(p.relative_to(ROOT)): sha(p) for p in files},
                  transition=dict(native_endpoint=nr, seam_step0=sr,
                      all_four_state_hashes_equal=True, matching_original_AdamW_continued=True,
                      AdamW_cursor_at_transition=18000, head_frozen_zero_training=True,
                      same_pool=True, initialization=init, native_export=export,
                      weight_tensors_loaded_by_this_audit=False, weight_file_hash_recomputed_by_this_audit=False,
                      evidence_boundary="State identity hashes are producer-recorded; runtime restoration and per-update AdamW step validation checked in trainer source, not independently tensor-loaded here"),
                  budget=dict(source_global=12000, native_updates=6000, seam_updates=3000, final_global=21000,
                              total_updates=9000, total_exposures=72000, all_training_head_calls=0, all_probe_calls=0),
                  native_complete=na, seam_first25=sa,
                  native_final_elapsed_seconds=native["elapsed_seconds"], native_observed_device_resources=native["resources"],
                  observed_seam_status=dict(status=seam["status"], stage_step=seam["stage_step"], global_step=seam["global_step"]),
                  no_new_model_calls=True, no_scoring=True, no_intervention=True,
                  interpretation="Training diagnostics and exact accounting only; no generation/visual quality conclusion")
    dest = OUT / "audits/native6000_to_seam25.json"
    with dest.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    text = ("# Native 6000 → SEAM 首 25 步交接审计\n\n"
            "状态：PASS。Native workflow/训练/attempt final_status 均完成；SEAM 已实际完成首 25 步。\n\n"
            "Native step6000 与 SEAM step0 的生成器、AdamW、RNG、冻结头四项 state hash 完全相同，"
            "均为 global18000/curriculum6000。注册终点仍为 global21000；新增预算固定6000+3000，"
            "无优化器重置、无新头训练。证据来自运行时旁车与初始化记录，本审计未读张量或重 hash 大权重。\n\n"
            "Native 完整账本：48000 前向、48000 反向、6000 更新、48000 实例；"
            "SEAM 首25步：200 前向、200 反向、25 更新、200 实例。均只有一个 attempt，"
            "已完成前缀失败/挂起/头/探针调用为0，逐实例与冻结日程完全匹配。\n\n"
            "Native 活跃 SEAM 槽为0、分母9(N−K)；SEAM 活跃槽等于来源 owner，"
            "tip=3S、bulk=9(N−K−S)、总自由标量9(N−K)−6S。逐例 SSE/分母/贡献和"
            "整批日志一致，所有有限数值与正梯度检查通过。各父真实曝光、阶段耗时、"
            "分配/保留显存峰与 loss 诊断详见配套 JSON。显存峰是 PyTorch allocator，非进程峰；"
            "并发环境下耗时不当作独占速度比较。\n\n"
            "边界：这是训练交接及账本审计，不是整体课程完成、生成质量、语义成功或视觉审阅结论。"
            "最终仍须独立核验 SEAM3000、controller READY_REVIEW、完整科学账本和实际逐图审阅。\n")
    with (OUT / "audits/native6000_to_seam25.md").open("x", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(json.dumps({k: result[k] for k in ("schema", "status", "created_utc", "observed_seam_status")}, ensure_ascii=False))


if __name__ == "__main__":
    run()
