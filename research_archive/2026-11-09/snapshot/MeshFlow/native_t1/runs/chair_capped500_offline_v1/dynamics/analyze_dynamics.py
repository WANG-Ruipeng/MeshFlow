"""Read-only CPU analysis of saved training logs; never imports model/torch/scoring code."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics as st
import time

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[1] / "chair_control_feedback_auto_v1"
ARMS = ("FM_CONTROL", "REPLAY_FM_CONTROL", "TRAJECTORY_REPLAY", "TRAJECTORY_REPLAY_CAPPED")
TEACHER, TR, CAP = ARMS[1:]
WINDOWS = {"first500": (1, 500), "second500": (501, 1000),
           "early96": (1, 96), "pre500_96": (405, 500),
           "post500_96": (501, 596), "late96": (905, 1000)}
START_SHA = "d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def avg(values):
    v = [float(x) for x in values if x is not None]
    return st.mean(v) if v else None


def median(values):
    v = [float(x) for x in values if x is not None]
    return st.median(v) if v else None


def summary(rows):
    hooks = [r for r in rows if r["aux_grad_norm"] is not None]
    states = defaultdict(list)
    for r in rows:
        if r["target_id"] is not None:
            states[r["target_id"]].append(r)
    result = dict(n=len(rows), FM_mean=avg(r["FM"] for r in rows),
        raw_aux_mean=avg(r["raw_aux"] for r in rows),
        raw_aux_equal_target_mean=avg(avg(r["raw_aux"] for r in s) for s in states.values()),
        weighted_aux_mean=avg(r["weighted_aux"] for r in rows),
        coefficient_mean=avg(r["coefficient"] for r in rows),
        coefficient_median=median(r["coefficient"] for r in rows),
        cap_count=sum(r["cap_active"] is True for r in rows),
        cap_fraction=avg(r["cap_active"] for r in rows),
        clip_fraction=avg(r["clip_active"] for r in rows),
        preclip_norm_mean=avg(r["preclip_norm"] for r in rows),
        preclip_norm_median=median(r["preclip_norm"] for r in rows),
        FM_norm_median=median(r["FM_grad_norm"] for r in rows),
        paired_FM_minus_baseline_mean=avg(r["FM_minus_baseline"] for r in rows),
        paired_FM_minus_teacher_mean=avg(r["FM_minus_teacher"] for r in rows),
        baseline_FM_mean=avg(r["baseline_FM"] for r in rows),
        teacher_FM_mean=avg(r["teacher_FM"] for r in rows),
        hook_n=len(hooks), hook_unique_targets=len({r["target_id"] for r in hooks}),
        observed_aux_norm_mean=avg(r["aux_grad_norm"] for r in hooks),
        observed_aux_norm_median=median(r["aux_grad_norm"] for r in hooks),
        observed_aux_FM_ratio_mean=avg(r["aux_FM_grad_ratio"] for r in hooks),
        observed_aux_FM_ratio_median=median(r["aux_FM_grad_ratio"] for r in hooks),
        observed_cosine_mean=avg(r["cosine"] for r in hooks),
        observed_cosine_median=median(r["cosine"] for r in hooks),
        observed_negative_cosines=sum(r["cosine"] < 0 for r in hooks),
        observed_negative_cosine_fraction=avg(r["cosine"] < 0 for r in hooks),
        selected_target_count=len(states),
        selected_parent_count=len({r["uid"] for r in rows if r["uid"]}),
        input_update_seconds_median=median(r["input_update_seconds"] for r in rows))
    result["weighted_aux_over_FM_ratio_of_means"] = result["weighted_aux_mean"] / result["FM_mean"]
    result["FM_excess_vs_same_step_baseline_percent"] = 100 * result["paired_FM_minus_baseline_mean"] / result["baseline_FM_mean"]
    result["FM_excess_vs_same_step_teacher_percent"] = 100 * result["paired_FM_minus_teacher_mean"] / result["teacher_FM_mean"]
    return result


def group_rows(rows, names):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[n] for n in names)].append(r)
    return groups


def run(old=OLD, out=HERE):
    old, out = Path(old).resolve(), Path(out).resolve()
    if old == out or old in out.parents:
        raise ValueError("New analysis outputs must be outside the protected original run.")
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "completion.json"
    if marker.exists():
        prior = json.loads(marker.read_text(encoding="utf-8"))
        if prior["script_sha256"] == digest(__file__) and all(Path(p).is_file() and digest(p) == h for p, h in prior["source_sha256"].items()) and all((out/n).is_file() and digest(out/n) == h for n, h in prior["output_sha256"].items()):
            return {"status": "CACHE_PASS", "output": str(out), "model_calls": 0}
        raise RuntimeError("Existing completion identity differs; preserve it instead of silently replacing.")
    sources, checks = {}, Counter()

    def load(path, lines=False):
        path = path.resolve()
        data = path.read_bytes()
        sources[str(path)] = hashlib.sha256(data).hexdigest()
        text = data.decode("utf-8-sig")
        return [json.loads(s) for s in text.splitlines() if s.strip()] if lines else json.loads(text)

    def require(value, name):
        checks[name] += 1
        if not value:
            raise AssertionError(name)

    pool = load(old / "feedback/pool_v1/manifest.json")
    pool_hash = sources[str((old / "feedback/pool_v1/manifest.json").resolve())]
    raw = {a: load(old / f"training/explore_{a}_metrics.jsonl", lines=True) for a in ARMS}
    recipe = load(old / "decisions/H3_recipe.json")
    for arm in ARMS:
        done = load(old / f"training/explore_{arm}_complete.json")
        require(done["steps"] == 1000 and done["status"] == "PASS", "completion_1000")
        require([r["pilot_step"] for r in raw[arm]] == list(range(1, 1001)), "exact_step_sequence")
    for i in range(1000):
        for a, b in itertools.combinations(ARMS, 2):
            x, y = raw[a][i], raw[b][i]
            require(x["shared_batch_sha256"] == y["shared_batch_sha256"] and x["coupled_batch_sha256"] == y["coupled_batch_sha256"], "same_step_primary_identity")
            require(x["free_coordinate_denominator"] == y["free_coordinate_denominator"], "same_primary_denominator")
            require([q["shared_input_sha256"] for q in x["sample_diagnostics"]] == [q["shared_input_sha256"] for q in y["sample_diagnostics"]], "all_eight_primary_samples")
    flat, primary, identities = [], [], {}
    for arm in ARMS:
        for r in raw[arm]:
            step = r["pilot_step"]
            sd = r["sample_diagnostics"]
            require(len(sd) == 8 and len({s["uid"] for s in sd}) == 8, "primary_8_distinct_parents")
            require(sum(s["alpha"] == 1 for s in sd) == 4 and all(.9 <= s["alpha"] <= 1.1 for s in sd), "primary_4unit4uniform")
            require(sum(s["free_coordinates"] for s in sd) == r["free_coordinate_denominator"], "denominator_sum")
            require(sum(s["FM_contribution"] for s in sd).hex() == r["FM"].hex(), "actual_Python_FM_sum")
            audit = r.get("aux_audit")
            identity = None
            if audit:
                require(audit["pool_manifest_sha256"] == pool_hash, "fixed_pool_hash")
                require(audit["queue"] == ("S", "I", "E")[(step-1) % 3], "queue_cycle")
                require(audit["uid"] == pool["sorted_parents"][((step-1)//3) % 8], "parent_cycle")
                require(audit["state_index"] == (30 if ((step-1)//24) % 2 == 0 else 40), "time_cycle")
                require(audit["request_id"] == audit["selection"]["selected_request_id"], "selected_request_identity")
                for field in ("request_id", "uid", "candidate", "queue", "state_index", "C_sha256"):
                    require(audit[field] == raw[TR][step-1]["aux_audit"][field], "same_aux_selection_all_arms")
                vals = tuple(audit[k] for k in ("request_id", "state_index", "xt_sha256", "u_sha256", "target_sha256"))
                identity = hashlib.sha256(json.dumps(vals, separators=(",", ":")).encode()).hexdigest()
                key = (arm, audit["request_id"], audit["state_index"])
                require(key not in identities or identities[key] == identity, "fixed_target_across_training")
                identities[key] = identity
                if arm == CAP:
                    other = raw[TR][step-1]["aux_audit"]
                    require(all(audit[k] == other[k] for k in ("xt_sha256", "u_sha256", "target_sha256")), "CAP_TR_same_real_state_and_GT_target")
            balance = r.get("loss_balance")
            coefficient = balance["effective_weight"] if balance else .1 if audit else None
            if balance:
                expected = min(.1, r["FM"] / max(r["aux_unweighted"], 1e-30))
                require(coefficient.hex() == expected.hex(), "logged_detached_coefficient")
                require(balance["cap_active"] == (expected < .1), "logged_cap_activation")
            if audit:
                expected_loss = float(np.float32(np.float32(coefficient) * np.float32(r["aux_unweighted"])))
                require(expected_loss.hex() == r["aux_weighted"].hex(), "actual_FP32_weighted_loss")
            aux_norm = r.get("weighted_aux_gradient_norm")
            require((aux_norm is not None) == (audit is not None and (step == 1 or step % 25 == 0)), "actual_hook_schedule")
            row = dict(arm=arm, step=step, half="first500" if step <= 500 else "second500", FM=r["FM"],
                raw_aux=r["aux_unweighted"], weighted_aux=r["aux_weighted"], coefficient=coefficient,
                cap_active=balance["cap_active"] if balance else None, clip_active=r["grad_norm_before_clip"] > 1,
                preclip_norm=r["grad_norm_before_clip"], estimated_clip_coefficient=min(1., 1./(r["grad_norm_before_clip"]+1e-6)),
                FM_grad_norm=r["FM_gradient_norm"], aux_grad_norm=aux_norm,
                aux_FM_grad_ratio=aux_norm/r["FM_gradient_norm"] if aux_norm is not None and r["FM_gradient_norm"] else None,
                cosine=r.get("FM_aux_gradient_cosine"), target_id=identity,
                uid=audit["uid"] if audit else None, request_id=audit["request_id"] if audit else None,
                noise=audit["candidate"] if audit else None, queue=audit["queue"] if audit else None,
                t=round(audit["t"], 1) if audit else None, cycle_position=(step-1) % 48,
                baseline_FM=raw[ARMS[0]][step-1]["FM"], teacher_FM=raw[TEACHER][step-1]["FM"],
                FM_minus_baseline=r["FM"]-raw[ARMS[0]][step-1]["FM"], FM_minus_teacher=r["FM"]-raw[TEACHER][step-1]["FM"],
                input_update_seconds=r["OT_and_input_seconds"]+r["update_total_seconds"],
                parameter_update_norm=None, parameter_update_status="NOT_RECORDED_NOT_RECONSTRUCTED", shared_batch_sha256=r["shared_batch_sha256"])
            if audit:
                row.update({k: audit[k] for k in ("xt_sha256", "u_sha256", "target_sha256", "C_sha256")})
            flat.append(row)
            for s in sd:
                primary.append(dict(arm=arm, step=step, half=row["half"], uid=s["uid"], t_bin=min(4, int(s["t"]*5)),
                    free_coordinates=s["free_coordinates"], squared_error_sum_from_logged_contribution=s["FM_contribution"]*r["free_coordinate_denominator"]))
    write_csv(out / "per_step.csv", flat)
    hooks = [r for r in flat if r["aux_grad_norm"] is not None]
    write_csv(out / "actual_hooks.csv", hooks)
    phase_rows, strata, target_rows, gradient_pairs, gradient_summary = [], [], [], [], []
    dims = ((), ("t",), ("queue",), ("uid",), ("uid", "t"), ("queue", "t"), ("uid", "queue", "t"), ("target_id",))
    for arm in ARMS:
        ar = [r for r in flat if r["arm"] == arm]
        windows = {w: [r for r in ar if lo <= r["step"] <= hi] for w, (lo, hi) in WINDOWS.items()}
        for name, rows in windows.items():
            phase_rows.append(dict(arm=arm, window=name, first_step=WINDOWS[name][0], last_step=WINDOWS[name][1], **summary(rows)))
            if arm == ARMS[0]:
                continue
            if name.endswith("96"):
                require(Counter(r["cycle_position"] for r in rows) == Counter({i: 2 for i in range(48)}), "balanced_96step_cycle_positions")
            require(len({r["target_id"] for r in rows}) == 24, "all_24_targets_in_each_window")
            for dim in dims:
                for key, values in group_rows(rows, dim).items():
                    strata.append(dict(arm=arm, window=name, dimension="+".join(dim) or "all", group=json.dumps(key), **summary(values)))
        if arm == ARMS[0]:
            continue
        targets = sorted({r["target_id"] for r in ar})
        require(len(targets) == 24, "exact_24_aux_targets")
        for target in targets:
            alltarget = [r for r in ar if r["target_id"] == target]
            source = alltarget[0]
            result = {k: source[k] for k in ("arm", "target_id", "request_id", "uid", "noise", "t", "xt_sha256", "u_sha256", "target_sha256")}
            result["queues"] = "+".join(sorted({r["queue"] for r in alltarget}))
            for name, wr in windows.items():
                selected = [r for r in wr if r["target_id"] == target]
                ss = summary(selected)
                for key in ("n", "raw_aux_mean", "coefficient_mean", "cap_fraction", "clip_fraction", "hook_n", "observed_aux_FM_ratio_mean", "observed_cosine_mean"):
                    result[name+"_"+key] = ss[key]
            for a, b in (("early96", "pre500_96"), ("pre500_96", "late96"), ("first500", "second500")):
                first, last = result[a+"_raw_aux_mean"], result[b+"_raw_aux_mean"]
                result[a+"_to_"+b+"_raw_aux_fractional_change"] = last/first-1
            target_rows.append(result)
            gs = dict(arm=arm, target_id=target, uid=source["uid"], t=source["t"], request_id=source["request_id"])
            for half in ("first500", "second500"):
                hh = [r for r in windows[half] if r["target_id"] == target and r["aux_grad_norm"] is not None]
                gs[half+"_hook_n"] = len(hh)
                for k in ("aux_grad_norm", "aux_FM_grad_ratio", "cosine"):
                    gs[half+"_"+k+"_mean"] = avg(r[k] for r in hh)
            gs["both_halves_observed"] = gs["first500_hook_n"] > 0 and gs["second500_hook_n"] > 0
            gradient_pairs.append(gs)
        matched = [r for r in gradient_pairs if r["arm"] == arm and r["both_halves_observed"]]
        gsummary = dict(arm=arm, matched_targets=len(matched), all_targets=24, scope="Same actual target identity in both halves, but only saved hooks; sparse observations and unequal within-target steps.")
        for half in ("first500", "second500"):
            for k in ("aux_grad_norm", "aux_FM_grad_ratio", "cosine"):
                gsummary[half+"_equal_target_"+k] = avg(r[half+"_"+k+"_mean"] for r in matched)
        gradient_summary.append(gsummary)
    write_csv(out / "phase_summary.csv", phase_rows)
    write_csv(out / "auxiliary_strata.csv", strata)
    write_csv(out / "aligned_24_targets.csv", target_rows)
    write_csv(out / "matched_gradient_targets.csv", gradient_pairs)
    write_csv(out / "matched_gradient_summary.csv", gradient_summary)
    primary_summary = []
    for dim in (("arm", "half", "t_bin"), ("arm", "half", "uid")):
        for key, rows in group_rows(primary, dim).items():
            denom = sum(r["free_coordinates"] for r in rows)
            primary_summary.append(dict(dimension="+".join(dim), **dict(zip(dim, key)), samples=len(rows), free_coordinates=denom,
                coordinate_weighted_MSE_from_logged_contributions=sum(r["squared_error_sum_from_logged_contribution"] for r in rows)/denom))
    write_csv(out / "primary_time_parent_strata.csv", primary_summary)
    manifest = load(old / "derived/checkpoints/checkpoint_manifest.json")
    checkpoint_rows = []
    for entry in manifest["checkpoints"]:
        path = Path(entry["path"])
        side = load(Path(entry["sidecar_path"]))
        require(side["checkpoint_sha256"] == entry["actual_checkpoint_sha256"], "checkpoint_sidecar_existing_full_hash")
        require(side["metadata"]["source_checkpoint_sha256"] == START_SHA, "checkpoint_START_identity")
        exists = path.is_file()
        require(exists and path.stat().st_size == entry["bytes"], "checkpoint_exists_same_size")
        old_stat = entry.get("file_stat", {})
        require(path.stat().st_mtime_ns == old_stat["mtime_ns"], "checkpoint_mtime_since_full_hash")
        step = side["metadata"]["feedback_step"]
        checkpoint_rows.append(dict(arm=entry["arm"], feedback_step=step, conditional_total_step=side["metadata"]["conditional_total_step"], path=str(path),
            bytes=path.stat().st_size, existing_full_file_sha256=entry["actual_checkpoint_sha256"], newly_hashed_weight=False, newly_loaded_weight=False,
            full_recovery_state=side["metadata"]["full_recovery_state"], model_state_sha256=side["metadata"]["model_state_sha256"],
            DEV_existing_scored_outputs=16 if step in (500, 1000) else 0,
            existing_generation_scope="DEV16" if step in (500, 1000) else "NONE_SAVED_900",
            inference_required_for_new_DEV=step == 900,
            evidence="Prior complete weight hash + current size/mtime + sidecar; no current tensor loading."))
    write_csv(out / "checkpoint_availability.csv", checkpoint_rows)
    for name in ("training.py", "loss_balance.py", "replay.py", "stream.py"):
        p = old.parents[2] / "experiments/mf_h1_local0/chair_control_feedback_auto_v1" / name
        sources[str(p.resolve())] = digest(p)
    update_fields = sorted({k for rows in raw.values() for r in rows for k in r if "update" in k or "delta" in k})
    notes = dict(parameter_update_norm_status="NA_NOT_RECORDED", matching_fields=update_fields,
        reason="Only update wall time and optimizer-step count were saved, not parameter deltas or per-step AdamW moments. Gradient norm or clipping coefficient is not a parameter-update norm. Checkpoint hashes cannot measure a vector delta.")
    plot(out, flat, phase_rows)
    by_phase = {(r["arm"], r["window"]): r for r in phase_rows}
    for arm in ARMS:
        require(all(math.isfinite(r["FM"]) and math.isfinite(r["raw_aux"]) for r in flat if r["arm"] == arm), "finite_saved_losses")
    require(all(digest(p) == h for p, h in sources.items()), "all_old_read_sources_unchanged")
    result = dict(status="PASS", created_unix=time.time(), model_calls=0, new_training_updates=0, new_backward_calls=0, scoring_calls=0,
        sources_root=str(old), checks=dict(checks), windows=WINDOWS, stage_summary=phase_rows,
        matched_gradient_summary=gradient_summary, parameter_updates=notes, checkpoint_count=len(checkpoint_rows),
        source_sha256=sources, script_sha256=digest(__file__),
        scope="Saved training dynamics only; not an independent DEV efficacy test or identification of why a particular checkpoint generates a shape.")
    atomic_json(out / "summary.json", result)
    atomic_json(out / "source_hashes.json", dict(script_sha256=digest(__file__), read_only_sources=sources, unchanged_after=True))
    memo(out, by_phase, target_rows, gradient_summary, notes)
    output_names = ["per_step.csv", "actual_hooks.csv", "phase_summary.csv", "auxiliary_strata.csv", "aligned_24_targets.csv", "matched_gradient_targets.csv", "matched_gradient_summary.csv", "primary_time_parent_strata.csv", "checkpoint_availability.csv", "summary.json", "source_hashes.json", "DYNAMICS_MEMO.md", "dynamics.png"]
    atomic_json(marker, dict(status="PASS", script_sha256=digest(__file__), source_sha256=sources, output_sha256={n:digest(out/n) for n in output_names}, checks=sum(checks.values()), model_calls=0))
    return dict(status="PASS", checks=sum(checks.values()), output=str(out), model_calls=0, checkpoint_count=len(checkpoint_rows))


def plot(out, flat, phases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {TEACHER:"#287bb5", TR:"#e79535", CAP:"#9064bc"}
    def trailing(v, n=50):
        return [st.mean(v[max(0,i-n+1):i+1]) for i in range(len(v))]
    cap = [r for r in flat if r["arm"] == CAP]
    ax[0,0].plot(range(1,1001), trailing([r["coefficient"] for r in cap]), color=colors[CAP], label="Actual coefficient")
    ax[0,0].set_title("CAPPED coefficient: trailing 50-step mean")
    ax[0,0].set_ylabel("Detached auxiliary coefficient")
    for arm in (TEACHER, TR, CAP):
        rows = [r for r in flat if r["arm"] == arm]
        ax[0,1].plot(range(1,1001), trailing([r["clip_active"] for r in rows]), color=colors[arm], label=arm)
        hooks = [r for r in rows if r["aux_grad_norm"] is not None]
        ax[1,0].scatter([r["step"] for r in hooks], [r["aux_FM_grad_ratio"] for r in hooks], color=colors[arm], s=15, label=arm)
        ax[1,1].plot(range(1,1001), trailing([r["FM_minus_baseline"] for r in rows]), color=colors[arm], label=arm)
    ax[0,1].set_title("Total pre-clip norm >1: trailing fraction")
    ax[1,0].set_title("Actual weighted aux / FM gradient norm")
    ax[1,0].set_yscale("log")
    ax[1,1].set_title("Same-step primary FM minus FM_CONTROL")
    ax[1,1].axhline(0,color="gray",linewidth=.7)
    for a in ax.flat:
        a.axvline(500,color="gray",linestyle="--",linewidth=1)
        a.set_xlabel("Completed training step");a.grid(alpha=.2)
    handles, labels = ax[0,1].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=3,fontsize=8)
    fig.savefig(out/"dynamics.png",dpi=180)
    plt.close(fig)


def memo(out, phases, targets, gradients, update_notes):
    lines = ["# CAPPED500 离线训练动力学复核", "", "结论：日志支持后半程辅助项的实际作用没有随 raw MSE 一起变弱；但不能据此认定它导致 DEV 从500到1000回落。没有新增训练、模型前向、采样、匹配或评分；旧结果只读。", "", "记录从第1步开始，0是起点而非有损失观测的一步。主分段为1–500与501–1000。各臂每个同一步的8个主FM输入、分母和对应关系一致；跨时间段主输入不同，因此另列相对同一步FM对照的差值。", "", "| 臂 / 区间 | 主FM均值 | raw辅助MSE | 系数均值 | cap率 | clip率 | 实际aux/FM梯度比中位 | 负余弦/观测 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        for w in ("first500", "second500"):
            r=phases[arm,w]
            f=lambda x: "NA" if x is None else f"{x:.6f}"
            lines.append(f"| {arm} / {w} | {r['FM_mean']:.6f} | {r['raw_aux_mean']:.6f} | {f(r['coefficient_mean'])} | {f(r['cap_fraction'])} | {r['clip_fraction']:.3f} | {f(r['observed_aux_FM_ratio_median'])} | {r['observed_negative_cosines']}/{r['hook_n']} |")
    lines += ["", "cap系数按冻结公式由当前主FM和raw辅助MSE决定，激活时的加权loss近似等于FM。该加权loss本身不能充当独立学习证据。clip率是总梯度裁剪前范数>1的比例，不是参数更新范数。", "", "为减少周期边界与重复队列权重影响，补充四个96步窗口：1–96、405–500、501–596、905–1000。每个窗口都覆盖48个队列/父/t位置各两次，实际对应24个唯一状态目标；以下先在同一目标内取均值，再对24目标等权。teacher在自己的teacher-line状态内纵向对齐；它与TR/CAPPED并非同输入误差，只有TR和CAPPED的状态/GT目标逐哈希一致。", "", "| 臂 | 早期1–96 | 500前405–500 | 500后501–596 | 末期905–1000 | 500前到末期下降目标数 |", "|---|---:|---:|---:|---:|---:|"]
    for arm in (TEACHER, TR, CAP):
        vals=[phases[arm,w]["raw_aux_equal_target_mean"] for w in ("early96","pre500_96","post500_96","late96")]
        n=sum(r["pre500_96_to_late96_raw_aux_fractional_change"]<0 for r in targets if r["arm"]==arm)
        lines.append("| "+arm+" | "+" | ".join(f"{v:.6f}" for v in vals)+f" | {n}/24 |")
    lines += ["", "全部目标的t=.6/.8、队列、父与父×队列×t分层见 auxiliary_strata.csv；逐目标身份/次数/早中晚变化见 aligned_24_targets.csv。这些是模型沿训练过程遇到固定样本时的损失，不是在同一检查点重新统一评估24目标，不能把窗口均值冒称checkpoint精确性能。", "", "梯度仅来自第1步及每25步的真实加权辅助backward hooks，前半21次、后半20次；不存在的观测为NA。方向余弦没有显示CAPPED后半更负，不能声称方向冲突已被证实加剧。为限制样本构成影响，另仅配对两半都有hooks的相同目标：", "", "| 臂 | 两半均观测目标数/24 | 等权目标aux/FM比 前→后 | 等权目标余弦 前→后 |", "|---|---:|---|---|"]
    for r in gradients:
        a,b=r['first500_equal_target_aux_FM_grad_ratio'],r['second500_equal_target_aux_FM_grad_ratio']
        c,d=r['first500_equal_target_cosine'],r['second500_equal_target_cosine']
        lines.append(f"| {r['arm']} | {r['matched_targets']}/24 | {a:.6f}→{b:.6f} | {c:.6f}→{d:.6f} |")
    lines += ["", "该匹配仍是稀疏观测，不能外推到全部1000步的辅助梯度。CAPPED系数随着误差降低而增大是公式允许的机制，实际hooks与clip显示后半作用可增强；这与500后收益回落相容，但不是因果识别。固定24状态持续拟合也可能与生成轨迹、未覆盖状态、GT对应和继续FM漂移相关。没有逐检查点新生成或受控反事实，无法区分这些解释，更不能据此改变loss。", "", "真实参数step更新量：NA。日志未保存每步参数差/AdamW矩状态，现有gradient norm、clip或optimizer耗时不能代替更新幅度。这里只登记4000条实际更新记录及保存的时间。", "", "现存每臂权重为500、900（latest恢复点）、1000；另有保护START。大权重未加载、未重新全hash，核对先前完整hash清单、当前文件尺寸/mtime及sidecar。500/1000有完整旧DEV16输出，可继续零前向比较；900没有已保存DEV生成，无法纯离线估计其生成S/I/E。更早或600/700/800等点的权重已不在当前保存集合，损失日志不能重建其输出。500仍未过原形状/视觉联合门槛，因此现有资料支持把500作为值得保留的阶段现象，不支持宣称最优早停点或直接晋级。未来若另获授权，900可作为现存权重的补充验证点；本轮不执行。", "", "复现：CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -B analyze_dynamics.py。成功完成后再次运行仅核对源码/输入/输出hash并返回CACHE_PASS；未改变任何原实验文件。来源hash、检查计数及缺失口径见source_hashes.json、summary.json与completion.json。", ""]
    (out/"DYNAMICS_MEMO.md").write_text("\n".join(lines),encoding="utf-8")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old",default=str(OLD));p.add_argument("--out",default=str(HERE))
    args=p.parse_args()
    print(json.dumps(run(args.old,args.out),ensure_ascii=False,indent=2))
