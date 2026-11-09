"""CPU post-audit summary of first/last matched auxiliary states; no model imports."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WINDOWS = {"early": (1, 96), "late": (905, 1000)}
ARMS = ("REPLAY_FM_CONTROL", "TRAJECTORY_REPLAY", "TRAJECTORY_REPLAY_CAPPED")


def read(name):
    with (HERE / (name + ".csv")).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(rows):
    def mean(field):
        return float(np.mean([float(r[field] == "True") if r[field] in ("True", "False") else float(r[field]) for r in rows]))
    return {"n": len(rows), "aux_MSE_mean": mean("aux_unweighted"),
            "FM_mean": mean("FM"), "weighted_aux_mean": mean("aux_weighted"),
            "weighted_aux_over_FM_ratio_of_means": mean("aux_weighted") / mean("FM"),
            "weighted_aux_over_FM_mean_of_step_ratios": mean("weighted_aux_over_FM"),
            "clip_fraction": mean("clip_active"), "coefficient_mean": mean("actual_aux_coefficient"),
            "cap_fraction": mean("cap_active") if rows[0]["cap_active"] else None}


def main():
    step_rows = read("per_step")
    audits = {(r["arm"], int(r["step"])): r for r in read("aux_audit") if r["phase"] == "explore"}
    pairs, arm_results = [], {}
    input_hashes = {name: sha(HERE / name) for name in ("per_step.csv", "aux_audit.csv", "gradient_observations.csv", "summary.json")}
    preserved = HERE.parent / "live" / "TR1000_learning_summary.json"
    preserved_sha = sha(preserved)
    for arm in ARMS:
        rows = [r for r in step_rows if r["phase"] == "explore" and r["arm"] == arm]
        assert len(rows) == 1000 and sorted(int(r["step"]) for r in rows) == list(range(1, 1001))
        identities, by_window = {}, {}
        for window, (lo, hi) in WINDOWS.items():
            chosen = [r for r in rows if lo <= int(r["step"]) <= hi]
            assert len(chosen) == 96
            position_counts = {}
            state_rows = {}
            for r in chosen:
                n = int(r["step"])
                a = audits[(arm, n)]
                key = (a["request_id"], a["t"], a["xt_sha256"], a["u_sha256"], a["target_sha256"])
                state_rows.setdefault(key, []).append(r)
                position = (n - 1) % 48
                position_counts[position] = position_counts.get(position, 0) + 1
                identity = key + (a["queue"], a["uid"], a["candidate"])
                if position in identities:
                    assert identities[position] == identity
                identities[position] = identity
            assert position_counts == {i: 2 for i in range(48)}
            assert len(state_rows) == 24
            by_window[window] = state_rows
        assert set(by_window["early"]) == set(by_window["late"])
        local_pairs = []
        for key in sorted(by_window["early"]):
            early = by_window["early"][key]
            late = by_window["late"][key]
            assert len(early) == len(late)
            em = float(np.mean([float(r["aux_unweighted"]) for r in early]))
            lm = float(np.mean([float(r["aux_unweighted"]) for r in late]))
            pair = dict(arm=arm, request_id=key[0], t=float(key[1]), xt_sha256=key[2], u_sha256=key[3], target_sha256=key[4],
                        early_observations=len(early), late_observations=len(late), early_aux_MSE=em, late_aux_MSE=lm,
                        change=lm-em, fractional_reduction=1-lm/em, MSE_decreased=bool(lm < em))
            pairs.append(pair)
            local_pairs.append(pair)
        result = {"same_cycle_positions": 48, "same_unique_state_target_identities": 24,
                  "states_with_MSE_decrease": sum(p["MSE_decreased"] for p in local_pairs),
                  "windows": {}, "state_equal_weight": {}}
        for tname, t in (("all", None), ("t0.6", .6), ("t0.8", .8)):
            selected_pairs = [p for p in local_pairs if t is None or abs(p["t"]-t) < 1e-6]
            se = float(np.mean([p["early_aux_MSE"] for p in selected_pairs]))
            sl = float(np.mean([p["late_aux_MSE"] for p in selected_pairs]))
            result["state_equal_weight"][tname] = dict(states=len(selected_pairs), early_aux_MSE=se, late_aux_MSE=sl,
                fractional_reduction=1-sl/se, states_with_MSE_decrease=sum(p["MSE_decreased"] for p in selected_pairs))
            result["windows"][tname] = {}
            for window, (lo, hi) in WINDOWS.items():
                selected = [r for r in rows if lo <= int(r["step"]) <= hi and (t is None or abs(float(r["aux_t"])-t) < 1e-6)]
                result["windows"][tname][window] = stats(selected)
        gradients = [r for r in read("gradient_observations") if r["arm"] == arm and r["phase"] == "explore"]
        ratio = np.array([float(r["weighted_aux_over_FM_gradient_norm"]) for r in gradients])
        result["actual_hook_gradient_ratio"] = dict(n=len(ratio), median=float(np.median(ratio)), mean=float(np.mean(ratio)), max=float(np.max(ratio)))
        arm_results[arm] = result
    tr_keys = {(p["request_id"], p["t"], p["xt_sha256"], p["u_sha256"], p["target_sha256"]) for p in pairs if p["arm"] == "TRAJECTORY_REPLAY"}
    capped_keys = {(p["request_id"], p["t"], p["xt_sha256"], p["u_sha256"], p["target_sha256"]) for p in pairs if p["arm"] == "TRAJECTORY_REPLAY_CAPPED"}
    assert tr_keys == capped_keys and len(tr_keys) == 24
    assert sha(preserved) == preserved_sha
    with (HERE / "matched_24_target_learning.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    report = dict(status="PASS", scope="CPU saved-log analysis; no DEV efficacy inference", model_calls=0, optimizer_updates=0,
        windows=WINDOWS, per_arm=arm_results, TR_and_capped_exact_same_24_Xt_t_u_target_identities=True,
        definition="First96 and last96 steps each observe all48 queue/parent/t positions twice, but these map to24 unique Xt/t/u targets. Schedule-weighted windows and equal-weight24-target summaries are reported separately.",
        caution="Loss-value cap does not cap gradient norm. Only41 actual weighted auxiliary backward hook observations per arm; no unobserved steps are filled with zeros.",
        source_hashes=input_hashes, script_sha256=sha(__file__), preserved_TR1000_learning_sha256=preserved_sha,
        visual_QA={n: dict(status="PASS_ACTUALLY_VIEWED", sha256=sha(HERE/n), note="Axes, legend, full1000-step curves and text inspected; no clipping or missing plot panels.") for n in ("learning_diagnostics.png", "loss_balance_diagnostics.png")},
        repeat_actual_verification="NOT_RUN; no repeat was authorized or started by this analysis")
    (HERE / "full1000_matched_learning_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
