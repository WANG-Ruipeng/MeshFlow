"""Freeze the authorized two-arm mechanism comparison without neural calls."""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
from .common import ROOT, OUT, ARMS, EXPERIMENT, read, resolve, sha, write_new, resources

BASE = ROOT / "experiments/r2_bench_adaptation_strength_v1"


def freeze():
    if (OUT / "training_registration.json").exists():
        raise FileExistsError("Registration already frozen; use training --check-cpu")
    old = read(BASE / "training_registration.json")
    source = BASE / "training/A_COMMON_LR1E5/checkpoint_step0.pt"
    receipt = read(source.with_suffix(".json"))
    if sha(source) != receipt["sha256"]:
        raise ValueError("A initial checkpoint differs")
    original_eval = BASE / "eval_manifest.json"
    evaluation = OUT / "eval_manifest.json"
    if not evaluation.exists():
        shutil.copyfile(original_eval, evaluation)
    if sha(evaluation) != sha(original_eval):
        raise ValueError("Eval must be byte-identical to original A")
    if len(read(evaluation)["requests"]) != 48:
        raise ValueError("Expected 48 requests")
    own = Path(__file__).resolve().parent
    paths = [resolve(p) for p in old["training_code_hashes"]]
    paths += [own / n for n in ("__init__.py", "common.py", "training.py", "register.py",
                               "async_core.py", "perturbed_core.py",
                               "test_async_core.py", "test_perturbed_core.py", "cpu_gate.py")]
    gate = read(OUT / "cpu_gate.json")
    if gate["status"] != "PASS" or gate["neural_forwards"] != 0:
        raise ValueError("CPU gate missing")
    for p, h in old["training_code_hashes"].items():
        if sha(p) != h:
            raise ValueError("Original scientific source changed: " + p)
    reg = dict(schema="bench_state_schedule_registration_v1", status="FROZEN", experiment=EXPERIMENT,
        arms=list(ARMS), updates=1500, batch=8, microbatch=1, hybrid_lambda=.25,
        source_checkpoint=str(source), source_checkpoint_sha256=receipt["sha256"],
        source_generator_sha256=receipt["generator_state_sha256"],
        source_optimizer_sha256=receipt["optimizer_state_sha256"],
        source_rng_sha256=receipt["rng_sha256"],
        source_global_step=12000, final_global_step=13500, training_seed=20261020,
        model_config=old["model_config"], training_code_hashes={str(p): sha(p) for p in paths},
        protocol_sha256=sha(OUT / "PROTOCOL.md"), cpu_gate_sha256=sha(OUT / "cpu_gate.json"),
        optimizer="exact A step0 fresh AdamW import; all groups1e-5 beta(.9,.95) eps1e-8 wd0 clip1",
        new_parameters=0, fixed_probe_forwards=0, fresh_training_forwards_per_arm=12000,
        fresh_backwards_per_arm=12000, fresh_updates_per_arm=1500,
        precision="BF16forward FP32parameters/integration; deterministic SDPA math TF32off",
        tau=.7, perturbation=dict(probability_given_eligible=.25, time_interval=[.3,.9],
            alpha_interval=[.3,.8], center="original C corners vs all free target corners",
            state="xt+t*d on free only", target="(Y-xt_corrupted)/(1-t)",
            interpretation="off-path endpoint corrective regression, not true derivative of that corruption path"),
        asynchronous=dict(clock="min(t/.7,1)", prediction="local-clock velocity U=Y-Z_H",
            tip_loss="active only t<.7", denominator="original free scalar count",
            attention="C1/seam_s/bulk_t", corner="anchors1/tip_s/bulk_t",
            role="unchanged anchor identity", Geo="original C only",
            integration="Euler50 bulk dt; tip delta_s, frozen generated tip after step35"),
        failure_contract="No silent retry, rollback or numerical setting change. Exact-cursor safe pauses may resume without replay.",
        evaluation_manifest=str(evaluation), evaluation_manifest_sha256=sha(evaluation),
        expected_new_raw=96, expected_new_d0=96, expected_new_score_rows=288,
        generator_forward_sampling=4800, final_selection="1500 only; no best selection",
        reused_baseline=dict(root=str(BASE), label="A_S1500",
            eval_manifest_sha256=sha(original_eval)),
        evaluation_labels={arm+"_S1500": dict(arm=arm, subset="full", local_step=1500,
            global_step=13500, generator_path=str(OUT/"training"/arm/"generator_step1500.pt")) for arm in ARMS},
        no_commit_push_default_change=True)
    for key in ("manifest", "input_schedule"):
        reg[key] = old[key]
        reg[key+"_sha256"] = old[key+"_sha256"]
    write_new(OUT/"resource_initial.json", resources())
    write_new(OUT/"registration.json", reg)
    write_new(OUT/"training_registration.json", reg)
    return dict(status="FROZEN", arms=list(ARMS), updates_each=1500,
                new_raw=96, new_D0=96, SHA256=sha(OUT/"training_registration.json"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.parse_args()
    print(freeze())
