"""Freeze existing inputs, model identity and exact budget; CPU-only."""
from __future__ import annotations
import argparse
import copy
from .common import ROOT, OUT, OLD, ARMS, EXPERIMENT, read, resolve, sha, write_new, resources

INITIAL_HASH = "28e1dc6903c28820f9bd39a99eb9bdadc1f4c6289af4a686dfb664e4ce996132"


def prepare():
    path = OUT / "registration.json"
    if path.exists():
        value = read(path)
        for p, h in value["protected_inputs"].items():
            if sha(p) != h:
                raise ValueError("Frozen input changed: " + p)
        return value
    oldreg = read(OLD / "training_registration.json")
    original = read(ROOT / "experiments/r2_bench_parts_convergence_v1/registration.json")
    em = read(OLD / "eval_manifest.json")
    requests = em["requests"]
    assert len(requests) == 48
    # Membership selection is independent of all saved generated outputs/scores.
    group_field = "split" if "split" in requests[0] else "group"
    train = sorted({q["parent_id"] for q in requests if q[group_field].upper() == "TRAIN"})
    dev = sorted({q["parent_id"] for q in requests if q[group_field].upper() == "DEV"})
    assert len(train) == 4 and len(dev) == 8
    process = []
    for parent in train + dev[:4]:
        qs = [q for q in requests if q["parent_id"] == parent]
        cid = min(q["condition_id"] for q in qs)
        chosen = sorted((q for q in qs if q["condition_id"] == cid), key=lambda q: q["request_id"])
        assert len(chosen) == 2
        process.extend(q["request_id"] for q in chosen)
    assert len(process) == 16 and len(set(process)) == 16
    labels = {}
    for label, arm, step, subset in (
        ("COMMON_S0", ARMS[0], 0, "process"), ("OFFICIAL_S0", ARMS[2], 0, "process"),
        ("A_S500", ARMS[0], 500, "process"), ("B_S500", ARMS[1], 500, "process"), ("C_S500", ARMS[2], 500, "process"),
        ("A_S1500", ARMS[0], 1500, "full"), ("B_S1500", ARMS[1], 1500, "full"), ("C_S1500", ARMS[2], 1500, "full")):
        labels[label] = dict(arm=arm, local_step=step, global_step=(0 if arm==ARMS[2] else 12000)+step,
                             subset=subset, generator_path=str(OUT / "inference" / f"{arm}_S{step}_generator.pt"),
                             requests=48 if subset=="full" else 16)
    protected = [OLD / "training_registration.json", OLD / "eval_manifest.json",
                 resolve(oldreg["manifest"]), resolve(oldreg["input_schedule"]), resolve(oldreg["probe_manifest"]),
                 ROOT / "experiments/r2_bench_parts_convergence_v1/registration.json",
                 ROOT / "experiments/r2_bench_parts_convergence_v1/REPORT.md",
                 OLD / "REPORT.md"]
    # Protect actual C/noise/query/layout identities already registered by the old manifest.
    for q in requests:
        for k in ("inference_npz", "metadata_npz", "source_npz", "noise_npz", "old_layout_npz"):
            if k in q:
                hashkey = "old_layout_sha256" if k == "old_layout_npz" else k + "_sha256"
                if sha(q[k]) != q[hashkey]:
                    raise ValueError("Original evaluation asset changed: " + k)
                protected.append(resolve(q[k]))
        descriptor = q["query_descriptor"]
        if sha(descriptor["path"]) != descriptor["sha256"]:
            raise ValueError("Frozen query changed")
        protected.append(resolve(descriptor["path"]))
    value = dict(schema="bench_adaptation_strength_scope_v1", status="FROZEN",
        experiment=EXPERIMENT, user_authorization="2026-10-20: user approved proposed Bench three-arm training; Chair unchanged",
        arms=list(ARMS), updates_per_arm=1500, total_updates=4500, batch=8, microbatch=1,
        train_instances=36000, generator_train_forwards=36000, generator_train_backwards=36000,
        optimizer_policy="fresh AdamW for all three; no inherited momentum; step0 anew",
        parameter_groups={"theta":"backbone.*", "phi":"role_embedding and context_encoder.*"},
        theta_lr=dict(zip(ARMS,[1e-5,2e-6,2e-6])), phi_lr=1e-5,
        betas=[.9,.95], eps=1e-8, weight_decay=0., clip=1., hybrid_lambda=.25,
        data={"parents":58,"conditions":195,"schedule":"exact existing CLEAN1500 schedule and deterministic per-example random streams"},
        head_policy="Frozen Bench SEAM_S15000 C-only cached layout, zero head training/forward in fixed panel",
        source_owner_policy="Training and supervised diagnostics only, never inference",
        fixed_probe_steps=[0,500,1000,1500], fixed_probe_inputs=96,
        fixed_probe_calls=1056, fixed_probe_sharing="A/B share identical COMMON step0; other9 checkpoints + Cstep0 =11x96",
        evaluation_labels=labels, process_request_ids=process,
        process_selection="All4TRAIN parents + first4DEV sorted parent IDs; first condition ID per parent; both existing noise IDs; no result-based selection",
        evaluation_requests=224, Euler_steps=50, sampling_generator_forwards=11200, D0_forwards=224,
        head_forwards=0, barrier="ALL224RAW_before_any_D0",
        full_panel="existing48 requests; TRAIN16 DEV32, open14/26 closed2/6",
        post="one official Bench D0, C/seam restore, GEO_C/source winding; HINGE_OPP015 -> ALPHA -> HOLE",
        metric_policy="unchanged179 fields, RAW/BASE/FINAL separate, closed seam NA, I is C query coverage; absolute counts and fixed BASE comparisons retained",
        visual_policy="actual AI_PROXY review; matched full views and original front/side sections; all final48 requests, process16",
        decision="A/B tests theta LR under COMMON+fresh optimizer; B/C tests full initialization packages under same future budget; not full factorial",
        progression={"primary":"C adoption and whole-shape/support integrity across paired DEV parents, then actual bulk connectivity and costs",
                     "screen":"at least2DEV parents with same-C both-noise consistent adoption/integrity gain; aggregate adoption/integrity not worse",
                     "cost_reference_pp":1.0,"cost_reference_status":"predeclared engineering screen for FF/CF/I/small angles, not statistical success; severe individual surface loss blocks promotion",
                     "no_best_selection":True,"automatic_extension":False,
                     "new_parent_confirmation":"not dispatched in this phase; only considered if frozen terminal result meets screen"},
        closest_prior="Bench COMMON/SEAM longitudinal DEV worsening and CLEAN1500; R01 frozen adapters/R20 recipes are different tasks; no repeat of a failed boundary-loss family",
        substantive_change="factor inherited-backbone learning rate and official-vs-COMMON initial package, without changing dataset/objective/layout",
        limits=["No new data, loss, WEDGE, translation, optimizer carryover, head update, Chair modification, cleanup, commit or push",
                "No implicit replay of failed neural calls/updates; safe checkpoint continuation only",
                "C loss at1500 cannot rule out all official-start recipes; lower MSE alone cannot establish task benefit"],
        protected_inputs={str(p):sha(p) for p in sorted(set(protected))})
    OUT.mkdir(parents=True,exist_ok=True)
    write_new(OUT / "resources_before_preparation.json", resources())
    write_new(OUT / "eval_manifest.json", em)
    write_new(path, value)
    return value


def freeze_training():
    scope=prepare()
    path=OUT / "training_registration.json"
    if path.exists():
        return read(path)
    old=read(OLD / "training_registration.json")
    original=read(ROOT / "experiments/r2_bench_parts_convergence_v1/registration.json")
    # Existing original registration is authoritative for category EMA identity.
    official=original["official"]
    official_sha=original["official_sha256"]
    code=[ROOT / "r2/pilots/bench_adaptation_strength_v1/training.py",
          ROOT / "r2/pilots/bench_adaptation_strength_v1/common.py",
          ROOT / "r2/pilots/bench_adaptation_strength_v1/register.py",
          ROOT / "r2/pilots/bench_control_transfer_v1/train.py"]
    code += [resolve(p) for p in old["training_code_hashes"]]
    reg=dict(schema="bench_adaptation_strength_registration_v1",status="FROZEN",experiment=EXPERIMENT,
        arms=list(ARMS),updates=1500,batch=8,microbatch=1,hybrid_lambda=.25,
        source_checkpoint=old["source_checkpoint"],source_checkpoint_sha256=old["source_checkpoint_sha256"],
        source_generator_sha256=old["source_generator_sha256"],
        official=official,official_sha256=official_sha,official_initial_generator_sha256=INITIAL_HASH,
        model_config=copy.deepcopy(old["model_config"]),training_seed=20261020,
        fixed_probe_local_steps=[0,500,1000,1500],
        scope_sha256=sha(OUT / "registration.json"),
        training_code_hashes={str(p):sha(p) for p in sorted(set(code))})
    for key in ("manifest","input_schedule","probe_manifest"):
        reg[key]=old[key];reg[key+"_sha256"]=old[key+"_sha256"]
    write_new(path,reg)
    return reg


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--training",action="store_true");a=p.parse_args()
    r=freeze_training() if a.training else prepare()
    print(r["status"],flush=True)
