"""Freeze authorized scope before CPU processing and the formal training gate."""
from __future__ import annotations
import argparse
from .common import ROOT, OUT, SOURCE_BENCH, read, sha, write_new, resources


def initial():
    destination = OUT / "registration.json"
    if destination.exists():
        r = read(destination)
        for path, expected in r["protected_small_inputs"].items():
            if sha(path) != expected:
                raise RuntimeError("Frozen input changed: " + path)
        return r
    paths = [
        ROOT / "maintenance/bench_clean_preparation_cleanup_20261020/taskbook_review.json",
        SOURCE_BENCH / "registration.json", SOURCE_BENCH / "data/manifest.json",
        SOURCE_BENCH / "evaluation_inputs/manifest.json",
        SOURCE_BENCH / "fixed_loss_probe/manifest.json",
        ROOT / "experiments/r2_bench_parts_v1/data/conditions.csv",
        SOURCE_BENCH / "REPORT.md",
        ROOT / "experiments/r2_chair_multipart_seam_v1/REPORT.md",
    ]
    r = dict(
        schema="R2_BENCH_CLEAN_MULTIPART_SEAM_V1", status="FROZEN_SCOPE",
        user_authorization="User approved corrected plan on 2026-10-20: execute actual experiment",
        category="bench", initialization="COMMON12000 live + matching generator AdamW/RNG",
        source_checkpoint=str(SOURCE_BENCH / "training/COMMON/step12000.pt"),
        checkpoint_sha256="40aecccd1838a5b04dadeac13e72b9cf7315f3721c16e180b97844a04c2f8032",
        train_updates=1500, global_interval=[12000, 13500], batch=8, microbatch=1,
        train_samples=12000, generator_train_forwards=12000, generator_backwards=12000,
        head_updates=1500, head_train_sample_forwards=12000,
        data=dict(original_train=277, original_dev=39, train_closed=44, dev_closed=5,
                  eligible_train=195, eligible_parents=58, remaining_multicomponent=69,
                  rule="TRAIN, boundary_edges>0, source_owner_count>0; no target geometry repair",
                  parent_sampling="Uniform cyclic parents; within-parent cyclic parts and conditions",
                  batch="4 distinct parents each clean plus anisotropic augmented instance"),
        augmentation=dict(scale_range=[.92, 1.08], axes="xyz independently",
                          OT_order="Scale source C/Y and geometry first, then role-restricted OT, then HYBRID.25",
                          Gaussian="Unchanged standard Gaussian; no scaling of noise",
                          origin="Existing normalized model origin; no new per-C recentering"),
        generator_objective="Existing free-corner FM, unchanged weighting and scalar denominator",
        head=dict(parameters=18049, input_dim=28, layers=2, width=32, heads=4, dropout=.1,
                  normalization="Frozen featurewise mean/std on original-scale eligible TRAIN edges only; extra preprocessing versus Chair",
                  loss="Unweighted BCE averaged over each C then over batch",
                  optimizer="fresh AdamW lr.001 weight_decay.0001",
                  inference_threshold="logit>=0", source_owner_inference=False,
                  generator_gradient=False),
        fixed_probe=dict(local_steps=[0, 500, 1000, 1500], inputs="Exact old fixed loss-probe arrays",
                         forwards_per_checkpoint=96, total_generator_forwards=384,
                         extra_loss_probe_or_synthetic_neural_tests=False),
        evaluation=dict(requests=48, unique_C=24, noises_per_C=2,
                        TRAIN_requests=16, DEV_requests=32, closed_requests=8, open_requests=40,
                        new_labels=["NEW_HEAD", "OLD_S15000_LAYOUT"],
                        new_RAW_budget=96, Euler_steps=50, generator_forwards_budget=4800,
                        D0_forwards_budget=96, D0_category="official Bench", D0_iterations=1,
                        head_diagnostics="Original full TRAIN277/DEV39 including excluded C; closed C NA",
                        raw_barrier="All96 RAW must be saved before D0 begins",
                        historical_labels=["NATIVE_S21000", "SEAM_S15000", "SEAM_S21000"],
                        historical_CPU_requests=144,
                        comparison="5 inference configurations x48 requests x2 post stages =480 rows; RAW separate",
                        same_noise="Reuse exact original arrays, request IDs, C/N and query locations",
                        post="Original D0/GEO_C/source orientation; BASE->HINGE_OPP015->ALPHA->HOLE",
                        D0_guard=False, GEO_no_self_collapse=False,
                        safety="C exact; reject newly illegal CF/FF pairs, lost legal CF/FF edges, new nonmanifold/same-direction edges; new faces angle>=5deg q<=10",
                        metrics="Preserve179 original fields. FixedBASE denominators only for applicable face fractions; I uses Cqueries; graph area domains explicit; Native seam-only metrics NA",
                        visual="48 actual nonblind AI_PROXY comparison pages, same full views and original front/side section positions"),
        scope_limits=[
            "New short Bench recipe, not matched cumulative Chair SEAM history or same-budget Native comparison",
            "Source owner and slit labels diagnose a reference completion, not unique correct generation",
            "No predetermined success, no extra arms, new losses, threshold sweep, best checkpoint, or extension",
            "No automatic retry of started failed neural call/update; preserve ledger and require explicit recovery review",
            "Engineering fixes before neural work and CPU-only versioned fixes allowed without changing science",
            "No official downloads, cleanup, default switch, commit or push",
        ],
        dispatch=dict(CPU_workers_initial=2, threads_per_worker=1, GPU_training_workers=1,
                      headroom="max(2GiB,10%total) plus1.2 multiplier for any incremental comparable worker",
                      sampling="Incremental measured concurrency; budget unchanged; never overlap D0 before RAW barrier",
                      training_peak_reference="Chair multipart1500; measure actual Bench first authorized updates including optimizer",
                      independent_CPU_overlap=True),
        protected_small_inputs={str(p): sha(p) for p in paths},
    )
    write_new(OUT / "resources_before_preparation.json", resources())
    write_new(destination, r)
    return r


def training_gate():
    r = initial()
    required = [OUT / "data/manifest.json", OUT / "data/schedule.json",
                OUT / "data/cpu_checks.json", OUT / "eval_manifest.json"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("Training gate incomplete: " + repr(missing))
    m = read(required[0])
    assert len(m["train"]) == 195
    assert len({x["parent_id"] for x in m["train"]}) == 58
    if read(required[2]).get("status") != "PASS":
        raise RuntimeError("CPU checks did not pass")
    source = r["source_checkpoint"]
    if sha(source) != r["checkpoint_sha256"]:
        raise RuntimeError("COMMON12000 changed")
    code = sorted((ROOT / "r2/pilots/bench_clean_multipart_seam_v1").glob("*.py"))
    value = dict(status="PASS", budget_registration_sha256=sha(OUT / "registration.json"),
                 frozen_inputs={str(p): sha(p) for p in required},
                 source_checkpoint=source, source_sha256=r["checkpoint_sha256"],
                 implementation_hashes={str(p): sha(p) for p in code},
                 resources=resources(), no_updates_yet=True)
    write_new(OUT / "training_gate.json", value)
    return value


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--training-gate", action="store_true")
    a = p.parse_args()
    result = training_gate() if a.training_gate else initial()
    print(result["status"], flush=True)
