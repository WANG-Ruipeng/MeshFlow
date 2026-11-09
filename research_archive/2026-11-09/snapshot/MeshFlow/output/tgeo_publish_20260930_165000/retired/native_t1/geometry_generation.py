"""Fixed endpoint-only generation matrix and read-only light scoring."""
from pathlib import Path
import time
import json
import numpy as np
import torch
from .artifacts import atomic_json, file_sha256, state_sha256, array_hash
from .sampling import make_noise, clamped_sample, new_counts
from .geometry_pilot import (read_json, load_initial, free_memory, TimingObserver, ARMS, GENERATION_SEEDS)
from .portable_checkpoint import load_trained_checkpoint
from .geometry_checkpoint import load_geometry_checkpoint
from .geometry_evaluation import load_evaluation, score_output, binding_pair


def _load(args, name):
    if name=="T0":
        return load_initial(args)
    training=read_json(args.out/name/"training.json")
    if training["status"]!="PASS" or training["completed_updates"]!=500:
        raise RuntimeError("Only completed additional-step500 endpoints may be evaluated")
    endpoint=Path(training["checkpoints"][-1]["path"])
    if file_sha256(endpoint)!=training["checkpoints"][-1]["sha256"]:
        raise RuntimeError("Endpoint file changed")
    return load_trained_checkpoint(endpoint) if name=="T_base" else load_geometry_checkpoint(endpoint)


def _same(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a.view(np.uint32),b.view(np.uint32))


def generate(args):
    # Fail closed before source/asset checks, evaluation-array IO, model loading,
    # or creation of raw outputs. Explicit --stage generate cannot bypass STOPPED.
    gates = read_json(args.out/"gates.json")
    if gates.get("status") != "PASS":
        raise RuntimeError("Generation requires passed GPU gates")
    training_summary = read_json(args.out/"training_summary.json")
    if (training_summary.get("status") != "PASS"
            or training_summary.get("total_optimizer_updates") != 1500):
        raise RuntimeError("Generation requires all 1500 registered training updates")
    for arm in ARMS:
        completed = read_json(args.out/arm/"training.json")
        if (completed.get("status") != "PASS" or completed.get("completed_updates") != 500
                or completed.get("counts", {}).get("optimizer_updates") != 500
                or completed.get("update_in_progress") is not False):
            raise RuntimeError("Generation requires completed 500-update training for " + arm)
        checkpoints = completed.get("checkpoints", [])
        if not checkpoints or checkpoints[-1].get("cumulative_updates") != 2000:
            raise RuntimeError("Final cumulative2000 checkpoint is missing for " + arm)
        endpoint = Path(checkpoints[-1].get("path", ""))
        if not endpoint.is_file():
            raise FileNotFoundError("Final checkpoint does not exist for " + arm + ": " + str(endpoint))
    if (args.out/"generation.json").exists() or (args.out/"raw").exists():
        raise FileExistsError("Formal generation is never automatically repeated")
    registration=read_json(args.out/"registration.json")
    root=Path(__file__).resolve().parents[1]
    if any(file_sha256(root/name)!=digest for name,digest in registration["execution_sources"].items()):
        raise RuntimeError("Execution code changed after registration")
    from .baseline import BASELINE_CHECKPOINT
    if file_sha256(args.checkpoint or BASELINE_CHECKPOINT)!=registration["initialization_file_sha256"]:
        raise RuntimeError("Generation initialization changed after registration")
    bank,evaluation=load_evaluation(args.out/"registration/evaluation_inputs.npz",
                                    registration["evaluation_inputs_sha256"])
    args.out.joinpath("raw").mkdir()
    counts=new_counts();replay_counts=new_counts()
    report=dict(status="RUNNING",formal_counts=counts,replay_counts=replay_counts,
                rows=[],models={},replays=[],binding=[],
                expected_formal_rollouts=evaluation["formal_rollout_count"],
                timing=dict(sampling_seconds=0.,light_geometry_seconds=0.,archive_write_seconds=0.,
                            context_cuda_seconds=0.,context_wall_seconds=0.,model_load_seconds=0.))
    atomic_json(args.out/"generation.json",report)
    start=time.perf_counter()
    try:
        for name in ("T0",*ARMS):
            tick=time.perf_counter()
            model,audit=_load(args,name)
            if name=="T0":
                if audit["checkpoint_sha256"]!=registration["initialization_file_sha256"]:
                    raise RuntimeError("T0 is not the registered common initialization")
            elif audit["cumulative_updates"]!=2000:
                raise RuntimeError("Generation endpoint is not cumulative2000")
            if name in ("T_geo","T_graph"):
                if (audit["context"]["context_mode"]!=name.split("_",1)[1]
                        or audit["base_identity"]["checkpoint_sha256"]!=registration["initialization_file_sha256"]):
                    raise RuntimeError("Geometry endpoint architecture/common initialization differs")
            report["timing"]["model_load_seconds"]+=time.perf_counter()-tick
            state_before=state_sha256(model)
            report["models"][name]=dict(load=audit,state_before=state_before)
            first_output=None
            for p,j in sorted(bank["cases"]):
                case=bank["cases"][p,j]
                seeds=GENERATION_SEEDS if j==0 else GENERATION_SEEDS[:2]
                for seed in seeds:
                    if counts["rollout_attempts"]>=64:
                        raise RuntimeError("64 formal rollout cap exceeded")
                    identifier=f"{name}_P{p}_patch{j}_seed{seed}"
                    z=make_noise(seed)
                    C=torch.from_numpy(case.constraints.copy()).cuda()
                    observer=TimingObserver() if model.context_encoder is not None else None
                    model.context_observer=observer
                    output,path,sampling=clamped_sample(model,z.cuda(),C,counts)
                    timing=observer.result() if observer is not None else None
                    model.context_observer=None
                    report["timing"]["sampling_seconds"]+=sampling["seconds"]
                    if timing:
                        report["timing"]["context_cuda_seconds"]+=timing["cuda_event_seconds"]
                        report["timing"]["context_wall_seconds"]+=timing["wall_seconds"]
                    raw_path=args.out/"raw"/(identifier+".npz")
                    tick=time.perf_counter()
                    np.savez_compressed(raw_path,output=output,path=path,initial_gaussian=z.numpy())
                    report["timing"]["archive_write_seconds"]+=time.perf_counter()-tick
                    tick=time.perf_counter()
                    metrics=score_output(output,bank,p,j,model_name=name,noise_seed=seed)
                    report["timing"]["light_geometry_seconds"]+=time.perf_counter()-tick
                    row=dict(output_id=identifier,model=name,parent=p,patch_id=j,
                        patch_group="original" if j==0 else "new",seed=seed,
                        raw_path=str(raw_path.resolve()),raw_sha256=file_sha256(raw_path),
                        noise_sha256=array_hash(z),condition_sha256=array_hash(case.constraints),
                        sampling=sampling,context_timing=timing,metrics=metrics)
                    report["rows"].append(row)
                    if first_output is None:
                        first_output=row
                    atomic_json(args.out/"generation.json",report)
                    print(f"formal {len(report['rows'])}/{evaluation['formal_rollout_count']} {identifier}",flush=True)
                    del C,z,output,path
            report["models"][name]["state_after"]=state_sha256(model)
            if report["models"][name]["state_after"]!=state_before:
                raise RuntimeError("Frozen generation model changed")
            del model;free_memory()
            # Exactly three endpoint replays, one per trained arm, after a fresh strict reload.
            if name!="T0":
                if replay_counts["rollout_attempts"]>=3:
                    raise RuntimeError("Three endpoint replay cap exceeded")
                reloaded,reload_audit=_load(args,name)
                p,j,seed=first_output["parent"],first_output["patch_id"],first_output["seed"]
                C=torch.from_numpy(bank["cases"][p,j].constraints.copy()).cuda()
                output,path,sampling=clamped_sample(reloaded,make_noise(seed).cuda(),C,replay_counts)
                with np.load(first_output["raw_path"],allow_pickle=False) as reference:
                    exact=_same(output,reference["output"]) and _same(path,reference["path"])
                if not exact:
                    raise RuntimeError("Freshly reloaded full trajectory differs")
                report["replays"].append(dict(model=name,reference=first_output["output_id"],
                    all51_states_bitwise=True,endpoint_bitwise=True,load=reload_audit,sampling=sampling))
                del reloaded,C,output,path
                free_memory()
                atomic_json(args.out/"generation.json",report)
        for name in ("T0",*ARMS):
            for j in sorted(bank["references"]):
                for seed in (GENERATION_SEEDS if j==0 else GENERATION_SEEDS[:2]):
                    paired=[next(row["metrics"] for row in report["rows"]
                                 if (row["model"],row["parent"],row["patch_id"],row["seed"])==(name,p,j,seed))
                            for p in (0,1)]
                    report["binding"].append(binding_pair(*paired,patch_index=j))
        if counts["complete_rollouts"]!=evaluation["formal_rollout_count"]:
            raise RuntimeError("Formal generation count differs from frozen registration")
        gate=read_json(args.out/"gates.json")
        if replay_counts["complete_rollouts"]+gate["replay_counts"]["complete_rollouts"]>4:
            raise RuntimeError("Total replay cap exceeded")
        # External offline audit receives targets/IDs only after model sampling.
        patches=[]
        for case in evaluation["cases"]:
            p,j=case["parent_index"],case["patch_index"]
            prefix=f"parent{p}_patch{j}_"
            patches.append(dict(parent=p,patch_id=j,patch_group="original" if j==0 else "new",
                constraints_key=prefix+"constraints",target_key=f"parent{p}_target",
                known_face_ids_key=prefix+"source_face_ids",
                source_vertices_key=f"parent{p}_source_vertices",
                source_face_vertex_ids_key=f"parent{p}_source_face_vertex_ids",
                bbox_diagonal=case["bbox_diagonal"]))
        manifest=dict(schema="geometry_context_external_audit_v1",
            arrays_path=str((args.out/"registration/evaluation_inputs.npz").resolve()),
            arrays_sha256=registration["evaluation_inputs_sha256"],patches=patches,
            outputs=[{k:row[k] for k in ("output_id","model","parent","patch_id","patch_group","seed",
                                        "raw_path","raw_sha256")} for row in report["rows"]])
        atomic_json(args.out/"generated_outputs.json",manifest)
        report["status"]="PASS"
    except BaseException as error:
        report.update(status="STOPPED",error=repr(error))
        raise
    finally:
        report["seconds"]=time.perf_counter()-start
        atomic_json(args.out/"generation.json",report)
    return report
