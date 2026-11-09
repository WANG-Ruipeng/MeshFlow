"""Explicit 3x500 geometry-context pilot. No automatic budget extension."""
from .runtime import configure_stable_runtime
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import torch
import numpy as np

from .artifacts import atomic_json, file_sha256, state_sha256
from .baseline import load_baseline, BASELINE_CHECKPOINT
from .portable_checkpoint import load_trained_checkpoint
from .checkpoint import DEFAULT_CONFIG
from .data import load_archive, NativeTrainingStream, native_collate, model_inputs
from .training import train_step, make_optimizer
from .geometry_accounting import count_ot_calls
from .train_cli import save_checkpoint
from .model import native_execution
from .context_geometry import attach_context_encoder, ContextGeometryObserver
from .geometry_checkpoint import save_geometry_checkpoint, load_geometry_checkpoint

ARMS = ("T_base","T_geo","T_graph")
GENERATION_SEEDS = (9701,9702,9703,9704)
INPUT_SEED = 1010
ENCODER_SEED = 1010


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_initial(args, *, device="cuda"):
    if args.init_profile == "a-continue":
        model,audit=load_baseline(args.checkpoint or BASELINE_CHECKPOINT,device=device)
    else:
        if args.checkpoint is None:
            raise ValueError("Portable initialization requires --checkpoint")
        model,audit=load_trained_checkpoint(args.checkpoint,device=device)
    if audit.get("cumulative_updates") != 1500:
        raise ValueError("A single pure T1 cumulative1500 initialization is required")
    return model,audit


def free_memory():
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def recorded_forward(model, batch, counts, *, gradients=False):
    counts["forward_attempts"]=counts.get("forward_attempts",0)+1
    with torch.set_grad_enabled(gradients), native_execution(model,coordinates=batch["xt"]):
        result=model(*model_inputs(batch)).float()
    counts["forward_returns"]=counts.get("forward_returns",0)+1
    return result


def exact_tensor(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


class TimingObserver:
    requires_geometry_details=False
    requires_injection_details=False

    def __init__(self, diagnostic=False):
        self.requires_geometry_details=bool(diagnostic)
        self.requires_injection_details=bool(diagnostic)
        self.diagnostic=ContextGeometryObserver() if diagnostic else None
        self.events=[]
        self.wall=[]
        self.started=None

    def __call__(self, record):
        if record["site"]=="geometry_start":
            event=torch.cuda.Event(enable_timing=True);event.record()
            self.started=(event,time.perf_counter())
        elif record["site"]=="geometry_end":
            event=torch.cuda.Event(enable_timing=True);event.record()
            begin,wall=self.started
            self.events.append((begin,event));self.wall.append(time.perf_counter()-wall)
            self.started=None
        elif self.diagnostic is not None:
            self.diagnostic(record)

    def result(self):
        return dict(calls=len(self.events),
                    cuda_event_seconds=sum(a.elapsed_time(b) for a,b in self.events)/1000,
                    wall_seconds=sum(self.wall),
                    timing_scope="C extraction/validation, geometry, graph and FP32 MLP; excludes backbone; GPU events include intervening synchronization gaps",
                    diagnostics=self.diagnostic.as_dict() if self.diagnostic is not None else None)


def gradient_update_summary(module,before):
    groups={}
    for name,p in module.named_parameters():
        key=name.split(".")[0]
        row=groups.setdefault(key,dict(parameter_elements=0,gradient_squared_sum=0.,update_squared_sum=0.,
                                      nonzero_gradient_elements=0,nonzero_update_elements=0))
        grad=p.grad.detach().float() if p.grad is not None else torch.zeros_like(p)
        change=p.detach().float()-before[name].to(p.device)
        row["parameter_elements"]+=p.numel()
        row["gradient_squared_sum"]+=float(grad.double().square().sum())
        row["update_squared_sum"]+=float(change.double().square().sum())
        row["nonzero_gradient_elements"]+=int(torch.count_nonzero(grad))
        row["nonzero_update_elements"]+=int(torch.count_nonzero(change))
    for row in groups.values():
        row["gradient_RMS_after_global_clip"]=(row["gradient_squared_sum"]/row["parameter_elements"])**.5
        row["actual_update_RMS"]=(row["update_squared_sum"]/row["parameter_elements"])**.5
    return groups


def assert_contract(args):
    if args.updates != 500 or args.input_seed != INPUT_SEED:
        raise ValueError("Registered pilot fixes 500 updates per arm and new input seed1010")
    if not args.inputs.is_file():
        raise FileNotFoundError(args.inputs)
    if (args.checkpoint or BASELINE_CHECKPOINT).is_file() is False:
        raise FileNotFoundError(args.checkpoint or BASELINE_CHECKPOINT)


def register(args):
    from .geometry_evaluation import prepare_evaluation, load_evaluation
    from .context_geometry import face_geometry, ENCODER_METADATA
    assert_contract(args)
    args.out.mkdir(parents=True,exist_ok=True)
    path=args.out/"registration.json"
    if path.exists():
        raise FileExistsError(path)
    from .geometry_gates import GATE_POLICY, STRICT_GATE_POLICY, BASELINE_EVIDENCE_RESULTS, BASELINE_EVIDENCE_VELOCITIES
    evidence_path=getattr(args,"baseline_regression_evidence",None)
    evidence=None
    if evidence_path is not None:
        if evidence_path.resolve()!=BASELINE_EVIDENCE_RESULTS.resolve():
            raise ValueError("Revised policy requires the authorized baseline code audit")
        evidence=dict(results_path=str(evidence_path.resolve()),results_sha256=file_sha256(evidence_path),
            velocities_path=str(BASELINE_EVIDENCE_VELOCITIES.resolve()),
            velocities_sha256=file_sha256(BASELINE_EVIDENCE_VELOCITIES))
    archive=load_archive(args.inputs)
    # Construct a disposable stream only to validate support; do not consume it.
    stream=NativeTrainingStream(archive,seed=INPUT_SEED)
    initial_path=args.checkpoint or BASELINE_CHECKPOINT
    contract=dict(schema="native_t1_geometry_pilot_registration_v1",
        baseline_commit="2502d5f89c54706cc4700d3cdea4a12fd6792529",
        gate_policy=GATE_POLICY if evidence else STRICT_GATE_POLICY,baseline_regression_evidence=evidence,
        gate_policy_reason=("Original T0 and current wrapper are bitwise identical; FP32/first-block diagnosis localizes inherited BF16 order sensitivity. Historical numerical FAIL is retained separately from hard wiring gates."
            if evidence else "No baseline exemption: historical BF16 permutation tolerance remains a hard gate."),
        initialization_profile=args.init_profile,initialization_path=str(initial_path.resolve()),
        initialization_file_sha256=file_sha256(initial_path),base_cumulative_updates=1500,
        inputs_path=str(args.inputs.resolve()),inputs_sha256=archive.sha256,
        updates_per_arm=500,total_update_cap=1500,arms=list(ARMS),
        input_seed=INPUT_SEED,initial_stream_state=stream.state_dict(),
        encoder_initialization_seed=ENCODER_SEED,generation_seeds=list(GENERATION_SEEDS),
        formal_rollout_cap=64,replay_rollout_cap=4,
        diagnostic_steps=[1,10,100,500],full_generation_endpoints_only=500,
        bf16_permutation_atol=.015625,bf16_permutation_rtol=.03125,
        zero_repeat_bitwise_required=True,
        loss="original masked_free_fm",new_losses=False,
        optimizer=dict(name="AdamW",lr=1e-5,betas=[.9,.95],weight_decay=0,grad_clip=1),
        effective_batch=8,microbatch=1,precision="BF16 backbone; FP32 parameters/integration/geometry branch",
        stop_on_numerical_error=True,postprocessing="NONE",trained_objects_not_generalization=True)
    evaluation_path=args.out/"registration/evaluation_inputs.npz"
    if evaluation_path.exists():
        bank,evaluation=load_evaluation(evaluation_path)
    else:
        bank,evaluation=prepare_evaluation(args.inputs,args.out/"registration")
    if evaluation["inputs"]["sha256"]!=archive.sha256:
        raise RuntimeError("Evaluation and training source archives differ")
    if evaluation["status"]!="PASS":
        raise RuntimeError("Registered evaluation patches insufficient; no replacement or training")
    contract["evaluation_registration"]=str((args.out/"registration").resolve())
    contract["evaluation_inputs_sha256"]=file_sha256(evaluation_path)
    contract["evaluation_case_registration"]=evaluation
    contract["encoder_definition"]=dict(ENCODER_METADATA)
    contract["generation_seed_history_check"]=dict(status="REGISTERED",
        seeds=list(GENERATION_SEEDS),
        scope="Local native_t1 and experiments source/JSON/Markdown searched with rg; no prior use as generation/model-selection seeds found. Numeric 9702 occurrences were counters.",
        external_history_claim=False)
    condition_checks=[]
    for scope,cases in (("training_archive",archive.cases),("evaluation",bank["cases"])):
        for key,case in sorted(cases.items()):
            phi,adjacency,facts=face_geometry(torch.from_numpy(case.constraints.copy()))
            condition_checks.append(dict(scope=scope,key=list(key),K=case.K,
                minimum_area=float(facts["area"].min()),nonmanifold_edges=int(facts["nonmanifold_edge_count"]),
                adjacency_directed_edges=int(adjacency.sum()),phi_finite=bool(torch.isfinite(phi).all())))
    contract["condition_checks"]=condition_checks
    contract["registration_analytic_geometry_calls"]=len(condition_checks)
    root=Path(__file__).resolve().parents[1]
    source_names=("native_t1/model.py","native_t1/context_geometry.py","native_t1/geometry_checkpoint.py",
        "native_t1/geometry_pilot.py","native_t1/geometry_gates.py","native_t1/geometry_generation.py",
        "native_t1/geometry_accounting.py",
        "native_t1/geometry_evaluation.py","native_t1/data.py","native_t1/training.py",
        "native_t1/sampling.py","native_t1/runtime.py","native_t1/baseline.py",
        "native_t1/portable_checkpoint.py","native_t1/train_cli.py",
        "models/equidit.py","models/attention.py","models/utils.py","utils/ot_utils.py",
        "configs/snet/base-120m-ot-v-chair.yaml")
    contract["execution_sources"]={name:file_sha256(root/name) for name in source_names}
    atomic_json(path,contract)
    print("REGISTERED "+str(path),flush=True)
    return contract


def train_arms(args):
    from .geometry_gates import fixed_batch
    contract=read_json(args.out/"registration.json")
    if read_json(args.out/"gates.json").get("status")!="PASS":
        raise RuntimeError("GPU gates must pass before training")
    if file_sha256(args.inputs)!=contract["inputs_sha256"]:
        raise RuntimeError("Training archive changed after registration")
    if file_sha256(args.checkpoint or BASELINE_CHECKPOINT)!=contract["initialization_file_sha256"]:
        raise RuntimeError("Initialization changed after registration")
    root=Path(__file__).resolve().parents[1]
    if any(file_sha256(root/name)!=digest for name,digest in contract["execution_sources"].items()):
        raise RuntimeError("Execution code changed after registration")
    archive=load_archive(args.inputs)
    reference_inputs=None
    encoder_initial_hash=None
    outcomes={}
    for arm in ARMS:
        directory=args.out/arm
        if directory.exists():
            raise FileExistsError("Training never resumes/retries an existing arm: "+str(directory))
        directory.mkdir()
        counts=dict(forward_attempts=0,forward_returns=0,backward_attempts=0,backward_returns=0,
                    optimizer_step_attempts=0,optimizer_updates=0,OT_calls=0,OT_returns=0)
        report=dict(status="RUNNING",arm=arm,requested_updates=500,completed_updates=0,
            counts=counts,logs=[],checkpoints=[],update_in_progress=False,
            timing=dict(input_OT_seconds=0.,training_step_seconds=0.,checkpoint_seconds=0.,
                        context_cuda_seconds=0.,context_wall_seconds=0.,context_forward_calls=0),
            validation_counts=dict(forward_attempts=0,forward_returns=0,backward_attempts=0,
                                   backward_returns=0,optimizer_updates=0))
        start=time.perf_counter()
        model=None;optimizer=None
        atomic_json(directory/"training.json",report)
        try:
            model,identity=load_initial(args)
            if identity["checkpoint_sha256"]!=contract["initialization_file_sha256"]:
                raise RuntimeError("Common checkpoint identity mismatch")
            report["initialization"]=identity
            report["pure_initial_state_sha256"]=state_sha256(model)
            if arm!="T_base":
                report["context"]=attach_context_encoder(model,arm.split("_",1)[1],seed=ENCODER_SEED)
                own_encoder=state_sha256(model.context_encoder)
                if encoder_initial_hash is None: encoder_initial_hash=own_encoder
                if own_encoder!=encoder_initial_hash: raise RuntimeError("Encoder initializations differ")
                report["encoder_initial_sha256"]=own_encoder
            model.set_trainable(True).eval()
            optimizer=make_optimizer(model)
            report["parameters"]=sum(p.numel() for p in model.parameters())
            report["all_parameters_trainable"]=all(p.requires_grad for p in model.parameters())
            report["optimizer_parameter_count"]=sum(p.numel() for g in optimizer.param_groups for p in g["params"])
            stream=NativeTrainingStream(archive,seed=INPUT_SEED)
            local_inputs=[]
            torch.cuda.reset_peak_memory_stats()
            with (directory/"input_stream.jsonl").open("x",encoding="utf-8") as output:
                for step in range(1,501):
                    report["update_in_progress"]=True
                    tick=time.perf_counter()
                    with count_ot_calls(counts):
                        samples=stream.next_effective_batch()
                    row=dict(stream.last_audit,context_hashes=[s["context_sha256"] for s in samples])
                    if reference_inputs is not None and row!=reference_inputs[step-1]:
                        raise RuntimeError("Target/C/K/Gaussian/OT/t/mask/permutation stream mismatch")
                    local_inputs.append(row)
                    output.write(json.dumps(row,allow_nan=False)+"\n");output.flush()
                    report["timing"]["input_OT_seconds"]+=time.perf_counter()-tick
                    observe=step in (1,10,100,500)
                    observer=None;before=None
                    if model.context_encoder is not None:
                        observer=TimingObserver(diagnostic=observe)
                        model.context_observer=observer
                        if observe:
                            before={k:p.detach().cpu().clone() for k,p in model.context_encoder.named_parameters()}
                    tick=time.perf_counter()
                    metrics=train_step(model,optimizer,samples,counts=counts)
                    report["timing"]["training_step_seconds"]+=time.perf_counter()-tick
                    report["completed_updates"]=step
                    report["update_in_progress"]=False
                    timing=observer.result() if observer is not None else None
                    if timing:
                        report["timing"]["context_cuda_seconds"]+=timing["cuda_event_seconds"]
                        report["timing"]["context_wall_seconds"]+=timing["wall_seconds"]
                        report["timing"]["context_forward_calls"]+=timing["calls"]
                    if observe:
                        log=dict(step=step,**metrics)
                        if observer is not None:
                            log["context_observation"]=timing
                            log["encoder_gradient_update"]=gradient_update_summary(model.context_encoder,before)
                        report["logs"].append(log)
                    model.context_observer=None
                    report["peak_cuda_allocated"]=torch.cuda.max_memory_allocated()
                    if step in (250,500):
                        checkpoint=directory/f"additional_step{step}_cumulative_step{1500+step}.pt"
                        tick=time.perf_counter()
                        if arm=="T_base":
                            saved=save_checkpoint(checkpoint,model,optimizer,stream,archive,DEFAULT_CONFIG,1500,step)
                        else:
                            saved=save_geometry_checkpoint(checkpoint,model,optimizer,stream,archive,
                                                           base_identity=identity,completed=step)
                        report["timing"]["checkpoint_seconds"]+=time.perf_counter()-tick
                        report["checkpoints"].append(saved)
                    if step%25==0 or observe:
                        print(f"{arm} updates={step}/500",flush=True)
                    atomic_json(directory/"training.json",report)
            if reference_inputs is None:
                reference_inputs=local_inputs
            report["input_stream_sha256"]=file_sha256(directory/"input_stream.jsonl")
            report["input_stream_matches_T_base"]=local_inputs==reference_inputs
            report["stream_end"]=stream.state_dict()
            # Save/reload compares the trained model's actual velocity, not file readability.
            model.zero_grad(set_to_none=True);model.set_trainable(False).eval()
            batch=fixed_batch(archive)
            before_reload=recorded_forward(model,batch,report["validation_counts"]).detach().cpu()
            np.save(directory/"reload_velocity_reference.npy",before_reload.numpy())
            del optimizer,model,batch
            optimizer=None;model=None;free_memory()
            endpoint=Path(report["checkpoints"][-1]["path"])
            if arm=="T_base":
                reloaded,load_audit=load_trained_checkpoint(endpoint)
            else:
                reloaded,load_audit=load_geometry_checkpoint(endpoint)
            batch=fixed_batch(archive)
            after_reload=recorded_forward(reloaded,batch,report["validation_counts"]).detach().cpu()
            if not exact_tensor(before_reload,after_reload):
                raise RuntimeError("Saved/reloaded endpoint velocity differs")
            report["reload"]={"status":"PASS","velocity_bitwise":True,"load":load_audit}
            del reloaded,batch,after_reload,before_reload
            free_memory()
            report["status"]="PASS"
            outcomes[arm]=report
        except BaseException as error:
            report.update(status="STOPPED",error=repr(error))
            raise
        finally:
            report["seconds"]=time.perf_counter()-start
            atomic_json(directory/"training.json",report)
            if model is not None: del model
            if optimizer is not None: del optimizer
            free_memory()
    if sum(r["counts"]["optimizer_updates"] for r in outcomes.values())!=1500:
        raise RuntimeError("Total update accounting differs")
    atomic_json(args.out/"training_summary.json",dict(status="PASS",arms=outcomes,total_optimizer_updates=1500))
    return outcomes


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-profile",choices=("a-continue","trained"),default="a-continue")
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--inputs",type=Path,required=True)
    parser.add_argument("--input-seed",type=int,default=1010)
    parser.add_argument("--updates",type=int,default=500)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--stage",choices=("all","register","gates","train","generate","report"),default="all")
    parser.add_argument("--audit-results",type=Path)
    parser.add_argument("--baseline-regression-evidence",type=Path)
    return parser


def main(argv=None):
    args=build_parser().parse_args(argv)
    assert_contract(args)
    if args.stage in ("all","register"):
        register(args)
    if args.stage in ("all","gates"):
        from .geometry_gates import run_gates
        run_gates(args)
    if args.stage in ("all","train"):
        train_arms(args)
    if args.stage in ("all","generate"):
        from .geometry_generation import generate
        generate(args)
    if args.stage in ("all","report"):
        from .geometry_results import write_report
        write_report(args.out,audit_results=args.audit_results)


if __name__=="__main__":
    main()
