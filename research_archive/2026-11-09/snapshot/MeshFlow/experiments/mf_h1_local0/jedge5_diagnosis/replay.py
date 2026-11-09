"""Stage D only: exact historical B/J replay with an independent FP32 EMA.

Importing does not load assets/models. Only run_replay may train and it requires
root's completed A/B/C gate. Each arm is a separate process/writer; no resume or
retry. Checkpoint schema is private and never changes the public default.
"""
from native_t1.runtime import configure_stable_runtime
from pathlib import Path
import argparse, copy, hashlib, json, time, traceback, zipfile
import torch
from native_t1.artifacts import file_sha256, state_sha256, atomic_json

REPO = Path(__file__).resolve().parents[3]
ARMS = ("R_B", "R_J")
HALF_LIFE = 100
BETA = 2.0 ** (-1.0 / HALF_LIFE)
LAMBDA_EDGE = 4.071385484299878
LAMBDA_L5 = 0.3266104383520167
SCHEMA = "jedge5_diagnosis_raw_ema_v1"
INITIAL = REPO / "native_t1/checkpoints/tgeo_g0_cumulative2500.pt"
INITIAL_FILE = "3a66659e86586f54d2ec30ed3d4a18275ab69ba0f8cba2866ef67b340c481c0e"
INITIAL_STATE = "6fa39d1c8c1deade17467075160281dcf01948c45e90ca4a9ad111a7dd97c392"
INPUTS = REPO / "output/t1_publish_20260930/chair_n112.npz"
INPUT_SHA = "7059671d55f2f3be8777d13e02a62255d8a0c0c7c1c1c5b4f3b24357091ba6fc"
START_STREAM_SHA = "ca35c12af30f1b3ec9d6d5e9d191edf0a1dc6c7b61584b6976c0ac6c004c93a7"
PAIRED_STREAM_SHA = "609071dc47dd8805a43f79e4259ed649a39bee5808cb37978c2d939e5981e11c"
ENDPOINTS = {
    "R_B": (REPO / "native_t1/checkpoints/ablations/fm_cumulative3000.pt",
            "0f22ce498ca2c27dc74b43890fb8aad20f0f3e7153aa74ee6d3c14ea445a2660",
            "f006c49c8c70efa4bf2df6ca96431c844a25a1bdd38e2db4756115a37a23545b"),
    "R_J": (REPO / "native_t1/checkpoints/jedge5_cumulative3000.pt",
            "cc7a248fa9a8570059f225cd1c313f21f08585925407a79418df6d63c9cae0e9",
            "b9c1df5326ad4815ec3c32ed20e08dc34ff6248ff05be2c412e051fcb728aae3")}
HISTORIES = {
    "R_B": (REPO / "experiments/mf_h1_local0/outputs/loss_screen_20260930_232200/results.json", "L0"),
    "R_J": (REPO / "native_t1/runs/l3_l5_interface_pilot_20261001_104600/results.json", "JEdge5")}
ZIP = REPO / "native_t1/maintenance/jedge5_cleanup_20261001_115000/retired_loss_campaign_source.zip"
LOG_STEPS = (1, 50, 100, 250, 500)


def seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def inc(counts, key):
    counts[key] = counts.get(key, 0) + 1


def hash_state(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def replay_recipe(arm):
    if arm not in ARMS: raise ValueError("Only R_B and R_J")
    return dict(arm=arm, loss="FM" if arm == "R_B" else "FM+ramp*(lambda_edge*D_edge+lambda5*L5)/8",
                lambda_edge=LAMBDA_EDGE if arm == "R_J" else 0., lambda5=LAMBDA_L5 if arm == "R_J" else 0.,
                gate="original_FP32_t>=0.5", ramp="min(additional_step/50,1)",
                effective_batch=8, microbatch=1, fresh_AdamW=True, lr=1e-5, betas=[.9,.95], weight_decay=0., clip=1.,
                autocast_scope="independent native_execution per microbatch, backward inside; optimizer outside all scopes")


def ema_spec():
    return dict(half_life_updates=HALF_LIFE, beta=BETA, initialize="clone G0_2500 parameters",
                parameter_dtype="torch.float32", buffers="copy raw without averaging",
                math="shadow.mul_(beta).add_(raw.detach(), alpha=1-beta)",
                update_order="after successful optimizer.step and finite-parameter check",
                optimizer_or_forward_uses_EMA=False, initial_coefficient_after500=BETA**500)


class FP32EMA:
    """Independent shadow of every trainable parameter and every raw buffer.

    Production uses the raw model's CUDA device, without a second model. The
    CPU fixture uses the same two eager FP32 operations. No RNG is consumed.
    """
    def __init__(self, model):
        self.parameters = {}
        self.buffers = {}
        self.updates = 0
        for name, value in model.named_parameters():
            if value.dtype != torch.float32 or not value.requires_grad:
                raise ValueError("EMA requires every raw model parameter trainable FP32")
            self.parameters[name] = value.detach().clone()
        if not self.parameters: raise ValueError("Empty EMA")
        for name, value in model.named_buffers(): self.buffers[name] = value.detach().clone()
        self.state_keys = tuple(model.state_dict())
        self.device = str(next(iter(self.parameters.values())).device)

    @torch.no_grad()
    def update(self, model, completed_updates):
        if type(completed_updates) is not int or completed_updates != self.updates + 1:
            raise ValueError("EMA requires exactly one update per successful optimizer step")
        current = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        if current.keys() != self.parameters.keys() or buffers.keys() != self.buffers.keys():
            raise ValueError("EMA tensor membership changed")
        for name, raw in current.items():
            shadow = self.parameters[name]
            if raw.dtype != torch.float32 or raw.shape != shadow.shape or raw.device != shadow.device or not raw.requires_grad:
                raise ValueError("EMA parameter identity/dtype/device changed: " + name)
            shadow.mul_(BETA).add_(raw.detach(), alpha=1-BETA)
        for name, raw in buffers.items():
            shadow = self.buffers[name]
            if raw.shape != shadow.shape or raw.dtype != shadow.dtype or raw.device != shadow.device:
                raise ValueError("EMA buffer identity changed: " + name)
            shadow.copy_(raw.detach())
        self.updates = completed_updates

    def state_dict(self):
        all_values = dict(self.parameters, **self.buffers)
        return {name: all_values[name] for name in self.state_keys}


def audit_batch(stream, samples, step, labels):
    # Exact historical JSON layout. Shared hashes cover Z/OT/Y/face and corner maps.
    return dict(step=step, batch_index=stream.batch_index, sample_index=stream.sample_index,
                shared=[s["shared_input_sha256"] for s in samples], C=[s["context_sha256"] for s in samples],
                target_labels=[v["mapping_sha256"] for v in labels], parents=[s["parent_index"] for s in samples],
                K=[s["K"] for s in samples], t=[float(s["t"]) for s in samples], stream_sha256=seal(stream.state_dict()))


def load_history():
    records = {}; reference = None; start = None
    for arm, (path, name) in HISTORIES.items():
        record = json.loads(path.read_text(encoding="utf-8")); entry = record["arms"][name]
        rows = entry["stream_audit"]
        if entry["status"] != "PASS" or entry["completed_updates"] != 500 or len(rows) != 500 or seal(rows) != PAIRED_STREAM_SHA:
            raise ValueError("Historical 500-batch replay asset differs: " + arm)
        if seal(record["stream_start"]) != START_STREAM_SHA:
            raise ValueError("Historical starting stream differs")
        if reference is not None and (reference != rows or start != record["stream_start"]):
            raise ValueError("Historical B and J input streams differ")
        reference, start = rows, record["stream_start"]
        records[arm] = dict(path=str(path), sha256=file_sha256(path), logs=entry["logs"],
                            final_stream_state=entry["final_stream_state"])
    if (start["seed"], start["batch_index"], start["sample_index"]) != (1010,1000,8000):
        raise ValueError("Wrong replay segment")
    return copy.deepcopy(start), copy.deepcopy(reference), records



def audit_historical_sources():
    """Read-only old/new algorithm identity. No tensor/model/OT evaluation."""
    import ast
    record=json.loads(HISTORIES["R_J"][0].read_text(encoding="utf-8"))
    rows=[]
    unchanged=("native_t1/data.py","native_t1/model.py","native_t1/context_geometry.py",
               "native_t1/runtime.py","models/equidit.py","models/attention.py")
    for rel in unchanged:
        saved=next(x for x in record["source_snapshot"]["files"] if x["original_path"].endswith("/"+rel))
        old=Path(saved["snapshot_path"]);current=REPO/rel
        a,b=file_sha256(old),file_sha256(current)
        if a!=saved["sha256"] or a!=b:raise RuntimeError("Historical raw math/source changed: "+rel)
        rows.append(dict(path=rel,archived_path=str(old),sha256=b,check="exact_file_bytes"))
    saved=next(x for x in record["source_snapshot"]["files"] if x["original_path"].endswith("/native_t1/training.py"))
    old=Path(saved["snapshot_path"])
    if file_sha256(old)!=saved["sha256"]:raise RuntimeError("Historical FM/AdamW reference changed")
    def functions(source):return {f.name:f for f in ast.parse(source).body if isinstance(f,ast.FunctionDef)}
    before,after=functions(old.read_text()),functions((REPO/"native_t1/training.py").read_text())
    for name in ("make_optimizer","masked_fm_loss"):
        if ast.dump(before[name],include_attributes=False)!=ast.dump(after[name],include_attributes=False):
            raise RuntimeError("Historical training function changed: "+name)
        rows.append(dict(path="native_t1/training.py",function=name,check="exact_function_AST"))
    with zipfile.ZipFile(ZIP) as archive:
        originals={name:archive.read("experiments/mf_h1_local0/"+name).decode("utf-8") for name in
                   ("l3_l5_interface_pilot/edge_loss.py","l3_l5_interface_pilot/training.py","loss_screen/training.py","loss_screen/distribution_losses.py")}
    for rel,names in (("l3_l5_interface_pilot/edge_loss.py",("edge_loss","centered_segment_distance_sq")),
                      ("loss_screen/distribution_losses.py",("_geometry","_quadrature","_squared_distances","_kernel_values","_weighted_kernel_values"))):
        target="edge.py" if rel.startswith("l3_") else "surface.py"
        a=functions(originals[rel]);b=functions((REPO/"native_t1/losses"/target).read_text())
        for name in names:
            if ast.dump(a[name],include_attributes=False)!=ast.dump(b[name],include_attributes=False):raise RuntimeError("Loss arithmetic changed: "+name)
            rows.append(dict(path="native_t1/losses/"+target,function=name,check="exact_function_AST"))
    return dict(status="PASS",checks=rows,archive=dict(path=str(ZIP),sha256=file_sha256(ZIP)),
                trainer_sources={k:hashlib.sha256(v.encode()).hexdigest() for k,v in originals.items() if k.endswith("training.py")},
                arithmetic_notes=["Public train_step changed to optional recipes; this module uses its own archived per-micro training loop.",
                                  "Public L5 removes only the unused L4 branch and detached GT probability preparation; tiny original-vs-public FP32/FP64 values and gradients are checked separately.",
                                  "Historical diagnostic-only detached velocity gradients, L0 probes and parameter snapshots do not enter raw gradients or RNG and are not repeated.",
                                  "Each successful raw update is followed by independent FP32 EMA; no cached BF16 weights survive an optimizer boundary."],
                formal_replay="NOT_RUN; source/CPU parity does not replace final raw500 hash equality")


def read_stage_gate(out):
    path = Path(out) / "gates/ABC_COMPLETE.json"
    if not path.is_file(): raise RuntimeError("Stage D blocked: A/B/C completion gate is absent")
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS": raise RuntimeError("Stage D blocked: A/B/C gate did not pass")
    return dict(path=str(path.resolve()), sha256=file_sha256(path), status="PASS")


def validate_payload(payload, *, expected_arm=None, expected_kind=None):
    from native_t1.portable_checkpoint import NATIVE_META
    from native_t1.context_geometry import ENCODER_METADATA
    if payload.get("schema") != SCHEMA or payload.get("native_meta") != NATIVE_META:
        raise ValueError("Wrong diagnosis checkpoint schema/Native metadata")
    arm, kind, step = payload.get("arm"), payload.get("kind"), payload.get("completed_updates")
    if arm not in ARMS or kind not in ("raw", "EMA") or type(step) is not int or step not in (250,500):
        raise ValueError("Invalid fixed raw/EMA endpoint")
    if expected_arm is not None and arm != expected_arm: raise ValueError("Wrong arm")
    if expected_kind is not None and kind != expected_kind: raise ValueError("Wrong raw/EMA identity")
    if payload.get("recipe") != replay_recipe(arm) or payload.get("ema_protocol") != ema_spec():
        raise ValueError("Recipe/EMA protocol differs")
    if payload.get("base_cumulative_updates") != 2500 or payload.get("cumulative_updates") != 2500+step:
        raise ValueError("Wrong cumulative update identity")
    if payload.get("initial_state_sha256") != INITIAL_STATE or payload.get("input_sha256") != INPUT_SHA:
        raise ValueError("Wrong replay assets")
    if payload.get("initial_stream_sha256") != START_STREAM_SHA or payload.get("expected500_stream_sha256") != PAIRED_STREAM_SHA:
        raise ValueError("Wrong replay stream contract")
    if payload.get("ema_updates") != step or payload.get("optimizer_restored") is not False:
        raise ValueError("EMA updates or optimizer identity differs")
    stream = payload.get("stream_state", {})
    if (stream.get("seed"),stream.get("batch_index"),stream.get("sample_index")) != (1010,1000+step,8000+step*8):
        raise ValueError("Wrong stream progress")
    if seal(stream) != payload.get("stream_state_sha256"): raise ValueError("Stream hash differs")
    context = payload.get("context", {})
    if context.get("context_mode") != "geo" or context.get("additional_parameters") != 149888:
        raise ValueError("Missing complete Geo encoder")
    if any(context.get(k) != v for k,v in ENCODER_METADATA.items()): raise ValueError("Changed Geo metadata")
    state = payload.get("model")
    if not isinstance(state, dict) or not state or not any(k.startswith("context_encoder.") for k in state) or "role_embedding" not in state:
        raise ValueError("Incomplete generator state")
    if any(not isinstance(v,torch.Tensor) or v.device.type != "cpu" or (v.is_floating_point() and v.dtype != torch.float32) for v in state.values()):
        raise ValueError("Checkpoint model must contain CPU FP32 floating tensors")
    if any(not bool(torch.isfinite(v).all()) for v in state.values()): raise ValueError("Nonfinite state")
    if hash_state(state) != payload.get("state_sha256"): raise ValueError("State hash differs")
    return payload


def save_checkpoint(path, model, ema, stream, arm, step, kind, config_path):
    from native_t1.portable_checkpoint import NATIVE_META, config_sha256
    path=Path(path)
    if path.exists() or path.with_suffix(".partial").exists(): raise FileExistsError(path)
    state = model.state_dict() if kind == "raw" else ema.state_dict()
    state = {k:v.detach().cpu().clone() for k,v in state.items()}
    payload=dict(schema=SCHEMA, arm=arm, kind=kind, model=state, state_sha256=hash_state(state),
                 native_meta=dict(NATIVE_META), context=model.context_encoder.metadata(), config_sha256=config_sha256(config_path),
                 recipe=replay_recipe(arm), ema_protocol=ema_spec(), ema_updates=ema.updates, ema_device=ema.device,
                 initial_state_sha256=INITIAL_STATE, initial_stream_sha256=START_STREAM_SHA,
                 expected500_stream_sha256=PAIRED_STREAM_SHA, input_sha256=INPUT_SHA,
                 base_cumulative_updates=2500, completed_updates=step, cumulative_updates=2500+step,
                 stream_state=stream.state_dict(), stream_state_sha256=seal(stream.state_dict()),
                 optimizer_restored=False, optimizer_state_saved=False,
                 fixed_raw500_reference=ENDPOINTS[arm][2],
                 replay_verdict="See per-arm replay.json; EMA evaluation requires final exact raw replay")
    validate_payload(payload,expected_arm=arm,expected_kind=kind)
    temporary=path.with_suffix(".partial");torch.save(payload,temporary);temporary.replace(path)
    return dict(path=str(path.resolve()),sha256=file_sha256(path),bytes=path.stat().st_size,
                state_sha256=payload["state_sha256"],kind=kind,arm=arm,completed_updates=step,cumulative_updates=2500+step,
                ema_updates=ema.updates,schema=SCHEMA,strict_payload_validation=True)


def load_replay_checkpoint(path, *, device="cuda", expected_arm=None, expected_kind=None, expected_sha256=None):
    from native_t1.checkpoint import DEFAULT_CONFIG
    from native_t1.portable_checkpoint import _read_config, config_sha256
    from native_t1.model import NativeInpaintingModel
    from native_t1.context_geometry import ContextGeometryEncoder
    from models.equidit import DiT
    from models.utils import get_embedder
    path=Path(path);digest=file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256: raise ValueError("Checkpoint file hash differs")
    payload=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    validate_payload(payload,expected_arm=expected_arm,expected_kind=expected_kind)
    if payload["config_sha256"] != config_sha256(DEFAULT_CONFIG): raise ValueError("Model config differs")
    config=_read_config(DEFAULT_CONFIG);configure_stable_runtime()
    with torch.device("meta"):model=NativeInpaintingModel(DiT(**config),trainable=False)
    model.context_encoder=ContextGeometryEncoder("geo",seed=payload["context"]["initialization_seed"])
    model.backbone.x_embedder.embed_fn,_=get_embedder(config["pe_freq"],input_dims=3)
    keys=model.load_state_dict(payload["model"],strict=True,assign=True)
    model.set_trainable(False).eval()
    if state_sha256(model) != payload["state_sha256"]:raise RuntimeError("Strict full-state restoration differs")
    audit=dict(path=str(path.resolve()),sha256=digest,state_sha256=payload["state_sha256"],arm=payload["arm"],kind=payload["kind"],
               completed_updates=payload["completed_updates"],cumulative_updates=payload["cumulative_updates"],
               strict=True,missing_keys=list(keys.missing_keys),unexpected_keys=list(keys.unexpected_keys),
               parameters=sum(p.numel() for p in model.parameters()),ema_protocol=payload["ema_protocol"],schema=SCHEMA)
    model.to(device)
    return model,audit


def first_state_difference(actual, expected):
    if actual.keys() != expected.keys():return dict(status="KEY_MISMATCH",missing=sorted(expected.keys()-actual.keys()),unexpected=sorted(actual.keys()-expected.keys()))
    for key in sorted(actual):
        a,b=actual[key].detach().cpu().contiguous(),expected[key].detach().cpu().contiguous()
        if a.shape != b.shape or a.dtype != b.dtype:return dict(status="SHAPE_OR_DTYPE",tensor=key)
        if not torch.equal(a.view(torch.uint8),b.view(torch.uint8)):
            d=a.double()-b.double()
            return dict(status="TENSOR_DIFFERENCE",tensor=key,shape=list(a.shape),max_abs=float(d.abs().max()),RMS=float(d.square().mean().sqrt()),
                        differing_bytes=int((a.view(torch.uint8)!=b.view(torch.uint8)).sum()))
    return dict(status="BITWISE_EQUAL")


def _plain(value):
    if torch.is_tensor(value):return value.detach().item() if value.numel()==1 else value.detach().cpu().tolist()
    if isinstance(value,dict):return {k:_plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_plain(v) for v in value]
    return value


def _component(name,pred,target,counts):
    from native_t1.losses import evaluate
    inc(counts,name+"_calls_attempted")
    with torch.autocast(device_type=pred.device.type,enabled=False):result=evaluate(name,pred,target)
    if result["loss"].ndim or not bool(torch.isfinite(result["loss"])):raise FloatingPointError("Nonfinite "+name)
    inc(counts,name+"_calls_returned")
    for key,value in _plain(result.get("counts",{})).items():
        if isinstance(value,(int,float)) and not isinstance(value,bool):counts[name+"_"+key]=counts.get(name+"_"+key,0)+value
    return result


def run_replay(out, arm):
    """Root-dispatched single arm. Failed attempts never automatically resume."""
    from native_t1.geometry_checkpoint import load_geometry_checkpoint
    from native_t1.checkpoint import DEFAULT_CONFIG
    from native_t1.data import load_archive, NativeTrainingStream, native_collate, model_inputs
    from native_t1.losses import TrainingGeometryTargets, prepare_targets, work_counters
    from native_t1.training import make_optimizer, masked_fm_loss
    from native_t1.model import native_execution
    from native_t1.accounting import count_ot_calls
    replay_recipe(arm)
    out=Path(out).resolve();gate=read_stage_gate(out)
    root=out/"stage_d"/arm
    root.mkdir(parents=True,exist_ok=False)
    c={};r=dict(status="PREPARING",arm=arm,recipe=replay_recipe(arm),EMA=ema_spec(),gate=gate,
               completed_updates=0,checkpoints={},stream_audit=[],logs=[],memory_by_update=[],counts=c,
               timing=dict(CPU_OT_and_stream_seconds=0.,supervision_prepare_seconds=0.,model_update_seconds=0.,
                           EMA_update_seconds=0.,explicit_sync_seconds=0.,checkpoint_seconds=0.),
               replay_reference_state_sha256=ENDPOINTS[arm][2],first_observed_training_scalar_mismatch=None)
    def write():atomic_json(root/"replay.json",r)
    model=optimizer=ema=None;start=time.perf_counter();write()
    try:
        r["historical_source_audit"]=audit_historical_sources()
        stream_start,old_audits,history=load_history();r["history"]={k:{x:v[x] for x in ("path","sha256")} for k,v in history.items()}
        for path,digest in ((INITIAL,INITIAL_FILE),(INPUTS,INPUT_SHA),ENDPOINTS[arm][:2]):
            if file_sha256(path)!=digest:raise RuntimeError("Protected replay asset differs: "+str(path))
        archive=load_archive(INPUTS,INPUT_SHA);bank=TrainingGeometryTargets(archive)
        stream=NativeTrainingStream(archive,seed=1010);stream.load_state_dict(copy.deepcopy(stream_start))
        if seal(stream.state_dict()) != START_STREAM_SHA:raise RuntimeError("Wrong start stream")
        model,identity=load_geometry_checkpoint(INITIAL,device="cuda",expected_recipe="fm")
        if identity["state_sha256"]!=INITIAL_STATE or identity["cumulative_updates"]!=2500 or identity["parameters"]!=130401155:raise RuntimeError("Wrong G0 generator")
        model.set_trainable(True).eval();optimizer=make_optimizer(model)
        if optimizer.state:raise RuntimeError("Optimizer is not fresh")
        ema=FP32EMA(model);r.update(status="TRAINING",initial_identity=identity,initial_stream_sha256=seal(stream.state_dict()),
                                   EMA_device=ema.device,parameters=sum(p.numel() for p in model.parameters()),
                                   initial_EMA_state_sha256=hash_state(ema.state_dict()))
        if r["initial_EMA_state_sha256"]!=INITIAL_STATE:raise RuntimeError("EMA initialization differs")
        old_logs={x["step"]:x for x in history[arm]["logs"]};write()
        for step in range(1,501):
            if c.get("optimizer_step_attempts",0)>=500:raise RuntimeError("Fixed per-arm budget exhausted")
            torch.cuda.reset_peak_memory_stats()
            tick=time.perf_counter()
            with count_ot_calls(c):samples=stream.next_effective_batch()
            r["timing"]["CPU_OT_and_stream_seconds"]+=time.perf_counter()-tick
            tick=time.perf_counter();labels=[bank.sample(s) for s in samples];audit=audit_batch(stream,samples,step,labels)
            if audit!=old_audits[step-1]:
                r["first_input_mismatch"]=dict(step=step,actual=audit,expected=old_audits[step-1]);raise RuntimeError("Historical input stream differs before update")
            r["stream_audit"].append(audit)
            prepared=[prepare_targets(label) for label in labels] if arm=="R_J" else None
            r["timing"]["supervision_prepare_seconds"]+=time.perf_counter()-tick
            denom=9*sum(int((s["valid_mask"]&~s["known_mask"]).sum()) for s in samples)
            if denom not in (7524,7452,7488):raise RuntimeError("FM denominator changed")
            optimizer.zero_grad(set_to_none=True);fm_total=0.;edge_total=0.;surface_total=0.;active_count=0
            ramp=min(step/50.,1.);r["batch_in_progress"]=step;tick=time.perf_counter()
            for micro,s in enumerate(samples):
                b=native_collate([s],"cuda");active=bool(b["t"][0]>=.5);active_count+=int(active)
                # Same lifetime as both archived training.py files. No outer autocast.
                with native_execution(model,coordinates=b["xt"]) as runtime:
                    inc(c,"forward_attempts");v=model(*model_inputs(b)).float();inc(c,"forward_returns")
                    fm=masked_fm_loss(v,b["u"],b["valid_mask"],b["known_mask"],denominator=denom);total=fm
                    if arm=="R_J" and active:
                        pred=(b["xt"]+(1-b["t"][:,None,None])*v)[b["loss_mask"]].reshape(-1,3,3)
                        a=_component("D_edge",pred,prepared[micro]["D_edge"],c)
                        d=_component("L5",pred,prepared[micro]["L5"],c)
                        total=total+ramp*(LAMBDA_EDGE*a["loss"]+LAMBDA_L5*d["loss"])/8
                        edge_total+=float(a["loss"].detach())/8;surface_total+=float(d["loss"].detach())/8
                    if not bool(torch.isfinite(total)):raise FloatingPointError("Nonfinite historical objective")
                    inc(c,"backward_attempts");total.backward();inc(c,"backward_returns")
                fm_total+=float(fm.detach())
            norm=float(torch.nn.utils.clip_grad_norm_(list(model.parameters()),1.,error_if_nonfinite=True))
            r["optimizer_step_in_progress"]=True;inc(c,"optimizer_step_attempts");optimizer.step();inc(c,"optimizer_updates")
            r["completed_updates"]=step;r["optimizer_step_in_progress"]=False;r["batch_in_progress"]=False
            sync=time.perf_counter();torch.cuda.synchronize();r["timing"]["explicit_sync_seconds"]+=time.perf_counter()-sync
            if not bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()):raise FloatingPointError("Nonfinite raw parameter after update")
            r["timing"]["model_update_seconds"]+=time.perf_counter()-tick
            tick=time.perf_counter();inc(c,"EMA_update_attempts");ema.update(model,step);torch.cuda.synchronize();inc(c,"EMA_updates")
            r["timing"]["EMA_update_seconds"]+=time.perf_counter()-tick
            r["memory_by_update"].append(dict(step=step,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                             scope="one full effective update including raw optimizer, shadow EMA and retained tensors; excludes checkpoint save"))
            if step in LOG_STEPS:
                weighted=ramp*(LAMBDA_EDGE*edge_total+LAMBDA_L5*surface_total)
                row=dict(step=step,cumulative_updates=2500+step,FM=fm_total,D_edge=edge_total,L5=surface_total,weighted_geometry=weighted,
                         total=fm_total+weighted,grad_norm_before_clip=norm,active_samples=active_count,free_coordinate_denominator=denom,runtime=runtime)
                old=old_logs[step];comparisons={"FM":(row["FM"],old["FM"]),"grad_norm_before_clip":(norm,old["grad_norm_before_clip"])}
                if arm=="R_J":comparisons["weighted_geometry"]=(weighted,old["weighted_geometry"])
                row["historical_scalar_exact"]={k:a==b for k,(a,b) in comparisons.items()}
                if r["first_observed_training_scalar_mismatch"] is None and not all(row["historical_scalar_exact"].values()):
                    r["first_observed_training_scalar_mismatch"]=dict(step=step,values={k:{"actual":a,"historical":b} for k,(a,b) in comparisons.items() if a!=b})
                r["logs"].append(row)
            if step in (250,500):
                tick=time.perf_counter();before_stream=seal(stream.state_dict())
                r["checkpoints"][str(step)]={kind:save_checkpoint(root/f"{kind}_additional{step}_cumulative{2500+step}.pt",model,ema,stream,arm,step,kind,DEFAULT_CONFIG) for kind in ("raw","EMA")}
                if before_stream!=seal(stream.state_dict()):raise RuntimeError("Checkpoint advanced RNG")
                r["timing"]["checkpoint_seconds"]+=time.perf_counter()-tick
            if step%25==0 or step in LOG_STEPS:
                write();print(json.dumps(dict(event="replay_progress",arm=arm,updates=step,EMA_updates=ema.updates)),flush=True)
        r["paired_stream_sha256"]=seal(r["stream_audit"]);r["final_stream_state"]=stream.state_dict()
        if r["paired_stream_sha256"]!=PAIRED_STREAM_SHA or stream.state_dict()!=history[arm]["final_stream_state"]:raise RuntimeError("Final stream mismatch")
        if any(c.get(k)!=v for k,v in {"optimizer_updates":500,"forward_returns":4000,"backward_returns":4000,"OT_returns":4000,"EMA_updates":500}.items()):raise RuntimeError("Completed execution counts differ")
        raw_hash=state_sha256(model);r["raw500_state_sha256"]=raw_hash;r["raw500_bitwise_reference_equal"]=raw_hash==ENDPOINTS[arm][2]
        r["status"]="PASS_EXACT_REPLAY" if r["raw500_bitwise_reference_equal"] else "REPLAY_MISMATCH"
        if not r["raw500_bitwise_reference_equal"]:
            old=torch.load(ENDPOINTS[arm][0],map_location="cpu",weights_only=True,mmap=True)
            r["first_final_state_difference"]=first_state_difference(model.state_dict(),old["model"])
            del old
        r["EMA_evaluation_allowed"]=r["raw500_bitwise_reference_equal"]
    except Exception as exc:
        r.update(status="STOPPED",error=repr(exc),traceback=traceback.format_exc(),EMA_evaluation_allowed=False)
    finally:
        r["timing"]["wall_seconds"]=time.perf_counter()-start
        r["timing"]["note"]="Buckets are nested: explicit synchronization is included in model/EMA time, checkpoint in wall. CPU_OT wraps stream.next_effective_batch including OT. Not isolated GPU kernel time."
        r["surface_work_counters"]=work_counters();write()
    return r


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument("--out",required=True);parser.add_argument("--arm",choices=ARMS,required=True)
    args=parser.parse_args(argv);result=run_replay(args.out,args.arm)
    if result["status"]!="PASS_EXACT_REPLAY":raise SystemExit(2)

if __name__ == "__main__": main()
