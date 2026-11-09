"""Read-only source-checkpoint review. No forward, CUDA, optimizer or checkpoint write."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
from pathlib import Path
import hashlib, json, time
from native_t1 import runtime
import torch
from experiments.mf_h1_local0.chair_hybrid_coupling_v1.checkpoint import build_model
from experiments.mf_h1_local0.chair_domain_control_v1.checkpoint import tensor_state_hash
torch.set_num_threads(2)
ROOT=Path("/mnt/e/MeshFlow")
OUT=ROOT/"native_t1/runs/chair_capped500_half_v1/audit"
REG=ROOT/"native_t1/runs/chair_capped500_gradbudget_v1/registration/checkpoints.json"
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(4<<20),b""):h.update(b)
    return h.hexdigest()
def write(p,obj):
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
tick=time.perf_counter()
reg=json.loads(REG.read_text())
source={}
payload={}
for label in ("START","HIST_CAPPED500"):
    p=Path(reg[label]["path"])
    actual_sha=sha(p)
    assert actual_sha==reg[label]["checkpoint_sha256"]
    q=torch.load(p,map_location="cpu",weights_only=True,mmap=True)
    assert q["update_in_progress"] is False
    actual_model_sha=tensor_state_hash(q["model"])
    assert actual_model_sha==q["metadata"]["model_state_sha256"]
    if p.with_suffix(".json").exists():
        side=json.loads(p.with_suffix(".json").read_text())
        assert side["checkpoint_sha256"]==actual_sha and side["metadata"]==q["metadata"]
    source[label]={"path":str(p),"bytes":p.stat().st_size,"file_sha256":actual_sha,
      "schema":q["schema"],"top_level_keys":sorted(q),
      "metadata":{k:q["metadata"].get(k) for k in ("arm","context_encoder","config_sha256",
       "conditional_total_step","pilot_step","feedback_step","model_state_sha256","source_checkpoint_sha256")},
      "model_state_sha256_actually_verified":actual_model_sha}
    payload[label]=q
s=payload["START"]["model"];c=payload["HIST_CAPPED500"]["model"]
assert list(s)==list(c)
assert source["START"]["metadata"]["config_sha256"]==source["HIST_CAPPED500"]["metadata"]["config_sha256"]
cpu_rng=torch.random.get_rng_state().clone()
model=build_model(payload["START"],"cpu",False)
assert not torch.cuda.is_initialized()
params=dict(model.named_parameters())
buffers=dict(model.named_buffers())
state=model.state_dict()
assert set(state)==set(s)
assert set(s)==set(params)|{k for k in buffers if k in state}
rows=[]; unequal=0; unchanged=[]; noalias=True; sums=[0.,0.,0.]; maxabs=0.; formula_difference=0
halfhash=hashlib.sha256()
for key,a in s.items():
    b=c[key]
    assert a.shape==b.shape and a.dtype==b.dtype and a.device.type==b.device.type=="cpu"
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    equal=torch.equal(a.view(torch.uint8),b.view(torch.uint8))
    role="parameter" if key in params else "persistent_buffer"
    if role=="persistent_buffer":assert equal
    if role=="parameter":
        assert a.dtype==torch.float32
        half=(a+b)*0.5
        alternative=a+0.5*(b-a)
        formula_difference+=int((half.view(torch.uint8)!=alternative.view(torch.uint8)).any())
        assert half.dtype==torch.float32 and torch.isfinite(half).all()
        noalias=noalias and half.data_ptr()!=a.data_ptr() and half.data_ptr()!=b.data_ptr()
        diff=b.double()-a.double()
        sums[0]+=float(a.double().square().sum());sums[1]+=float(b.double().square().sum());sums[2]+=float(diff.square().sum())
        maxabs=max(maxabs,float(diff.abs().max()))
        unequal+=not equal
        if equal:unchanged.append(key)
    else:half=a.clone()
    halfhash.update(key.encode());halfhash.update(str((tuple(half.shape),half.dtype)).encode())
    halfhash.update(half.contiguous().view(torch.uint8).numpy().tobytes())
    rows.append({"name":key,"role":role,"shape":list(a.shape),"dtype":str(a.dtype),
      "numel":a.numel(),"source_tensors_bitwise_equal":equal})
# Hash uses sorted order, matching tensor_state_hash.
h=hashlib.sha256()
for key in sorted(s):
    v=(s[key]+c[key])*0.5 if key in params else s[key].clone()
    h.update(key.encode());h.update(str((tuple(v.shape),v.dtype)).encode())
    h.update(v.contiguous().view(torch.uint8).numpy().tobytes())
assert all(not p.requires_grad for p in model.parameters()) and all(not m.training for m in model.modules())
assert tensor_state_hash(s)==source["START"]["metadata"]["model_state_sha256"]
assert tensor_state_hash(c)==source["HIST_CAPPED500"]["metadata"]["model_state_sha256"]
sources=[
"experiments/mf_h1_local0/chair_hybrid_coupling_v1/checkpoint.py",
"experiments/mf_h1_local0/chair_control_feedback_auto_v1/checkpoint.py",
"experiments/mf_h1_local0/control_mvp48/model_mvp.py",
"native_t1/model.py","native_t1/context_geometry.py","models/equidit.py","models/utils.py",
"models/attention.py","configs/snet/base-120m-ot-v-chair.yaml"]
OUT.mkdir(parents=True,exist_ok=True)
write(OUT/"interpolation_tensor_inventory.json",rows)
receipt={"status":"PASS","scope":"Independent CPU-only schema/state/strict-load review; no blend checkpoint written",
"sources":source,"key_order_identical":True,"parameter_tensors":len(params),
"parameter_elements":sum(p.numel() for p in params.values()),"buffers":[{"name":k,"dtype":str(v.dtype),"shape":list(v.shape),"persistent":k in state} for k,v in buffers.items()],
"persistent_buffer_count":sum(k in state for k in buffers),"all_state_tensors_FP32":all(x.dtype==torch.float32 for x in s.values()),
"all_source_and_half_tensors_finite":True,"changed_parameter_tensors":unequal,"unchanged_parameter_names":unchanged,
"start_parameter_l2":sums[0]**.5,"capped_parameter_l2":sums[1]**.5,"source_delta_l2":sums[2]**.5,
"source_delta_max_abs":maxabs,"half_formula":"FP32 (START[name] + HIST_CAPPED500[name]) * 0.5, out-of-place, no autocast",
"half_model_state_sha256_for_stated_formula":h.hexdigest(),
"alternative_start_plus_half_delta_bitwise_different_tensor_count":formula_difference,
"out_of_place_no_source_alias":noalias,"source_model_state_hashes_unchanged":True,
"strict_cpu_model_load":"PASS","all_parameters_frozen":True,"all_modules_eval":True,
"constructor_preserved_cpu_rng":torch.equal(cpu_rng,torch.random.get_rng_state()),
"cuda_initialized":torch.cuda.is_initialized(),"model_forward_calls":0,"model_backward_calls":0,"optimizer_updates":0,
"checkpoint_payload_cpu_loads":2,"source_file_hashes_actually_recomputed":True,
"recommended_schema":"chair_capped500_half_inference_v1",
"recommended_payload_keys":["schema","model","metadata"],
"recommended_metadata":["inference_only=true","source file and model SHA256 for both endpoints","alpha=.5","exact arithmetic expression and FP32 dtype","model_state_sha256","architecture config SHA256","context_encoder=geo","new training updates=0"],
"loading":"Validate new payload identity and finite tensors; original.build_model(payload,device,False) uses strict=True assign=True then eval. Do not use old checkpoint.load/validate on a synthetic mixed payload.",
"buffer_policy":"All registered buffers must be byte-identical and copied, never averaged. No integer/bool state averaging. For this instantiated topology actual named_buffers is recorded above. Non-state positional embedder closures are rebuilt by original build_model/get_embedder from frozen pe_freq config.",
"limits":["Weight interpolation is not output or velocity interpolation.","This test supplies no inference quality or Euler validation.","No optimizer/RNG/stream state can be inferred for the mixed model; do not label it a resumable training checkpoint."],
"source_code_sha256":{p:sha(ROOT/p) for p in sources},
"inventory_path":str(OUT/"interpolation_tensor_inventory.json"),"inventory_sha256":sha(OUT/"interpolation_tensor_inventory.json"),
"review_script_sha256":sha(Path(__file__)),"seconds":time.perf_counter()-tick}
write(OUT/"interpolation_review.json",receipt)
print(json.dumps({k:receipt[k] for k in ("status","parameter_tensors","parameter_elements","buffers","changed_parameter_tensors","half_model_state_sha256_for_stated_formula","alternative_start_plus_half_delta_bitwise_different_tensor_count","constructor_preserved_cpu_rng","seconds")}))
