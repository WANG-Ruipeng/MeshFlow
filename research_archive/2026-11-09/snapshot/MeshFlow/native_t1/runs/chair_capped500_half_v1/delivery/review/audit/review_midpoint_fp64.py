"""Independent review of the root-selected FP64 midpoint; no model construction or forward."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OMP_NUM_THREADS"]="2"
os.environ["MKL_NUM_THREADS"]="2"
from pathlib import Path
import hashlib,json,time
import torch
torch.set_num_threads(2)
ROOT=Path("/mnt/e/MeshFlow")
OUT=ROOT/"native_t1/runs/chair_capped500_half_v1/audit"
prior=json.loads((OUT/"interpolation_review.json").read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4<<20),b""):h.update(b)
 return h.hexdigest()
def state_update(h,k,v):
 h.update(k.encode());h.update(str((tuple(v.shape),v.dtype)).encode());h.update(v.contiguous().view(torch.uint8).numpy().tobytes())
tick=time.perf_counter()
states={}
for label in ("START","HIST_CAPPED500"):
 p=Path(prior["sources"][label]["path"])
 assert sha(p)==prior["sources"][label]["file_sha256"]
 states[label]=torch.load(p,map_location="cpu",weights_only=True,mmap=True)["model"]
a,b=states["START"],states["HIST_CAPPED500"]
assert set(a)==set(b)
param_names={r["name"] for r in json.loads((OUT/"interpolation_tensor_inventory.json").read_text()) if r["role"]=="parameter"}
assert len(param_names)==194 and set(a)==param_names
h=hashlib.sha256();hfp32=hashlib.sha256();ha=hashlib.sha256();hb=hashlib.sha256()
changed=[]; maxround=0.;totaldelta=0.;midpointdelta=0.
for k in sorted(a):
 x,y=a[k],b[k]
 assert x.dtype==y.dtype==torch.float32 and x.shape==y.shape
 assert torch.isfinite(x).all() and torch.isfinite(y).all()
 ideal=(x.double()+y.double())*0.5
 z=ideal.to(torch.float32)
 simple=(x+y)*0.5
 assert torch.isfinite(z).all()
 if not torch.equal(z.view(torch.uint8),simple.view(torch.uint8)):changed.append(k)
 maxround=max(maxround,float((z.double()-ideal).abs().max()))
 totaldelta+=float((y.double()-x.double()).square().sum())
 midpointdelta+=float((z.double()-x.double()).square().sum())
 state_update(h,k,z);state_update(hfp32,k,simple);state_update(ha,k,x);state_update(hb,k,y)
assert ha.hexdigest()==prior["sources"]["START"]["metadata"]["model_state_sha256"]
assert hb.hexdigest()==prior["sources"]["HIST_CAPPED500"]["metadata"]["model_state_sha256"]
assert hfp32.hexdigest()==prior["half_model_state_sha256_for_stated_formula"]
assert not torch.cuda.is_initialized()
receipt={"status":"PASS","formula":"(START.double()+HIST_CAPPED500.double())*0.5 then one FP32 cast",
"independent_fp64_midpoint_model_state_sha256":h.hexdigest(),
"independent_fp32_midpoint_model_state_sha256":hfp32.hexdigest(),
"FP64_vs_FP32_sum_then_half_bitwise_different_tensor_names":changed,
"all_194_parameters_included":True,"parameter_elements":prior["parameter_elements"],"actual_registered_buffers":[],
"source_model_hashes_match":True,"maximum_rounding_abs":maxround,"source_delta_norm":totaldelta**.5,
"midpoint_delta_norm":midpointdelta**.5,"midpoint_delta_ratio":(midpointdelta/totaldelta)**.5,
"original_receipt_path":str(OUT/"interpolation_review.json"),"original_receipt_sha256":sha(OUT/"interpolation_review.json"),
"source_checkpoint_sha256":{k:v["file_sha256"] for k,v in prior["sources"].items()},
"reviewed_blend_source_sha256":sha(ROOT/"experiments/mf_h1_local0/chair_capped500_half_v1/blend.py"),
"review_script_sha256":sha(Path(__file__)),"checkpoint_payload_CPU_loads_this_receipt":2,
"checkpoint_payload_CPU_loads_cumulative_independent_audits":4,
"CUDA_initialized":False,"model_construction_calls_this_receipt":0,"model_forward_calls":0,"backward_calls":0,"optimizer_updates":0,
"implementation_review":"Fresh blend path averages all actual 194 FP32 parameters including role and C-only Geo in FP64, casts once; buffers must match and are copied; strict CPU reload, frozen inference-only payload. No optimizer/RNG/stream supplied.",
"limits":["No output-quality guarantee from midpoint arithmetic.","Fresh-path code review does not replace worker inference-only identity checks or frozen protocol validation."],
"seconds":time.perf_counter()-tick}
(OUT/"interpolation_review_fp64.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
print(json.dumps(receipt))
