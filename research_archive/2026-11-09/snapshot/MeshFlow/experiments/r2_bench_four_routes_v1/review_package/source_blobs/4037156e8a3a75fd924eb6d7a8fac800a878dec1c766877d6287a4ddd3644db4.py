"""Exact frozen Natural-N execution, with an independent wider-pool CPU gate.

No model function is copied, patched, or replaced. The runtime version remains
the inherited version because neural arithmetic and N128..799 are unchanged.
"""
from pathlib import Path
import numpy as np
from r2.pilots.bench_natural_n_recipe_v1 import runtime as inherited
from r2.pilots.bench_natural_n_recipe_v1.runtime import (
    NativeModel, build_model, native_execution, edge_features, inference_layout,
    supervised_sample, sample_for_training, verify_sample, VERSION, N_MIN, N_MAX,
)
WRAPPER_VERSION = "bench_high_joint_runtime_wrapper_v1"


def cpu_gate():
    from . import common as C, stream
    from r2.pilots.bench_parts_convergence_v1 import data as D
    import torch
    paths = [Path(__file__),Path(inherited.__file__),Path(__file__).with_name("stream.py"),
             Path(__file__).with_name("training.py"),Path(D.__file__)]
    hashes = {str(p):C.sha(p) for p in paths}
    dest=C.OUT/"runtime_cpu_gate.json"
    if dest.exists():
        saved=C.read(dest)
        if saved["source_hashes"] != hashes or saved["manifest_sha256"] != C.sha(C.OUT/"data/manifest.json"):
            raise ValueError("Runtime gate source or actual data changed")
        return saved
    if torch.cuda.is_initialized():raise RuntimeError("CPU gate must not own CUDA")
    prior=C.read(C.NARROW_ROOT/"runtime_cpu_gate.json")
    if prior["status"]!="PASS":raise ValueError("Inherited runtime structural evidence missing")
    for p,h in prior["source_hashes"].items():
        if C.sha(p)!=h:raise ValueError("Inherited runtime source changed: "+p)
    if any(globals()[n] is not getattr(inherited,n) for n in (
            "NativeModel","build_model","native_execution","inference_layout",
            "edge_features","sample_for_training","supervised_sample","verify_sample")):
        raise ValueError("Frozen runtime function identity changed")
    m=C.read(C.OUT/"data/manifest.json");pool=stream.validate_manifest(m)
    wide=stream.make_schedule(m["train"],"WIDE_HIGH")
    paired=stream.validate_frozen_narrow(wide,m["train"])
    layouts=[];protected={}
    for row in m["train"]+m.get("high_holdout",[]):
        case=D.load_case(row)
        path=C.resolve(row["case_path"])
        if C.sha(path)!=row["case_sha256"]:raise ValueError("Actual case bytes changed")
        protected[str(path)]=row["case_sha256"]
        layout=inference_layout(case["C_vertices"],case["C_faces_original"],case["C_faces_model"],int(row["N"]))
        if not np.array_equal(layout["C_vertices"][layout["C_faces_model"]],case["full_target"][case["known_ids"]]):
            raise ValueError("Preserved C differs from actual target")
        sample=dict(N=layout["N"],K=layout["K"],anchor_ids=layout["anchor_ids"],
            known_mask=layout["known_mask"],valid_mask=layout["valid_mask"],
            occupied_labels=case["occupied_labels"],occupied_for_FM=layout["occupied"],
            free_coordinate_count=layout["free_coordinate_count"])
        verify_sample(sample)
        layouts.append(dict(condition_id=row["condition_id"],N=int(row["N"]),K=int(row["K"]),
                            free_scalars=layout["free_coordinate_count"],model_calls=0,OT_calls=0))
    boundary=[]
    for n in (776,799):
        lay=inference_layout(case["C_vertices"],case["C_faces_original"],case["C_faces_model"],n)
        k=int(lay["K"])
        if int(lay["free_coordinate_count"]) != 9*(n-k) or np.any(lay["anchor_ids"][k:]!=-1):
            raise ValueError("High-N free-coordinate/anchor mask contract differs")
        valid=torch.from_numpy(lay["valid_mask"]);known=torch.from_numpy(lay["known_mask"])
        if valid.numel()!=n or int(valid.sum())!=n or int(known.sum())!=k:
            raise ValueError("CPU high-N tensor masks differ")
        boundary.append(dict(N=n,K=k,free_scalars=9*(n-k),CPU_tensor_masks_exact=True,OT_calls=0))
    for n in (127,800):
        try:inference_layout(case["C_vertices"],case["C_faces_original"],case["C_faces_model"],n)
        except ValueError:pass
        else:raise ValueError("Out-of-contract face count accepted")
    result=dict(status="PASS",runtime_version=VERSION,wrapper_version=WRAPPER_VERSION,N_contract=[128,799],
        inherited_runtime_gate=str(C.NARROW_ROOT/"runtime_cpu_gate.json"),
        inherited_runtime_gate_sha256=C.sha(C.NARROW_ROOT/"runtime_cpu_gate.json"),
        exact_inherited_function_identity=True,same_parameter_names_shapes=True,
        generator_parameters=prior["generator_parameters"],generator_tensors=prior["generator_tensors"],
        count_embedding_shape=prior["count_embedding_shape"],pool=pool,paired_stream=paired,
        layouts=layouts,mask_boundaries=boundary,rejected_N=[127,800],source_hashes=hashes,manifest_sha256=C.sha(C.OUT/"data/manifest.json"),
        protected=protected,model_forwards=0,production_OT_calls=0,tensor_weight_loads=0,CUDA_initialized=False)
    C.write_new(dest,result);return result


if __name__=="__main__":
    import argparse,json
    parser=argparse.ArgumentParser();parser.add_argument("--cpu-gate",action="store_true",required=True);parser.parse_args()
    r=cpu_gate();print(json.dumps({k:r[k] for k in ("status","runtime_version","model_forwards","production_OT_calls")}))
