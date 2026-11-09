"""Read-only CPU metrics and backend defaults; no model or CUDA initialization."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from native_t1.runtime import configure_stable_runtime
import torch
import numpy as np
R=ROOT/"native_t1/runs/geometry_gate_code_audit_20260930_150400"
configure_stable_runtime()
registration=json.loads((R/"registration.json").read_text())
with np.load(R/"velocities.npz",allow_pickle=False) as raw:
    ref=raw["current_reference"].reshape(1,112,3,3)
    actual=raw["current_combined"]
    transform=registration["transforms"]["combined"]
    faces=np.asarray(transform["faces"]);corners=np.asarray(transform["corners"])
    expected=np.take_along_axis(ref[:,faces],corners[None,:,:,None],axis=2).reshape(1,112,9)
    free=(faces>=12)
    delta=actual.astype(np.float64)-expected.astype(np.float64)
    reference_rms=float(np.sqrt(np.mean(expected[:,free].astype(np.float64)**2)))
    delta_rms=float(np.sqrt(np.mean(delta[:,free]**2)))
    allowed=np.float32(.015625)+np.float32(.03125)*np.abs(expected)
    ratios=np.abs(delta)/allowed
    violations=np.abs(delta)>allowed
    details=[dict(face=int(i),corner=int(j//3),xyz=int(j%3),known=bool(faces[i]<12),
                 reference=float(expected[0,i,j]),actual=float(actual[0,i,j]),
                 abs_error=float(abs(delta[0,i,j])),allowed=float(allowed[0,i,j]),
                 error_over_allowance=float(ratios[0,i,j]))
             for _,i,j in np.argwhere(violations)]
report=dict(schema="attachment_analysis_review_v1",new_model_calls=0,new_backward_calls=0,
    optimizer_updates=0,complete_rollouts=0,CUDA_initialized=torch.cuda.is_initialized(),
    fresh_process_environment=dict(torch_version=torch.__version__,compiled_CUDA=torch.version.cuda,
        math_SDPA_low_precision_reduction_allowed=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        BF16_GEMM_reduced_precision_reduction_allowed=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        scope="Read from a fresh CPU-only process in the same installed environment after configure_stable_runtime; no precision flags modified beyond original runtime policy."),
    free_velocity=dict(reference_RMS=reference_rms,difference_RMS=delta_rms,
        relative_RMS=delta_rms/reference_rms,relative_RMS_percent=100*delta_rms/reference_rms,
        units="Velocity model-coordinate units; relative to velocity RMS, not bbox/surface error"),
    violations=details,max_error_over_allowance=float(ratios.max()),
    official_references=[
"https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.scaled_dot_product_attention.html",
"https://docs.pytorch.org/docs/2.7/notes/numerical_accuracy.html",
"https://docs.pytorch.org/docs/2.7/generated/torch.use_deterministic_algorithms.html"],
    precision_comparison="NOT_RUN; prior 9-call BF16 diagnosis retained unchanged")
if report["CUDA_initialized"]:raise RuntimeError("CPU review unexpectedly initialized CUDA")
(R/"attachment_analysis_review.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k not in ("violations","official_references")},ensure_ascii=False))
