import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="4"
import json,time,traceback
from pathlib import Path
import numpy as np
from meshflow_control import denoise as D
ROOT=Path("/mnt/e/MeshFlow")
VALID=ROOT/"experiments/r2_denoise_publish_v1/gpu_smoke_validation"
OUT=ROOT/"experiments/r2_denoise_publish_v1/gpu_smoke"
REG=ROOT/"experiments/r2_c_restore_bounded_v1/registration.json"
CHECKPOINT=ROOT/"experiments/r2_postprocess_avmr_pmr_v1/assets/chair_denoiser_0030000.pt"
r=D.read_json(REG)["records"][0]
assert r["parent_id"]=="33ec1e64a02ff718faaadcdc0221eb72" and r["candidate"]==0 and r["N"]==218 and r["K"]==6
assert not OUT.exists()
FINAL=ROOT/"experiments/r2_denoise_publish_v1/migration/chain/D0_BASE"/r["request_id"]/"post.npz"
protected=[REG,Path(r["raw_path"]),Path(r["condition_source"]),Path(r["D0_denoised_path"]),Path(r["D0_post_path"]),FINAL]
protected_before={str(p):D.geo_io.file_sha256(p) for p in protected}
assert protected_before[r["raw_path"]]==r["raw_sha256"]
assert protected_before[r["condition_source"]]==r["condition_source_sha256"]
source_before=D.implementation_hashes()
resources=D.resources()
D.write_json(VALID/"resources_wsl_before.json",resources)
fields=resources["gpu"]["stdout"].strip().splitlines()[0].split(',')
total,free=float(fields[1]),float(fields[3])
headroom=max(2048.,.1*total)
assert free-headroom>=2600.
D.write_json(VALID/"dispatch.json",dict(status="REGISTERED",request_id=r["request_id"],N=218,K=6,
    selected_registration=str(REG),registration_sha256=protected_before[str(REG)],worker_count=1,
    CPU_threads=4,additional_headroom_MiB=headroom,free_before_MiB=free,
    allowed_D0_forwards=1,allowed_generator_forwards=0,allowed_training=0,allowed_noise=0,
    no_retry=True,reason="Only one authorized independent model call; observed D0 allowance 2600 MiB plus headroom",
    protected_before=protected_before,implementation_before=source_before,
    validation_driver_sha256=D.geo_io.file_sha256(__file__)))
tick=time.perf_counter()
def comparison(newpath,oldpath):
    new,old=D.load_npz(newpath),D.load_npz(oldpath)
    rows={key:dict(dtype_equal=new[key].dtype==value.dtype,shape_equal=new[key].shape==value.shape,
         bytes_equal=new[key].dtype==value.dtype and new[key].shape==value.shape and new[key].tobytes()==value.tobytes())
         for key,value in old.items() if key in new}
    return dict(new_path=str(newpath),reference_path=str(oldpath),missing_keys=sorted(set(old)-set(new)),
        extra_keys=sorted(set(new)-set(old)),arrays=rows,
        all_equal=set(old)==set(new) and all(v["bytes_equal"] for v in rows.values()))
try:
    result=D.run(input=r["raw_path"],known_faces=r["K"],out=OUT,
        condition=r["condition_source"],checkpoint=CHECKPOINT,coordination="none",winding="source")
    oldproposal=D.load_npz(r["D0_denoised_path"])["official_unconstrained_proposal"]
    newproposal=D.load_npz(OUT/"proposal.npz")["official_unconstrained_proposal"]
    proposal_same=oldproposal.dtype==newproposal.dtype and oldproposal.shape==newproposal.shape and oldproposal.tobytes()==newproposal.tobytes()
    proposal=dict(bytes_equal=proposal_same,shape=list(newproposal.shape),dtype=str(newproposal.dtype),
        max_abs_error=float(np.max(np.abs(newproposal.astype(np.float64)-oldproposal.astype(np.float64)))),
        new_file_sha256=D.geo_io.file_sha256(OUT/"proposal.npz"),old_file_sha256=D.geo_io.file_sha256(r["D0_denoised_path"]))
    canonical=comparison(OUT/"post_canonical.npz",r["D0_post_path"])
    final=comparison(OUT/"post.npz",FINAL)
    events=[json.loads(line) for line in (OUT/"ledger.jsonl").read_text().splitlines()]
    counts={name:sum(e["event"]==name for e in events) for name in ["D0_FORWARD_STARTED","D0_FORWARD_RETURNED","COMPLETE"]}
    protected_after={str(p):D.geo_io.file_sha256(p) for p in protected}
    unchanged=protected_after==protected_before and D.implementation_hashes()==source_before
    good=proposal_same and canonical["all_equal"] and final["all_equal"] and unchanged and all(v==1 for v in counts.values()) and result["model_forwards_this_invocation"]==1
    receipt=dict(status="PASS" if good else "FAIL",request_id=r["request_id"],proposal=proposal,canonical=canonical,final=final,
        operation_counts=counts,model_forwards=1,generator_forwards=0,training_updates=0,new_noise=0,
        protected_and_implementation_unchanged=unchanged,protected_after=protected_after,
        implementation_after=D.implementation_hashes(),checkpoint_sha256=D.OFFICIAL_SHA,
        seconds=time.perf_counter()-tick,denoise_receipt_sha256=D.geo_io.file_sha256(OUT/"denoise.json"),
        resources_after=D.resources())
    D.write_json(VALID/"verification.json",receipt)
    print(json.dumps({"status":receipt["status"],"proposal":proposal,"canonical_all_arrays_equal":canonical["all_equal"],
        "final_all_arrays_equal":final["all_equal"],"operation_counts":counts,"seconds":receipt["seconds"]},indent=2),flush=True)
    if not good:raise AssertionError("One-call migration mismatch; no retry authorized")
except BaseException as error:
    ledger=OUT/"ledger.jsonl"
    events=[json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    D.write_json(VALID/"failure.json",dict(status="STOPPED",error=repr(error),traceback=traceback.format_exc(),
        forward_attempts=sum(e["event"]=="D0_FORWARD_STARTED" for e in events),
        forward_returned=sum(e["event"]=="D0_FORWARD_RETURNED" for e in events),automatic_retry=False,
        seconds=time.perf_counter()-tick,resources_after=D.resources()))
    raise
