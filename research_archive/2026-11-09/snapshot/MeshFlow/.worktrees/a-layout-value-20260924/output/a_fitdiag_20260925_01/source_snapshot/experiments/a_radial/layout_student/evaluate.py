"""Frozen-checkpoint deployment, independent RMS, and explicit fallback accounting."""
from __future__ import annotations
import argparse, datetime, json, time, traceback
from pathlib import Path
import numpy as np
import torch
from ..legacy.networks import DomainVelocity
from ..exporter import RadialSupport, ASSETS
from ..io_utils import file_hash, array_hash, model_hash, write_json, save_npz
from ..layout_value.geometry import evaluate_mesh, audit_mesh
from .deployment import deploy, named_quality, qualification
from .train import model_sha256
from .integrity import mesh_identity

def emit(x): print(json.dumps(x,ensure_ascii=False),flush=True)

def metric_cache(run, uid, v, f, rv, rf, seeds, points):
    identity=array_hash(v,f,rv,rf);values=[]
    for seed in seeds:
        path=run/"metric_cache"/uid/f"{identity}_{seed}_{points}.json"
        if path.exists():
            item=json.loads(path.read_text())
        else:
            begin=time.perf_counter()
            item={"mesh_reference_hash":identity,
                  "metrics":evaluate_mesh(v,f,rv,rf,points=points,seed=seed),
                  "evaluation_seconds":time.perf_counter()-begin}
            write_json(path,item)
        values.append(item)
    return values

def aggregate(run):
    attempts=[json.loads(p.read_text()) for p in sorted((run/"attempts").glob("*/*/result.json"))]
    rows=[r for attempt in attempts for r in attempt.get("rows",[])]
    errors=[a for a in attempts if a["status"]!="success"]
    write_json(run/"results.json",{"rows":rows,"attempt_count":len(attempts),"errors":errors},
               replace=(run/"results.json").exists())
    return rows

def load_network(run, name, protocol):
    if name=="original_a":
        path=ASSETS/"original_A_10.pt"
        state=torch.load(path,map_location="cpu",weights_only=True)
        seed=None;updates=0
    elif name.startswith("student_"):
        seed=int(name.split("_")[1])
        path=run/"training"/f"seed_{seed}"/"final.pt"
        checkpoint=torch.load(path,map_location="cpu",weights_only=False)
        assert checkpoint["protocol_sha256"]==file_hash(run/"RUN_PROTOCOL.json")
        assert checkpoint["update"]==protocol["training"]["updates"] and checkpoint["is_predeclared_final"]
        assert checkpoint["config"]["seed"]==seed
        state=checkpoint["model"];updates=checkpoint["update"]
    else:return None,{"kind":name,"checkpoint":None,"deployment_weight":False}
    model=DomainVelocity().cpu();model.load_state_dict(state,strict=True)
    model.eval().requires_grad_(False)
    identity=model_sha256(model.state_dict())
    if name.startswith("student_"):
        assert identity==checkpoint["model_sha256"]
    return model,{"kind":name,"checkpoint":str(path),"file_sha256":file_hash(path),
                  "model_sha256":identity,"training_seed":seed,"updates":updates,
                  "parameter_count":sum(p.numel() for p in model.parameters()),
                  "architecture":"unchanged legacy.networks.DomainVelocity"}

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True)
    p.add_argument("--models",nargs="+",default=["fixed","original_a","teacher_reference","student_10","student_11"])
    p.add_argument("--uids",nargs="+")
    args=p.parse_args();run=Path(args.run);protocol=json.loads((run/"RUN_PROTOCOL.json").read_text())
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    uids=sorted(protocol["uids"]);selected=args.uids or uids
    assert file_hash(run/"DATA_MANIFEST.json")==protocol["hashes"]["data_manifest_sha256"]
    manifest=json.loads((run/"DATA_MANIFEST.json").read_text())["files"]
    assert file_hash(ASSETS/"original_A_10.pt")==protocol["hashes"]["original_a_sha256"]
    if not set(selected)<=set(uids):raise ValueError("Unregistered UID")
    support=RadialSupport()
    source_hashes={str(p.relative_to(Path(__file__).parents[1])):file_hash(p)
                   for p in Path(__file__).parents[1].rglob("*.py")}
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    write_json(run/"eval_invocations"/f"{stamp}.json",
               {"args":vars(args),"source_hashes":source_hashes,"protocol_sha256":file_hash(run/"RUN_PROTOCOL.json")})
    for name in args.models:
        start=time.perf_counter();flow,identity=load_network(run,name,protocol)
        load_seconds=time.perf_counter()-start
        for uid in selected:
            dest=run/"attempts"/uid/name
            if (dest/"result.json").exists():
                emit({"event":"skip_completed","uid":uid,"model":name});continue
            dest.mkdir(parents=True,exist_ok=False)
            write_json(dest/"START.json",{"uid":uid,"model":name,"identity":identity})
            emit({"event":"start","uid":uid,"model":name})
            try:
                # Prediction receives y, support, and Fixed connectivity only.
                # teacher_u is read ONLY by the explicitly offline teacher_reference arm.
                data_path=run/"data"/f"{uid}.npz"
                assert file_hash(data_path)==manifest[f"data/{uid}.npz"]
                with np.load(data_path,allow_pickle=False) as z:
                    y=z["logr"];base=z["base_u"];faces=z["faces"]
                    rv=z["ref_vertices"];rf=z["ref_faces"]
                    stored_proxy=z["proxy_points"]
                    teacher_u=z["teacher_u"] if name=="teacher_reference" else None
                split="train" if uid in protocol["train_uids"] else "internal_check"
                idx=uids.index(uid);seeds=[192401+idx,292401+idx];points=8192
                if name=="teacher_reference":
                    fixed=deploy(None,y,support,base,faces,fixed_only=True)
                    radius=support.query(y,teacher_u)
                    mesh={"vertices":teacher_u*radius[:,None],"faces":faces,"u":teacher_u,"r":radius}
                    q=named_quality(mesh["vertices"],faces);audit=audit_mesh(mesh["vertices"],faces,exact=True)
                    result={"raw_mesh":mesh,"raw_quality":q,"raw_audit":audit,"floor":fixed["floor"],
                            "raw_qualification":qualification(q,audit,fixed["floor"]),
                            "timing":{"offline_teacher_reference":True,"generation_seconds":None}}
                    repeats=[];policies=["offline_teacher"]
                else:
                    result=deploy(flow,y,support,base,faces,fixed_only=name=="fixed")
                    repeats=[]
                    for _ in range(3):
                        rep=deploy(flow,y,support,base,faces,fixed_only=name=="fixed")
                        assert array_hash(rep["raw_mesh"]["vertices"],faces)==array_hash(result["raw_mesh"]["vertices"],faces)
                        repeats.append(rep["timing"])
                    if name!="fixed":
                        assert result["timing"]["proxy_points_hash"]==array_hash(stored_proxy)
                    policies=["raw"] if name=="fixed" else ["raw","quality_fallback"]
                rows=[]
                for policy in policies:
                    is_fallback_policy=policy=="quality_fallback"
                    prefix="delivered" if is_fallback_policy else "raw"
                    mesh=result[prefix+"_mesh"];q=result[prefix+"_quality"];audit=result[prefix+"_audit"]
                    mesh_path=dest/f"{policy}.npz";save_npz(mesh_path,**mesh)
                    # Read back the actual artifact; logical connectivity and storage hashes have explicit meanings.
                    mesh_identity_fields=mesh_identity(mesh_path,faces)
                    metrics=metric_cache(run,uid,mesh["vertices"],faces,rv,rf,seeds,points)
                    qualified=qualification(q,audit,result["floor"])
                    timing=result["timing"]
                    if policy=="offline_teacher":
                        delivery_seconds=hot_seconds=None
                    else:
                        timing_key="quality_fallback_total_seconds" if is_fallback_policy else "raw_with_audit_seconds"
                        delivery_seconds=timing[timing_key]
                        hot_seconds=float(np.median([x[timing_key] for x in repeats]))
                    rows.append({
                        "uid":uid,"split":split,"model":name,"policy":policy,"status":"success",
                        "offline_teacher":policy=="offline_teacher",
                        "mesh_file":str(mesh_path.relative_to(run)),
                        **mesh_identity_fields,"mainfield_hash":array_hash(y),
                        "data_file_sha256":file_hash(data_path),"identity":identity,
                        "quality":q,"audit":audit,"quality_floor":result["floor"],**qualified,
                        "metrics":[x["metrics"] for x in metrics],
                        "metric_cache_evaluation_seconds":[x["evaluation_seconds"] for x in metrics],
                        "fallback_to_fixed":bool(is_fallback_policy and result["delivery"]["fallback_to_fixed"]),
                        "fallback_reasons":result.get("delivery",{}).get("fallback_reasons",[]) if is_fallback_policy else [],
                        "generation_seconds":timing["generation_seconds"],
                        "delivery_with_audit_seconds":delivery_seconds,
                        "hot_delivery_with_audit_seconds_median_3":hot_seconds,
                        "model_load_seconds":load_seconds,
                        "timing":timing,"hot_timing_repeats":repeats,
                        "cost_scope":"raw: CPU layout/export+self audit, excludes Fixed floor setup; policy: includes Fixed setup+audit+raw+audit+decision; no AE/FM cost",
                        "independent_evaluation_not_used_for_deployment":True,
                    })
                write_json(dest/"result.json",{"uid":uid,"model":name,"status":"success","rows":rows})
                aggregate(run)
                emit({"event":"complete","uid":uid,"model":name,"raw_qualified":rows[0]["qualified"],
                      "raw_rms":rows[0]["metrics"][0]["rms_bbox_percent"],
                      "fallback":rows[-1]["fallback_to_fixed"]})
            except Exception as exc:
                err={"uid":uid,"model":name,"status":"error","error":repr(exc),"traceback":traceback.format_exc(),"rows":[]}
                write_json(dest/"result.json",err);aggregate(run);emit(err)
    aggregate(run)

if __name__=="__main__":main()
