"""CPU-only response evidence; no source-target reconstruction ranking."""
from __future__ import annotations
import os
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ.setdefault(k,"2")
import argparse, collections, json, math
import numpy as np
from . import common as C

def rms(a):return float(np.sqrt(np.mean(np.asarray(a,dtype=np.float64)**2)))
def corner_rms(a):return float(np.sqrt(np.mean(np.sum(np.asarray(a,dtype=np.float64).reshape(-1,3)**2,axis=1))))
def width(a):return float(np.quantile(np.asarray(a).reshape(-1,3)[:,0],.95)-np.quantile(np.asarray(a).reshape(-1,3)[:,0],.05)) if np.size(a) else None
def area(a):
    a=np.asarray(a,dtype=np.float64).reshape(-1,3,3)
    return float(np.linalg.norm(np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]),axis=1).sum()*.5)
def bf16(bits):return (np.asarray(bits,dtype=np.uint32)<<np.uint32(16)).view(np.float32)
def safe_div(a,b):return None if not b else float(a/b)
def equation_spec():
    return dict(version="width_response_descriptive_v1",fixed_L=C.L,GT="NOT_USED",
        velocity_RMS="sqrt(mean over all valid free coordinates of (vplus-vminus)^2))",
        velocity_relative="delta velocity RMS / ((RMS(vplus)+RMS(vminus))/2), NA if denominator zero",
        corner_displacement="sqrt(mean over free corner instances of squared Euclidean displacement); same RAW slots, not semantic correspondence",
        curve_width="q95(x_free_corner_instances)-q05(x_free_corner_instances); triangulation-sensitive",
        width_slope="(Wplus-Wminus)/((alpha_plus-alpha_minus)*Wbase); descriptive, no requirement slope=1",
        negative_and_zero="Report exact numerical zero separately from bitwise equality, continuous values otherwise",
        acceptance_thresholds="NONE; no post-hoc weak-response cutoff",
        condition_response="C hard preservation or numerical change is not semantic adoption",
        POST_correspondence="No cross-alpha pointwise subtraction after face deletion/reindex; use width/quality descriptors separately",
        spatial_bins="free-face centroid X relative to original C min/max X; descriptive, no semantic part assignment",
        source_versions="Original sampling/evaluation remain read-only")
def freeze():
    C.write(C.OUT/"analysis/definition.json",equation_spec())
def run():
    if C.read(C.OUT/"analysis/definition.json")!=equation_spec():raise ValueError("Analysis definition drift")
    reg=C.read(C.OUT/"registration.json")
    rawrows=C.read(C.OUT/"evaluation/raw_index.json")["rows"];postrows=C.read(C.OUT/"evaluation/post_index.json")["rows"]
    raw={r["request_id"]:r for r in rawrows};post={r["request_id"]:r for r in postrows}
    lookup={(q["base_request_id"],q["variant"]):q for q in reg["requests"]}
    by=[q for q in reg["requests"] if q["variant"]=="base"]
    peroutput=[];curves=[];trajectories=[];probes=[];faces=[];memory=[]
    for base in by:
        data={};Cinputs={};paths={}
        for variant in ("m095","base","p105"):
            q=lookup[base["base_request_id"],variant];Cinputs[variant]=C.arrays(q["input_path"])["C"]
            a=C.arrays(raw[q["request_id"]]["raw_path"])
            paths[variant]=a["trajectory"].reshape(51,q["N"],3,3)
            for stage,index,key in (("RAW",raw,"raw_path"),("POST",post,"post_path")):
                a=C.arrays(index[q["request_id"]][key]);tri=a["raw" if stage=="RAW" else "output"].reshape(-1,3,3)
                free=tri[q["K"]:];data[variant,stage]=free
                row={k:q[k] for k in ("request_id","base_request_id","base_condition_id","parent_id","candidate","variant","alpha","N","K","region_type","tier")}
                row.update(stage=stage,width_q95_q05=width(free),free_faces=len(free),free_area=area(free),
                    C_width_minmax=float(np.ptp(Cinputs[variant][...,0])),C_width_q95_q05=width(Cinputs[variant]),
                    free_mean_x=float(free[...,0].mean()) if len(free) else None,
                    free_mean_y=float(free[...,1].mean()) if len(free) else None,
                    free_mean_z=float(free[...,2].mean()) if len(free) else None)
                peroutput.append(row)
        lo,hi=Cinputs["base"][...,0].min(),Cinputs["base"][...,0].max()
        dC=Cinputs["p105"].astype(float)-Cinputs["m095"].astype(float);dCrms=corner_rms(dC)
        da=float(lookup[base["base_request_id"],"p105"]["alpha"]-lookup[base["base_request_id"],"m095"]["alpha"])
        for stage in ("RAW","POST"):
            minus,zero,plus=[data[v,stage] for v in ("m095","base","p105")]
            wm,w0,wp=map(width,(minus,zero,plus))
            row={k:base[k] for k in ("base_request_id","base_condition_id","parent_id","candidate","K","N","region_type","tier")}
            row.update(stage=stage,width_minus=wm,width_base=w0,width_plus=wp,alpha_delta=da,
                width_delta=None if wm is None or wp is None else wp-wm,
                width_slope=None if not w0 or wm is None or wp is None else (wp-wm)/(da*w0),
                width_monotone=bool(wm<=w0<=wp) if None not in (wm,w0,wp) else None,
                C_corner_displacement_rms=dCrms,free_area_minus=area(minus),free_area_base=area(zero),free_area_plus=area(plus))
            if stage=="RAW":
                delta=plus.astype(float)-minus.astype(float)
                row.update(exact_zero_response=bool(np.array_equal(plus,minus)),bitwise_equal=C.same(plus,minus),
                    free_corner_displacement_rms=corner_rms(delta),free_corner_displacement_L=corner_rms(delta)/C.L,
                    displacement_over_C=safe_div(corner_rms(delta),dCrms),
                    displacement_RMS_x=rms(delta[...,0]),displacement_RMS_y=rms(delta[...,1]),displacement_RMS_z=rms(delta[...,2]),
                    changed_coordinate_fraction=float(np.mean(plus!=minus)))
            curves.append(row)
        for step in range(51):
            fminus,fzero,fplus=[paths[v][step,base["K"]:] for v in ("m095","base","p105")]
            d=fplus.astype(float)-fminus.astype(float)
            trajectories.append(dict(base_request_id=base["base_request_id"],parent_id=base["parent_id"],candidate=base["candidate"],
                step=step,time=step/50,corner_displacement_rms=corner_rms(d),corner_displacement_L=corner_rms(d)/C.L,
                width_minus=width(fminus),width_base=width(fzero),width_plus=width(fplus),
                exact_zero=bool(np.array_equal(fplus,fminus))))
        for step in (5,15,30,45):
            folder=C.OUT/"probes"/base["base_request_id"]/("step_"+str(step).zfill(2))
            rec=C.read(folder/"receipt.json")
            a,b=C.arrays(folder/"m095.npz"),C.arrays(folder/"p105.npz")
            k=base["K"];vm,vp=a["velocity"][k:],b["velocity"][k:]
            dv=vp.astype(float)-vm.astype(float)
            hminus,hplus=bf16(a["known_hidden_bf16_bits"]),bf16(b["known_hidden_bf16_bits"])
            dh=hplus.astype(float)-hminus.astype(float)
            row={name:base[name] for name in ("base_request_id","base_condition_id","parent_id","candidate","K","N","region_type","tier")}
            row.update(step=step,time=float(np.float32(step/50)),delta_velocity_rms=rms(dv),
                relative_velocity_response=safe_div(rms(dv),.5*(rms(vm)+rms(vp))),
                mean_delta_vx=float(dv.reshape(-1,3)[:,0].mean()),mean_delta_vy=float(dv.reshape(-1,3)[:,1].mean()),
                mean_delta_vz=float(dv.reshape(-1,3)[:,2].mean()),RMS_delta_vx=rms(dv.reshape(-1,3)[:,0]),
                RMS_delta_vy=rms(dv.reshape(-1,3)[:,1]),RMS_delta_vz=rms(dv.reshape(-1,3)[:,2]),
                delta_velocity_max=float(abs(dv).max()),exact_zero_velocity=bool(np.array_equal(vm,vp)),
                bitwise_equal_velocity=C.same(vm,vp),changed_velocity_fraction=float(np.mean(vm!=vp)),
                known_embedding_delta_rms=rms(dh),known_embedding_delta_max=float(abs(dh).max()),
                known_embedding_changed_fraction=float(np.mean(a["known_hidden_bf16_bits"]!=b["known_hidden_bf16_bits"])),
                known_embedding_changed=rec["known_embedding_changed"],
                free_embedding_bitwise_equal=rec["free_embedding_bitwise_equal"])
            probes.append(row)
            centers=a["free_state"].reshape(-1,3,3).mean(axis=1)
            dvfaces=dv.reshape(-1,3,3)
            for face,(center,df) in enumerate(zip(centers,dvfaces)):
                pos="LEFT_OF_C" if center[0]<lo else "RIGHT_OF_C" if center[0]>hi else "WITHIN_C_X_SPAN"
                faces.append(dict(base_request_id=base["base_request_id"],step=step,time=step/50,
                    free_slot=face,raw_face_slot=face+k,centroid_x=float(center[0]),centroid_y=float(center[1]),centroid_z=float(center[2]),
                    spatial_x_bin=pos,delta_velocity_rms=rms(df),mean_delta_vx=float(df[:,0].mean()),
                    mean_delta_vy=float(df[:,1].mean()),mean_delta_vz=float(df[:,2].mean())))
    C.csvwrite(C.OUT/"analysis/per_output_geometry.csv",peroutput)
    C.csvwrite(C.OUT/"analysis/per_curve_response.csv",curves)
    C.csvwrite(C.OUT/"analysis/per_trajectory_response.csv",trajectories)
    C.csvwrite(C.OUT/"analysis/per_fixed_state_response.csv",probes)
    C.csvwrite(C.OUT/"analysis/per_free_face_velocity_response.csv",faces)
    summaries=[]
    for group in ("ALL","SEAT_HALF","SEAT_FULL","BACK_HALF","BACK_FULL"):
        for stage in ("RAW","POST"):
            selected=[x for x in curves if x["stage"]==stage and (group=="ALL" or x["region_type"]+"_"+x["tier"]==group)]
            # Exactly two equally weighted noises per parent; parent means explicit.
            parents=collections.defaultdict(list)
            for x in selected:parents[x["parent_id"]].append(x)
            keys=("width_slope","width_delta","free_corner_displacement_rms","free_corner_displacement_L","displacement_over_C")
            summary=dict(group=group,stage=stage,parents=len(parents),curves=len(selected))
            for key in keys:
                values=[float(np.mean([x[key] for x in rows if x.get(key) is not None])) for rows in parents.values() if any(x.get(key) is not None for x in rows)]
                summary[key+"_parent_mean"]=float(np.mean(values)) if values else None
                summary[key+"_parent_median"]=float(np.median(values)) if values else None
            summary["positive_width_curves"]=sum(x["width_delta"]>0 for x in selected if x["width_delta"] is not None)
            summary["monotone_width_curves"]=sum(x["width_monotone"] for x in selected if x["width_monotone"] is not None)
            summary["exact_zero_RAW_curves"]=sum(x.get("exact_zero_response",False) for x in selected) if stage=="RAW" else None
            summaries.append(summary)
    C.csvwrite(C.OUT/"analysis/group_response_summary.csv",summaries)
    C.write(C.OUT/"analysis/response_completion.json",dict(status="COMPLETE",outputs=len(peroutput),curves=len(curves),fixed_state_pairs=len(probes),
        trajectory_rows=len(trajectories),free_face_rows=len(faces),no_model_calls=True,
        known_embedding_changed_pairs=sum(x["known_embedding_changed"] for x in probes),
        exact_zero_velocity_pairs=sum(x["exact_zero_velocity"] for x in probes),
        raw_exact_zero_curves=sum(x.get("exact_zero_response",False) for x in curves if x["stage"]=="RAW"),
        definition_sha256=C.sha(C.OUT/"analysis/definition.json")))
    plot(probes,curves,trajectories)
    return summaries
def plot(probes,curves,trajectory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(15,4.3))
    ids=sorted({r["base_request_id"] for r in probes})
    for rid in ids:
        pp=sorted([r for r in probes if r["base_request_id"]==rid],key=lambda r:r["step"])
        axs[0].plot([r["time"] for r in pp],[r["relative_velocity_response"] for r in pp],alpha=.65,marker=".")
        tt=sorted([r for r in trajectory if r["base_request_id"]==rid],key=lambda r:r["step"])
        axs[1].plot([r["time"] for r in tt],[r["corner_displacement_L"] for r in tt],alpha=.65)
        cc=next(r for r in curves if r["base_request_id"]==rid and r["stage"]=="RAW")
        axs[2].plot([.95,1,1.05],[cc["width_minus"]/cc["width_base"],1,cc["width_plus"]/cc["width_base"]],alpha=.65,marker=".")
    axs[0].set(xlabel="Baseline trajectory t",ylabel="RMS(delta v) / mean RMS(v)",title="Same free state, different C")
    axs[1].set(xlabel="Euler time",ylabel="Free corner displacement / fixed L",title="Same noise, independent rollouts")
    axs[2].set(xlabel="C X scaling",ylabel="Free robust X width / base width",title="Width response (no affine target)")
    for ax in axs:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(C.OUT/"analysis/response_overview.png",dpi=170);plt.close(fig)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--freeze",action="store_true");a=p.parse_args()
    if a.freeze:freeze()
    else:run()
