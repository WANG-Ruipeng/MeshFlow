"""Presentation only for the frozen Edge-only factorial; no scoring or models."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from .evaluation import MODELS,EFFECTS,_path,_sha

MAIN=(
 "remainder_symmetric_RMS_bbox_pct","free_to_R12_RMS_bbox_pct","R12_to_free_RMS_bbox_pct",
 "fixed_R12_coverage_within_1pct_fraction","boundary_RMS_bbox_pct","boundary_p95_bbox_pct",
 "boundary_coverage_fraction","Gamma_to_free_edges_RMS_bbox_pct","Gamma_to_free_edges_p95_bbox_pct",
 "Gamma_to_free_edges_coverage_0.001_fraction","CF_proper_affected_free_fraction","CF_proper_pair_count",
 "FF_illegal_pair_count","free_minimum_angle_lt5_fraction","free_q_gt10_fraction","free_q_mean",
 "free_total_surface_area")
QUESTIONS=(
 "Does Edge-only provide an independent complete-generation gain?",
 "Is L5 still necessary on the Edge background?",
 "Do joint gains extend to previously unobserved context sets?",
 "Do surface proximity and free-edge proximity agree?",
 "Do P0/patch2 failure and seed concentration persist?",
 "What bias does five-point quadrature show, and is a separate follow-up warranted?")
DISPLAY={"L0":"L0 / M00","Edge_only":"Edge-only / M10","L5":"L5 / M01","JEdge5":"JEdge5 / M11"}
LABELS={
 "remainder_symmetric_RMS_bbox_pct":"Free symmetric RMS (%L)",
 "free_to_R12_RMS_bbox_pct":"Free to GT RMS (%L)","R12_to_free_RMS_bbox_pct":"GT to free RMS (%L)",
 "fixed_R12_coverage_within_1pct_fraction":"Fixed-GT coverage <1% L",
 "boundary_RMS_bbox_pct":"Gamma to surface RMS (%L)","boundary_p95_bbox_pct":"Gamma to surface p95 (%L)",
 "boundary_coverage_fraction":"Gamma to surface coverage <0.1% L",
 "Gamma_to_free_edges_RMS_bbox_pct":"Gamma to free edges RMS (%L)",
 "Gamma_to_free_edges_p95_bbox_pct":"Gamma to free edges p95 (%L)",
 "Gamma_to_free_edges_coverage_0.001_fraction":"Gamma to free edges coverage <0.1% L",
 "CF_proper_affected_free_fraction":"CF affected free-face fraction",
 "CF_proper_pair_count":"CF proper pairs","FF_illegal_pair_count":"FF illegal pairs",
 "free_minimum_angle_lt5_fraction":"Free min-angle <5deg fraction","free_q_gt10_fraction":"Free q>10 fraction",
 "free_q_mean":"Free mean q","free_total_surface_area":"Free total area (model units squared)"}

def _fmt(v):
    if v is None:return "NOT_RUN"
    if isinstance(v,bool):return str(v)
    if isinstance(v,float):return f"{v:.7g}"
    return str(v).replace("|","\\|").replace("\n"," ")

def _table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","| "+" | ".join("---" for _ in headers)+" |"]+
        ["| "+" | ".join(_fmt(v) for v in row)+" |" for row in rows])+"\n"

def _wins(stat):
    c=stat.get("improvement_counts",{})
    return f"{c.get('better')}/{c.get('available')}" if c.get("better") is not None else "DESCRIPTIVE"

def _worst(stat):
    w=stat.get("worst")
    if not w:return "DESCRIPTIVE" if stat.get("higher_is_better") is None else "NOT_RUN"
    where="/".join(str(w[k]) for k in ("unit_id","seed") if w.get(k) is not None)
    return _fmt(w["value"])+" @ "+where

def _answers(r):
    values=r.get("interpretation",{}).get("answers",{})
    return [values.get(str(i),values.get(i,"PENDING_INTERPRETATION")) if isinstance(values,dict)
        else values[i-1] if i<=len(values) else "PENDING_INTERPRETATION" for i in range(1,7)]

def _flatten_counts(value,prefix=""):
    out=[]
    if isinstance(value,dict):
        for k,v in value.items():
            name=prefix+"."+str(k) if prefix else str(k)
            if isinstance(v,dict):out.extend(_flatten_counts(v,name))
            elif isinstance(v,(int,float)) and not isinstance(v,bool):out.append((name,v))
    return out

def _presentation_fields(value, prefix=""):
    """Compact saved metadata only; no new measurements or derived totals."""
    rows=[]
    if isinstance(value,dict):
        for key,item in value.items():
            name=prefix+"."+str(key) if prefix else str(key)
            if isinstance(item,dict):rows.extend(_presentation_fields(item,name))
            elif isinstance(item,(str,int,float,bool)) or item is None:rows.append((name,item))
            elif isinstance(item,(list,tuple)):
                if len(item)<=8 and all(isinstance(x,(str,int,float,bool)) or x is None for x in item):
                    rows.append((name,", ".join(_fmt(x) for x in item)))
                else:rows.append((name+" (record count; details in results.json)",len(item)))
    return rows

def _teacher_section(r):
    teacher=r.get("teacher",{})
    if not teacher:return "Teacher: NOT_RUN or not yet attached.\n"
    models=teacher.get("models",{})
    rows=[]
    for m in MODELS:
        b=models.get(m,{})
        means=b.get("means",{})
        rows.append((m,b.get("status"),len(b.get("rows",[])),b.get("reused",b.get("reused_existing_teacher")),
            *[means.get(k) for k in ("FM","L3","L5","D_edge")]))
    out=_table(["Model","Status","States","Reused","FM","L3","L5","D_edge"],rows)
    out+="\nOnly Edge-only adds the registered 16 teacher forwards. Old models are read-only reuse after endpoint/state/metric identity checks; NOT_COMPARABLE remains explicit. These are eight training pairs at t=.5/.75, not 16 independent objects or a training-distribution expectation. Teacher ranks do not determine generation quality.\n"
    return out

def render_report(r,figures=()):
    summary=r.get("summaries",{}).get("pilot")
    if summary is None:raise ValueError("Complete scored factorial summary required")
    lines=["# Edge-only fixed-coefficient factorial report","",
        "This run adds one Edge-only +500-update arm from G0 cumulative2500 and reuses the L0/L5/JEdge5 cumulative3000 endpoints. All registered outputs in this noise batch are freshly generated. The default model is unchanged.",""]
    for i,(q,a) in enumerate(zip(QUESTIONS,_answers(r)),1):
        lines.extend([f"**{i}. {q}**",str(a),""])
    lines.extend(["The evidence concerns two previously trained parents and one training seed. New context sets were not used in fine-tuning or earlier recorded evaluation, but their objects and some constituent triangles were observed as training targets. No new-shape or training-seed generalization is established.","",
        "## Frozen matrix and assets",""])
    ev=r["protocol"]["evaluation"];reg=r.get("registration",{})
    lines.append(_table(["Item","Value"],[
        ("Models",", ".join(MODELS)),("Generation seeds",ev["seeds"]),
        ("Registered units",len(summary["units"])),("Formal outputs",len(r.get("generated",[]))),
        ("Old/new context units",str(sum(u["patch_group"]=="old" for u in summary["units"]))+"/"+str(sum(u["patch_group"]=="new" for u in summary["units"]))),
        ("N/K/free and Euler","112 / 12 / 100; 50 Euler steps"),
        ("All-C states","51 per rollout; exact preservation checked from raw"),
        ("Git HEAD",r.get("git",{}).get("HEAD")),
        ("Seed registration",reg.get("seed_audit",{}).get("status")),
        ("Historical context audit scope",reg.get("history_audit",{}).get("scope","See registration JSON"))]))
    assets=r.get("assets",{});asset_rows=[]
    for name in ("initial_checkpoint","inputs","evaluation_inputs","teacher_samples"):
        a=assets.get(name,{})
        asset_rows.append((name,a.get("path"),a.get("bytes"),a.get("sha256")))
    for name,a in assets.get("endpoints",{}).items():
        asset_rows.append(("endpoint "+name,a.get("path"),a.get("bytes"),a.get("sha256",a.get("checkpoint_sha256"))))
    lines.append(_table(["Identity","Path","Bytes","File SHA256"],asset_rows))
    lines.append("Context novelty was registered by geometry and raw source-ID BFS before outputs. New patch3 in each parent differs from a historical set by only one face (11/12 overlap); it is a new face set with weak geometric novelty. Patch4 has substantially lower historical overlap. No patch was selected by generated quality.\n")
    lines.append(_table(["Unit","Group","Area/full area","Interface edges","Interface length","Connected components","Faces different from training C12","Max historical overlap"],
        [(u["unit_id"],u["patch_group"],u.get("patch_area_fraction"),u.get("source_interface_edge_count"),
          u.get("source_interface_length"),u.get("source_connected_components"),
          u.get("different_faces_from_training"),u.get("max_historical_overlap_fraction")) for u in summary["units"]]))

    state_rows=[(m,a.get("state_sha256",a.get("model_state_sha256")),a.get("cumulative_updates")) for m,a in assets.get("endpoints",{}).items()]
    lines.append(_table(["Model","Generator state SHA256","Cumulative updates"],state_rows))
    lines.extend(["Removed historical training-file paths were resolved through verified compact exports with identical tensors and input RNG. No approximate endpoint substitution or old-output reuse is implied.","",
        "## Complete generation: old and new contexts separately","",
        "Errors, p95 and collision counts are lower-is-better; coverage is higher-is-better. Area and exact-connectivity counts are descriptive. Each macro is the noise mean within a parent/patch cell, then the equal mean of those cells. Four noisy outputs are not four independent parent objects.",""])
    for group,scope in summary["scopes"].items():
        lines.extend([f"### {group.upper()} contexts",""])
        if not scope["units"]:
            lines.append("NOT_RUN: no eligible registered units.\n");continue
        rows=[(LABELS.get(k,k),*[scope["models"][m]["macro_mean"].get(k) for m in MODELS]) for k in MAIN]
        lines.append(_table(["Metric",*[DISPLAY[m] for m in MODELS]],rows))
        lines.append("Model means, medians and worst observed values for the two primary new-context outcomes (old cells retained as costs):\n")
        lines.append(_table(["Model","Metric","Mean","Median","Worst"],
            [(m,LABELS[k],scope["models"][m]["metrics"][k]["mean"],scope["models"][m]["metrics"][k]["median"],
              _worst(scope["models"][m]["metrics"][k])) for m in MODELS
             for k in ("remainder_symmetric_RMS_bbox_pct","boundary_p95_bbox_pct","Gamma_to_free_edges_p95_bbox_pct")]))
        lines.append("All metrics, including topology inheritance, degeneracy, NaN/Inf, coplanar/unresolved pairs, and all five fixed coverages:\n")
        lines.append("<details><summary>Complete scalar macro table</summary>\n")
        lines.append(_table(["Metric",*MODELS],[(k,*[scope["models"][m]["macro_mean"][k] for m in MODELS]) for k in summary["metric_names"]]))
        lines.append("</details>\n")
    lines.extend(["## Matched four-cell effects","",
        "Edge_off=M10-M00; L5_off=M01-M00; Edge_on=M11-M01; L5_on=M11-M10; I=M11-M10-M01+M00. Each difference is formed for the same parent/patch/seed before means or medians. A paired median is not a difference of model medians. I is fixed-coefficient non-additivity; unequal auxiliary exposure prevents a pure-mechanism synergy claim.",""])
    for group,scope in summary["scopes"].items():
        if not scope["units"]:continue
        lines.extend([f"### {group.upper()} paired effects",""])
        for e in EFFECTS:
            b=scope["effects"][e]
            lines.append(f"**{e}**\n")
            lines.append(_table(["Metric","Mean delta","Paired median","Better/outputs","Worst delta"],
                [(LABELS.get(k,k),b["macro_mean"][k],b["metrics"][k]["median"],_wins(b["metrics"][k]),_worst(b["metrics"][k])) for k in MAIN]))
        lines.append("Parent means remain separate within each old/new context group:\n")
        lines.append(_table(["Effect","Parent","Free RMS","Surface p95","Edge p95","Strict surface coverage","Strict edge coverage","FF pairs"],
            [(e,p,*[b["macro_mean"][k] for k in ("remainder_symmetric_RMS_bbox_pct","boundary_p95_bbox_pct",
                "Gamma_to_free_edges_p95_bbox_pct","boundary_coverage_fraction","Gamma_to_free_edges_coverage_0.001_fraction","FF_illegal_pair_count")])
             for e,v in scope["effects"].items() for p,b in v["parents"].items()]))
        lines.append("Cell-level paired means/medians and improvement counts (including P0/patch2):\n")
        lines.append(_table(["Effect","Cell","RMS mean/median/wins","Surface p95 mean/median/wins","Edge p95 mean/median/wins","FF mean/wins","Strict edge coverage delta"],
            [(e,uid,*[f"{_fmt(b['metrics'][k]['mean'])} / {_fmt(b['metrics'][k]['median'])} / {_wins(b['metrics'][k])}"
                for k in ("remainder_symmetric_RMS_bbox_pct","boundary_p95_bbox_pct","Gamma_to_free_edges_p95_bbox_pct")],
              f"{_fmt(b['metrics']['FF_illegal_pair_count']['mean'])} / {_wins(b['metrics']['FF_illegal_pair_count'])}",
              b["metrics"]["Gamma_to_free_edges_coverage_0.001_fraction"]["mean"])
             for e,v in scope["effects"].items() for uid,b in v["cells"].items()]))
    lines.extend(["## Every seed and leave-one-seed-out sensitivity","",
        "All four seeds remain in the primary results. Each row below omits exactly one registered seed and recomputes the equal-cell macro; no row selects a new winner. Full per-cell/per-parent omitted-seed values are retained in results.json.",""])
    sensitivity=("remainder_symmetric_RMS_bbox_pct","boundary_p95_bbox_pct","Gamma_to_free_edges_p95_bbox_pct",
        "boundary_coverage_fraction","Gamma_to_free_edges_coverage_0.001_fraction","FF_illegal_pair_count")
    for group,scope in summary["scopes"].items():
        if not scope["units"]:continue
        lines.extend([f"### {group.upper()} seed sensitivity",""])
        rows=[]
        for e,b in scope["effects"].items():
            rows.extend((e,"seed-only",v["seed"],*[v[k] for k in sensitivity]) for v in b["seeds"])
            rows.extend((e,"omit-seed",s,*[v["macro_mean"][k] for k in sensitivity]) for s,v in b["leave_one_seed_out"].items())
        lines.append(_table(["Effect","Scope","Seed",*[LABELS.get(k,k) for k in sensitivity]],rows))
    lines.extend(["## Condition binding and geometric interpretation","",
        "The four-grid binding reuses the same raw free sets and fixed target samples. Positive binding is already established in prior baselines; it is not a newly invented success gate. Low boundary distance is neither shared-index connectivity nor watertightness.",""])
    binding_rows=[]
    for j,b in summary["bindings"].items():
        if b["status"]!="PASS":
            binding_rows.append((j,"NOT_RUN",None,None,None,None));continue
        for m,v in b["models"].items():
            binding_rows.append((j,m,v["metrics"]["Delta0"]["mean"],v["metrics"]["Delta1"]["mean"],
                v["metrics"]["B"]["mean"],v["both_positive_count"]))
    lines.append(_table(["Patch","Model","Delta0 mean","Delta1 mean","B mean","Both positive /4"],binding_rows))
    lines.append(_table(["Patch","Effect","B mean","B paired median","Higher B /4"],
        [(j,e,v["metrics"]["B"]["mean"],v["metrics"]["B"]["median"],_wins(v["metrics"]["B"]))
         for j,b in summary["bindings"].items() if b["status"]=="PASS" for e,v in b["effects"].items()]))
    lines.extend(["Gamma-to-free-edges uses the exact same registered boundary points/weights, every finite free triangle edge, no GT correspondence, welding, duplicate-edge removal or coordinate changes. Zero-length edges are points. Point-to-edge distance must be at least point-to-surface distance within 1e-10 L. Both diagnostics report strict coverages at 0.05%, 0.1%, 0.2%, 0.5%, 1.0% L; 0.1% remains the historical reference.","",
        "## Teacher, training and quadrature evidence",""])
    lines.append(_teacher_section(r))
    train=r.get("training",{})
    lines.append(_table(["Step","FM","Gated Edge mean","Weighted geometry","Total","Active/8","Clip-before norm"],
        [(x["step"],x.get("FM"),x.get("edge_gated_mean"),x.get("weighted_geometry"),x.get("total"),
          x.get("active_samples"),x.get("grad_norm_before_clip")) for x in train.get("logs",[])]))
    lines.append(_table(["Step","Parameter group","Gradient RMS before clip","Update RMS","Gradient finite/nonzero","Update finite/nonzero"],
        [(x["step"],g,v.get("RMS"),x.get("parameter_updates",{}).get(g,{}).get("RMS"),
          str(v.get("finite"))+"/"+str(v.get("nonzero")),
          str(x.get("parameter_updates",{}).get(g,{}).get("finite"))+"/"+str(x.get("parameter_updates",{}).get(g,{}).get("nonzero")))
         for x in train.get("logs",[]) for g,v in x.get("gradients_before_clip",{}).items()]))
    lines.append("Training log forward values precede their recorded optimizer update: the step500 forward uses cumulative2999 weights, whereas complete generation and the new teacher diagnostic use the saved cumulative3000 endpoint. Different logged batches/times do not establish convergence. Edge-only never adds L5 to its training backward.\n")
    lines.append(_table(["Step","Microbatch","t","FM gradient norm","Edge gradient norm","Weighted Edge/FM","Cosine","Angle degrees"],
        [(x["step"],x.get("velocity_gradient",{}).get("microbatch_index"),x.get("velocity_gradient",{}).get("t"),
          x.get("velocity_gradient",{}).get("FM_L2"),x.get("velocity_gradient",{}).get("edge_L2"),
          x.get("velocity_gradient",{}).get("weighted_ratio"),x.get("velocity_gradient",{}).get("cosine"),
          x.get("velocity_gradient",{}).get("angle_degrees")) for x in train.get("logs",[]) if x["step"] in (100,500)]))
    q=r.get("quadrature_diagnostic",r.get("quadrature",{}))
    lines.append("The independent FP64 quadrature diagnostic is delivered as quadrature_diagnostic.json and its loss/derivative figure. Five-point production loss and dense references are separate; diagnostic findings do not alter this arm, coefficients or sampler.\n")
    if q:
        lines.append(_table(["Quadrature item","Value"],[("Status",q.get("status")),
            ("Dense cases not converged",q.get("dense_not_converged",[])),("Cases",len(q.get("cases",[])))]))
        qrows=[]
        for x in q.get("cases",[]):
            original=x.get("quadrature",{}).get("5",{})
            dense=x.get("quadrature",{}).get("1025",{})
            analytic=x.get("analytic",{})
            qrows.append((x.get("name"),original.get("loss"),dense.get("loss"),analytic.get("loss"),
                original.get("parameter_derivative"),analytic.get("parameter_derivative"),x.get("dense_convergence",{}).get("status")))
        lines.append(_table(["Case","Five-point","1025-point","Analytic continuous","Five derivative","Analytic derivative","Dense convergence"],qrows))
    if q:
        lines.append("![Independent fixed quadrature loss and directional derivative diagnostic](quadrature_loss_derivative.png)\n")
    lines.append("For a collinear shift 0<delta<ell/4, the production five-point value is delta^2/(8 L^2), while the continuous symmetric mean is delta^3/(3 ell L^2). The ratio is 3 ell/(8 delta): small shifts can be over-weighted by sparse endpoint quadrature. A symmetric midpoint collapse can retain positive loss with zero gradient. Neither observation proves that denser integration improves generation.\n")
    lines.extend(["## Runtime, costs, checks and endpoint",""])
    accounting=r.get("actual_accounting",{})
    lines.append("Actual total accounting is copied from the final ledger; stage-specific counters below are components and must not be added to these totals again.\n")
    lines.append(_table(["Actual total ledger field","Recorded value"],_presentation_fields(accounting)) if accounting else "Final total ledger: NOT_RECORDED.\n")
    environment=r.get("environment",{})
    environment_keys=("python","platform","torch","torch_cuda","cuda","cudnn","numpy","scipy","trimesh", "GPU","gpu","gpu_name","gpu_memory","gpu_memory_total","driver","driver_version","nvidia_smi","CUDA_initialised_by_environment_probe")
    environment_rows=[]
    for key in environment_keys:
        if key not in environment:continue
        value=environment[key]
        if isinstance(value,dict):environment_rows.extend(_presentation_fields(value,key))
        else:environment_rows.append((key,value))
    lines.append(_table(["Runtime environment / GPU","Recorded value"],environment_rows) if environment_rows else "Environment identity: see [environment.json](environment.json).\n")
    if environment.get("note"):lines.append(str(environment["note"])+"\n")
    timing_summary=r.get("timing_summary",{})
    lines.append("Final timing buckets are separately recorded; metadata labels identify seconds and any inclusion/exclusion rules. Overlapping diagnostic and stage timers are not summed here.\n")
    lines.append(_table(["Final timing field","Recorded value"],_presentation_fields(timing_summary)) if timing_summary else "Final timing summary: NOT_RECORDED; available stage timings follow.\n")
    runtime=next((x.get("runtime") for x in train.get("logs",[]) if x.get("runtime")),{})
    if runtime:lines.append(_table(["Actual runtime field","Value"],[(k,v) for k,v in runtime.items() if not isinstance(v,(dict,list))]))
    lines.append(_table(["Counter","Actual"],_flatten_counts(r.get("counts",{}))))
    lines.append(_table(["Training/diagnostic loss work","Actual"],_flatten_counts(r.get("loss_accounting",{}))))
    for name,record in r.get("report_test_records",{}).items():
        lines.append(_table(["Saved test record","Status","Checks","CPU/autograd scope"],
            [(name,record.get("status"),record.get("checks",record.get("tests")),record.get("scope",record.get("counts")))]))

    lines.append(_table(["Offline audit counter","Actual"],_flatten_counts(r.get("evaluation_ledger",{}).get("counts",{}))))
    timing=[("training."+k,v) for k,v in train.get("timing",{}).items()]
    timing += [("stage."+k,v) for k,v in r.get("timing",{}).items() if isinstance(v,(float,int))]
    timing.append(("offline audit",r.get("evaluation_ledger",{}).get("seconds")))
    lines.append(_table(["Cost bucket","Seconds"],timing))
    lines.append("Model-update timing includes geometry loss, backward, synchronization and scheduled diagnostics. CPU OT/stream preparation, supervision setup and checkpoint saving are separately recorded. Sampling includes the recorded transfer/trajectory-storage work; these times are not pure network throughput. The inference definition remains 50 generator calls per completed rollout.\n")
    lines.append(_table(["Check","Status"],[
        ("Wiring",r.get("wiring",{}).get("status")),("Strict saved endpoint reload",train.get("reload_check",{}).get("status")),
        ("Training",train.get("status")),("Evaluation",r.get("evaluation_ledger",{}).get("status")),
        ("Final protection",r.get("final_protection_audit",{}).get("status","See external protection record")),
        ("Default model changed",False)]))
    lines.append("The inherited BF16 face-permutation FAIL remains an inherited numerical limitation at its original tolerance; this auxiliary-loss experiment does not certify that it was fixed. Independent numerical triangle tests are not exact-predicate or continuous-collision proofs. Unexecuted audits remain NOT_RUN.\n")
    cp=train.get("checkpoints",{}).get("500",{})
    if cp:
        lines.append("Strict Edge-only inference load (one process; no training or sampling):\n")
        lines.append("~~~bash\npython -B -c \"from experiments.mf_h1_local0.edge_factorial_pilot.checkpoint import load; m,a=load('"+str(cp.get("path"))+"'); print(a)\"\n~~~\n")
    lines.extend(["## Delivery files", ""])
    lines.append(_table(["Artifact","Link"],[
        ("Complete result and metadata", "[results.json](results.json)"),
        ("Per-output metrics", "[compact_per_output.csv](compact_per_output.csv)"),
        ("Matched per-seed factorial effects", "[compact_per_seed_effects.csv](compact_per_seed_effects.csv)"),
        ("Code diff", "[code.diff](code.diff)"),
        ("Frozen input / patch / noise registration", "[registration/registration.json](registration/registration.json)"),
        ("CPU mathematical fixture record", "[cpu_math_tests.json](cpu_math_tests.json)"),
        ("Independent raw-output / asset protection audit", "[final_raw_asset_audit.json](final_raw_asset_audit.json)"),
        ("Independent mathematics / wiring review", "[math_and_wiring_review.json](math_and_wiring_review.json)"),
        ("Independent scalar summary review", "[summary_review.json](summary_review.json)"),
        ("Environment versions and GPU", "[environment.json](environment.json)"),
        ("Independent quadrature diagnostic", "[quadrature_diagnostic.json](quadrature_diagnostic.json)")]))
    lines.extend(["## Fixed first-seed raw views","",
        "The first registered seed was fixed before generation. Every image uses unchanged raw coordinates, the same Y-up camera (elevation18, azimuth-60), and one common set of bounds across models/parents/patches. It is a fixed illustration, not an outcome-selected representative.",""])
    for figure in figures:
        lines.append(f"![Patch {figure['patch']}; full/free, both parents, four models]({Path(figure['path']).name})\n")
    lines.append("Next step: "+str(r.get("interpretation",{}).get("next_step","STOP; no automatic new experiment or default switch.")))
    lines.append("\nNo quadrature change, new coefficient, source anchoring, third loss, extra training or automatic default switch is performed.")
    return "\n".join(lines)+"\n"

def _figures(r,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    out=_path(out);seed=int(r["protocol"]["evaluation"]["seeds"][0])
    units=r["summaries"]["pilot"]["units"];index={}
    for row in r["generated"]:
        if int(row["seed"])!=seed:continue
        path=_path(row["raw_path"])
        if _sha(path)!=row["raw_sha256"]:raise ValueError("Raw changed before plotting")
        with np.load(path,allow_pickle=False) as z:full=z["output"].copy().reshape(112,3,3)
        if full.dtype!=np.float32 or not np.isfinite(full).all():raise ValueError("Invalid raw figure input")
        index[row["model"],int(row["parent"]),int(row["patch"])]=full
    all_points=np.concatenate([x.reshape(-1,3) for x in index.values()])
    low,high=all_points.min(0),all_points.max(0);center=(low+high)/2;radius=float((high-low).max())*.55
    paths=[out/f"fixed_seed{seed}_patch{j}_v2.png" for j in sorted({u["patch"] for u in units})]
    if any(p.exists() for p in paths):raise FileExistsError("Fixed figures already exist; no silent overwrite")
    artifacts=[]
    for path,j in zip(paths,sorted({u["patch"] for u in units})):
        fig=plt.figure(figsize=(16,14))
        for parent in (0,1):
            for free_only in (False,True):
                row=2*parent+int(free_only)
                for col,m in enumerate(MODELS):
                    ax=fig.add_subplot(4,4,row*4+col+1,projection="3d")
                    full=index.get((m,parent,j))
                    if full is None:ax.text2D(.15,.5,"NOT REGISTERED",transform=ax.transAxes)
                    else:
                        if free_only:
                            collection=Poly3DCollection(full[12:],facecolor="#6e9cc5",edgecolor="#263746",
                                linewidth=.18,alpha=1.,zsort="average")
                        else:
                            # One collection gives a common depth ordering for C and free faces.
                            collection=Poly3DCollection(full,
                                facecolors=["#dd8557"]*12+["#6e9cc5"]*100,
                                edgecolors=["#7c3421"]*12+["#263746"]*100,
                                linewidths=[.35]*12+[.18]*100,alpha=1.,zsort="average")
                        ax.add_collection3d(collection)
                    ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)
                    ax.set_box_aspect((1,1,1));ax.set_proj_type("ortho")
                    ax.view_init(elev=18,azim=-60,vertical_axis="y");ax.set_axis_off()
                    ax.set_title(f"P{parent} {'free' if free_only else 'full'} | {DISPLAY[m]}",fontsize=10)
        fig.suptitle(f"Fixed seed {seed}, patch {j} ({'old' if j<3 else 'new'}); raw geometry",fontsize=16)
        fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)
        artifacts.append(dict(path=str(path.resolve()),sha256=_sha(path),bytes=path.stat().st_size,patch=j,seed=seed,
            camera=dict(elev=18,azim=-60,vertical_axis="y",projection="ortho"),center=center.tolist(),radius=radius,
            geometry_modified=False,GT_shown=False,visual_review="PENDING_ROOT_REVIEW"))
    return artifacts

def write_report(r,out):
    """Root writes its results JSON later; no geometry scoring or model work."""
    tick=time.perf_counter();out=_path(out);path=out/"EDGE_FACTORIAL_REPORT.md"
    if path.exists():raise FileExistsError(path)
    if len(r.get("scores",[]))!=len(r["generated"]) or r.get("evaluation_ledger",{}).get("status")!="PASS":
        raise ValueError("Completed independent scoring required")
    qpath=out/"quadrature_diagnostic.json"
    if "quadrature_diagnostic" not in r and qpath.is_file():
        r["quadrature_diagnostic"]=json.loads(qpath.read_text(encoding="utf-8"))
    records={}
    for candidate in (out/"cpu_math_tests.json",Path(__file__).with_name("evaluation_test_record.json"),Path(__file__).with_name("report_test_record.json")):
        if candidate.is_file():records[candidate.name]=json.loads(candidate.read_text(encoding="utf-8"))
    r["report_test_records"]=records
    figures=_figures(r,out)
    text=render_report(r,figures)
    with path.open("x",encoding="utf-8") as f:f.write(text)
    r["report"]=dict(path=str(path.resolve()),sha256=_sha(path),bytes=path.stat().st_size,
        figures=figures,seconds=time.perf_counter()-tick,
        scope="Report and raw fixed-view plotting together; zero new metrics/model/OT/gradient calls")
    return r["report"]
