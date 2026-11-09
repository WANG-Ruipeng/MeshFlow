"""Fixed Edge x surface factorial, read-only independent geometry evaluation.

All four endpoints are scored only after the complete registered raw matrix.
The historical full scorer is unchanged; this module adds scalar grouping and
the preregistered finite-edge boundary diagnostic. No model/OT/autograd/repair.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from ..tgeo_m10_evaluation import (
    _array_hash, PARENT_UIDS, score_output, audit_context, cross_binding,
    concentration, scalar_metrics as original_scalars)
from ..a_then_fc_evaluation import DESCRIPTIVE_ONLY
from ..t1_gated_factorial_evaluation import HIGHER_BETTER

MODELS=("L0","Edge_only","L5","JEdge5")
EFFECTS={
    "Edge_off":{"Edge_only":1,"L0":-1},
    "L5_off":{"L5":1,"L0":-1},
    "Edge_on":{"JEdge5":1,"L5":-1},
    "L5_on":{"JEdge5":1,"Edge_only":-1},
    "I":{"JEdge5":1,"Edge_only":-1,"L5":-1,"L0":1}}
THRESHOLDS=(.0005,.001,.002,.005,.01)
DISTANCE_ORDER_TOL_BBOX=1e-10
PROTOCOL=dict(schema="edge_factorial_evaluation_v1",phase="pilot",models=list(MODELS),
    effects=EFFECTS,coverage_thresholds_bbox_fraction=list(THRESHOLDS),
    coverage_comparison="strictly less than threshold",
    edge_distance_surface_tolerance_bbox=DISTANCE_ORDER_TOL_BBOX,
    grouping="Old and new context cells separately; equal cell means within each group",
    medians="Differences formed per parent/patch/seed before any median",
    interaction="Fixed coefficient non-additivity; no equal-total-strength control",
    no_selection=True,no_automatic_default_change=True,
    independent_geometry="Unmodified CF1200/FF4950 and per-context cached CC66",
    scope="Two observed parents, one training seed; new context sets and new noise are not new-object generalization")

def _path(value):
    text=str(value).replace(chr(92),"/")
    if len(text)>2 and text[1]==":" and Path("/mnt").is_dir():
        return Path("/mnt/"+text[0].lower()+text[2:])
    return Path(value)

def _sha(path):
    h=hashlib.sha256()
    with _path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def _write(path,value):
    path=_path(path)
    with path.open("x",encoding="utf-8") as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
    return dict(path=str(path.resolve()),sha256=_sha(path),bytes=path.stat().st_size)

def _units(units):
    out=[]
    for x in units:
        p=int(x.get("parent",x.get("parent_index",-1)))
        j=int(x.get("patch",x.get("patch_index",-1)))
        g=x.get("patch_group","old" if j<3 else "new")
        if p not in (0,1) or j not in range(5) or g!=("old" if j<3 else "new"):
            raise ValueError("Invalid registered unit")
        out.append(dict(x,parent=p,patch=j,patch_group=g,unit_id=f"P{p}_patch{j}"))
    if len({(u["parent"],u["patch"]) for u in out})!=len(out):
        raise ValueError("Duplicate registered units")
    if not {(p,j) for p in (0,1) for j in range(3)} <= {(u["parent"],u["patch"]) for u in out}:
        raise ValueError("All original six cells, including P0/patch2, are required")
    return sorted(out,key=lambda u:(u["parent"],u["patch"]))

def _identity(row):
    return row["model"],int(row["parent"]),int(row["patch"]),int(row["seed"])

def _expected(units,seeds):
    return {(m,u["parent"],u["patch"],int(s)) for m in MODELS for u in units for s in seeds}

def load_bank(path,expected_sha):
    """Same 512-point and Gamma formulas, with the registered actual unit set.

    If one parent's new patch is unavailable, only the scorer's unused cross
    reference slot is temporarily duplicated from the available own target.
    Own/full metrics remain unchanged; that cross row is removed and binding
    is NOT_RUN. No missing context is sampled or included in aggregation.
    """
    from ..native_patch_metrics import prepare_common_reference
    from ..evaluate import sample_mesh_surface
    from ..native_inpainting_metrics import array_sha
    from ..guided_native_geometry_audit import extract_boundary_segments,_sample_boundary
    path=_path(path)
    if _sha(path)!=expected_sha:raise ValueError("Frozen evaluation archive SHA differs")
    with np.load(path,allow_pickle=False) as z:
        registration=json.loads(str(z["registration_json"].item()))
        arrays={k:z[k].copy() for k in z.files if k!="registration_json"}
    if registration.get("status")!="PASS" or set(arrays)!=set(registration["arrays"]):
        raise ValueError("Incomplete frozen registration")
    for k,v in arrays.items():
        rec=registration["arrays"][k]
        if _array_hash(v)!=rec["sha256"] or list(v.shape)!=rec["shape"] or str(v.dtype)!=rec["dtype"]:
            raise ValueError("Frozen array differs: "+k)
        v.setflags(write=False)
    rows=registration["cases"];banks={}
    patches=sorted({int(x["patch_index"]) for x in rows})
    for j in patches:
        cases={};refs=[];parents=sorted({int(x["parent_index"]) for x in rows if int(x["patch_index"])==j})
        for p in parents:
            pre=f"parent{p}_patch{j}_";pp=f"parent{p}_"
            rec=next(x for x in rows if int(x["parent_index"])==p and int(x["patch_index"])==j)
            Y=arrays[pp+"target"];C=arrays[pre+"constraints"];ids=arrays[pre+"source_face_ids"]
            fids=arrays[pre+"free_source_ids"];V=arrays[pp+"source_vertices"];F=arrays[pp+"source_face_vertex_ids"]
            if Y.dtype!=np.float32 or Y.shape!=(112,9) or C.dtype!=np.float32 or C.shape!=(12,3,3):
                raise ValueError("N112/K12 FP32 shape differs")
            if len(set(ids.tolist()))!=12 or fids.tolist()!=[i for i in range(112) if i not in ids]:
                raise ValueError("Known/free source decomposition differs")
            if V[F].tobytes()!=Y.tobytes() or Y[ids].tobytes()!=C.tobytes() or Y[fids].tobytes()!=arrays[pre+"free_target"].tobytes():
                raise ValueError("Trusted source geometry differs")
            cases[p,12]=dict(parent_index=p,parent_uid=PARENT_UIDS[p],patch_index=j,K=12,N=112,
                constraints=C,full_target=Y,free_target=arrays[pre+"free_target"],
                source_face_ids=ids,free_source_ids=fids,pre_ot_free=arrays[pre+"pre_ot_free"],
                bbox_diagonal=float(rec["bbox_diagonal"]))
            free=arrays[pre+"free_target"].reshape(100,3,3).astype(np.float64)
            rng=np.random.RandomState(9911);sample_mesh_surface(free,512,rng)
            target=sample_mesh_surface(free,512,rng)
            if target.tobytes()!=arrays[pre+"pairwise_target_points"].tobytes():
                raise ValueError("Original pairwise512 reference differs")
            segments,_=extract_boundary_segments(C)
            points,sids,weights,_=_sample_boundary(segments,9919,512)
            for key,value in (("boundary_segments",segments),("boundary_points",points),
                ("boundary_point_segment_ids",sids),("boundary_weights",weights)):
                saved=arrays[pre+key]
                if value.dtype!=saved.dtype or value.shape!=saved.shape or value.tobytes()!=saved.tobytes():
                    raise ValueError("Original Gamma reference differs: "+pre+key)
            refs.append(dict(parent=p,patch=j,status="PASS",pairwise_target_point_sha256=array_sha(target),
                boundary_point_count=len(points),case_registration=rec))
        first=cases[parents[0],12]
        score_cases={p:cases.get((p,12),first) for p in (0,1)}
        common=prepare_common_reference([score_cases[p]["free_target"] for p in (0,1)],
            [score_cases[p]["bbox_diagonal"] for p in (0,1)])
        for p in parents:
            frozen=arrays[f"parent{p}_patch{j}_coverage_points"]
            if common["references"][p]["points"].tobytes()!=frozen.tobytes():
                raise ValueError("Original coverage512 reference differs")
            next(x for x in refs if x["parent"]==p)["coverage_point_sha256"]=array_sha(frozen)
        banks[j]=dict(cases_by_key={(p,12):score_cases[p] for p in (0,1)},
            actual_parents=parents,arrays=arrays,registration=registration,reference=common,
            reference_audit=refs,source_archive=dict(path=str(path.resolve()),sha256=expected_sha),
            missing_cross_reference="NOT_RUN" if len(parents)<2 else None)
    return banks

def _input_audit(bank,patch,ledger):
    """Copied original topology audit, restricted to actual registered parents."""
    from ..t1_native_confirmation_topology import audit_mesh,exact_index,EXACT_POLICY
    rows=[]
    for parent in bank["actual_parents"]:
        a=bank["arrays"];case=bank["cases_by_key"][parent,12]
        vertices=a[f"parent{parent}_source_vertices"];faces=a[f"parent{parent}_source_face_vertex_ids"]
        ids=case["source_face_ids"];C=case["constraints"]
        full=audit_mesh(vertices,faces);sourceC=audit_mesh(vertices,faces[ids])
        cv,cf=exact_index(C);exactC=audit_mesh(cv,cf)
        links={x["vertex_id"]:x for x in full["vertex_links"]["all"]}
        sub={x["vertex_id"]:x for x in sourceC["vertex_links"]["all"]}
        anomalies=[dict(source_vertex_id=v,C_link=sub[v],full_GT_link=links[v],
            full_GT_repairs_subpatch_link=links[v]["classification"]!="NONMANIFOLD_LINK")
            for v in sourceC["vertex_links"]["nonmanifold_vertex_ids"]]
        rows.append(dict(parent=parent,patch=patch,parent_uid=case["parent_uid"],
            status="PASS_IDENTITY_AND_DECOMPOSITION",full_archive_source_indices=full,
            C_source_archive_winding=sourceC,C_exact_archive_winding=exactC,
            C_link_anomalies_vs_full_GT=anomalies,exact_index_policy=EXACT_POLICY,
            source_face_ids=ids.tolist(),free_source_ids=case["free_source_ids"].tolist(),
            full_GT_role="Read-only source topology/score; never model input"))
    n=len(rows)
    for k,v in dict(input_parent_patch_audits=n,input_indexed_mesh_audits=3*n,input_exact_index_calls=n).items():
        ledger[k]=ledger.get(k,0)+v
    return dict(status="PASS",parents=rows,patch=patch)

def _metric_reference_checks(metrics,bank,parent,patch):
    from ..native_inpainting_metrics import array_sha
    from ..guided_native_geometry_audit import _sha_array
    extra=metrics["t1_native_confirmation"]
    if metrics["K"]!=12 or metrics["total_faces"]!=112 or metrics["free_faces"]!=100 or extra["C_PRESERVED"]!="PASS":
        raise ValueError("C/K/N/free score identity differs")
    if extra["nan_coordinate_count"] or extra["inf_coordinate_count"] or not extra["all_coordinates_finite"]:
        raise ValueError("Nonfinite recorded geometry")
    for target in bank["actual_parents"]:
        pre=f"parent{target}_patch{patch}_"
        if metrics["remainder"]["cross"][target]["target_sample_sha256"]!=array_sha(bank["arrays"][pre+"pairwise_target_points"]):
            raise ValueError("Pairwise512 target differs")
        if metrics["common_unknown_R12"]["cross"][target]["target_point_sha256"]!=array_sha(bank["arrays"][pre+"coverage_points"]):
            raise ValueError("Coverage512 target differs")
    pre=f"parent{parent}_patch{patch}_";boundary=metrics["guided_geometry_audit"]["boundary"]
    if boundary["point_sha256"]!=_sha_array(bank["arrays"][pre+"boundary_points"]) or boundary["weight_sha256"]!=_sha_array(bank["arrays"][pre+"boundary_weights"]):
        raise ValueError("Gamma frozen query identity differs")
    for scope,n in (("CF",1200),("FF",4950),("CC",66)):
        if extra["detector_views"][scope]["verified"]["tested_pairs"]!=n:
            raise ValueError("Independent pair denominator differs: "+scope)

def _raw(row,bank,parent):
    path=_path(row["raw_path"])
    if _sha(path)!=row["raw_sha256"]:raise ValueError("Raw file identity differs")
    with np.load(path,allow_pickle=False) as z:full=z["output"].copy();states=z["path"].copy()
    if full.dtype!=np.float32 or full.size!=112*9 or states.dtype!=np.float32 or states.size!=51*112*9:
        raise ValueError("Raw FP32 N112/51-state shape differs")
    full=full.reshape(112,3,3);states=states.reshape(51,112,3,3);C=bank["cases_by_key"][parent,12]["constraints"]
    if not np.isfinite(states).all() or states[-1].tobytes()!=full.tobytes() or any(x[:12].tobytes()!=C.tobytes() for x in states):
        raise ValueError("Nonfinite, changed C, or endpoint differs")
    return full

def _boundary_extension(full,bank,parent,patch,metrics,ledger):
    from .edge_metric import gamma_to_free_edges
    from ..guided_native_geometry_audit import point_to_triangle_surface_squared
    pre=f"parent{parent}_patch{patch}_";points=bank["arrays"][pre+"boundary_points"]
    weights=bank["arrays"][pre+"boundary_weights"];L=bank["cases_by_key"][parent,12]["bbox_diagonal"]
    free=full[12:].astype(np.float64)
    squared=point_to_triangle_surface_squared(points,free,L)
    ledger["additional_fixed_Gamma_surface_distance_calls"]=ledger.get("additional_fixed_Gamma_surface_distance_calls",0)+1
    if squared is None:raise ValueError("Gamma surface reference unavailable")
    d=np.sqrt(squared);original=metrics["guided_geometry_audit"]["boundary"]
    rms=100*float(np.sqrt(np.sum(weights*squared)))/L
    order=np.argsort(d,kind="stable");cdf=np.cumsum(weights[order])
    p95=100*float(d[order[min(int(np.searchsorted(cdf,.95*cdf[-1],side="left")),len(order)-1)]])/L
    if not np.isclose(rms,original["rms_bbox_pct"],rtol=1e-12,atol=1e-12) or not np.isclose(p95,original["p95_bbox_pct"],rtol=1e-12,atol=1e-12):
        raise ValueError("Extra thresholds changed original Gamma definition")
    coverage={str(t):float(weights[d<t*L].sum()) for t in THRESHOLDS}
    if coverage["0.001"]!=original["coverage_fraction"]:
        raise ValueError("Original strict Gamma coverage differs")
    edge=gamma_to_free_edges(free,points,weights,L,ledger=ledger)
    distances=np.asarray(edge["per_point_distances"],dtype=np.float64)
    if distances.shape!=d.shape:raise ValueError("Edge and surface query count differs")
    violations=distances+DISTANCE_ORDER_TOL_BBOX*L<d
    if bool(violations.any()):raise ValueError("Point-to-edge distance less than surface beyond fixed tolerance")
    return dict(schema="edge_factorial_boundary_extension_v1",
        surface=dict(coverage_by_bbox_fraction=coverage,per_point_distances=d.tolist()),
        edges=edge,ordering=dict(status="PASS",violations=int(violations.sum()),
            tolerance_bbox_fraction=DISTANCE_ORDER_TOL_BBOX,
            minimum_edge_minus_surface_bbox=float((distances-d).min()/L)),
        thresholds_bbox_fraction=list(THRESHOLDS),no_correspondences_used=True,
        geometry_modified=False,topological_connection_rate="NOT_DEFINED")

def scalar_metrics(metrics):
    values=original_scalars(metrics);ext=metrics.get("edge_factorial_boundary",{})
    if ext:
        edge=ext["edges"]
        values.update(Gamma_to_free_edges_RMS_bbox_pct=edge["RMS_bbox_pct"],
            Gamma_to_free_edges_p95_bbox_pct=edge["p95_bbox_pct"],
            Gamma_to_free_edges_zero_length_edge_count=edge["zero_length_edge_count"])
        for name,source in (("Gamma_surface",ext["surface"]),("Gamma_to_free_edges",edge)):
            for t in THRESHOLDS:
                values[name+"_coverage_"+str(t)+"_fraction"]=source["coverage_by_bbox_fraction"][str(t)]
    return values



def _direction(key):
    if key in DESCRIPTIVE_ONLY:return None
    return 1 if key in HIGHER_BETTER or key in ("Delta0","Delta1","B") or (key.startswith("Gamma_") and "_coverage_" in key) else -1

def _stat(rows,key,*,paired=False):
    from ..t1_native_confirmation_evaluation import summarize_values
    direction=_direction(key)
    out=summarize_values(rows,key,higher_better=direction==1)
    valid=[x for x in rows if x.get(key) is not None and np.isfinite(float(x[key]))]
    if direction is None:
        out.update(worst=None,higher_is_better=None,quality_direction="DESCRIPTIVE_ONLY")
    elif valid:
        worst=min(valid,key=lambda x:(direction*float(x[key]),int(x.get("parent",-1)),int(x.get("patch",-1)),int(x["seed"])))
        out["worst"]={k:worst.get(k) for k in ("parent","patch","unit_id","seed")}
        out["worst"]["value"]=float(worst[key])
    a=[float(x[key]) for x in valid]
    out["sign_counts"]=dict(negative=sum(v<0 for v in a),zero=sum(v==0 for v in a),positive=sum(v>0 for v in a))
    out["improvement_counts"]=(dict(better=sum(direction*v>0 for v in a),tie=sum(v==0 for v in a),
        worse=sum(direction*v<0 for v in a),available=len(a)) if paired and direction is not None
        else dict(better=None,tie=None,worse=None,available=len(a)))
    return out

def _means(rows,keys):
    return {k:float(np.mean([x[k] for x in rows])) if rows and all(x.get(k) is not None and np.isfinite(float(x[k])) for x in rows) else None for k in keys}

def _scope(rows,units,seeds,keys,*,paired=False):
    """All grouping below is scalar arithmetic; each registered cell is equal."""
    if not units:return dict(status="NOT_RUN_NO_REGISTERED_UNITS",cell_count=0)
    ids={u["unit_id"] for u in units};rows=[x for x in rows if x["unit_id"] in ids]
    cells={}
    for u in units:
        values=[x for x in rows if x["unit_id"]==u["unit_id"]]
        if {int(x["seed"]) for x in values}!=set(seeds) or len(values)!=len(seeds):
            raise ValueError("Incomplete cell for scalar grouping")
        cells[u["unit_id"]]=dict(parent=u["parent"],patch=u["patch"],patch_group=u["patch_group"],
            per_seed=values,metrics={k:_stat(values,k,paired=paired) for k in keys})
    means=[dict(unit_id=i,**{k:c["metrics"][k]["mean"] for k in keys}) for i,c in cells.items()]
    macro=_means(means,keys)
    parents={}
    for p in sorted({u["parent"] for u in units}):
        pr=[x for x in rows if x["parent"]==p]
        pmeans=[v for v in means if cells[v["unit_id"]]["parent"]==p]
        parents[str(p)]=dict(cell_count=len(pmeans),metrics={k:_stat(pr,k,paired=paired) for k in keys},
            macro_mean=_means(pmeans,keys))
    seed_rows=[]
    for seed in seeds:
        values=[x for x in rows if x["seed"]==seed]
        seed_rows.append(dict(seed=seed,cell_count=len(values),**_means(values,keys)))
    loso={}
    for omitted in seeds:
        keep=[x for x in rows if x["seed"]!=omitted]
        cell_means={u["unit_id"]:_means([x for x in keep if x["unit_id"]==u["unit_id"]],keys) for u in units}
        loso[str(omitted)]=dict(omitted_seed=omitted,seeds=[s for s in seeds if s!=omitted],
            macro_mean=_means(list(cell_means.values()),keys),cell_means=cell_means,
            parents={str(p):_means([cell_means[u["unit_id"]] for u in units if u["parent"]==p],keys)
                for p in sorted({u["parent"] for u in units})})
    return dict(status="PASS",cell_count=len(units),output_count=len(rows),cells=cells,
        metrics={k:_stat(rows,k,paired=paired) for k in keys},macro_mean=macro,
        parents=parents,seeds=seed_rows,
        seed_macro_metrics={k:_stat(seed_rows,k,paired=paired) for k in keys},
        leave_one_seed_out=loso,weighting="Mean seeds in each cell, then equal cells; all four seeds remain primary")

def aggregate_results(rows,units,seeds):
    """Complete fixed four-model matrix, pairing before every statistic."""
    rows=list(rows);units=_units(units);seeds=tuple(map(int,seeds))
    if len(seeds)!=4 or len(set(seeds))!=4:raise ValueError("Require four registered seeds")
    idx={_identity(x):x for x in rows}
    if len(rows)!=len(idx) or set(idx)!=_expected(units,seeds):raise ValueError("Incomplete/duplicate registered matrix")
    if any(x.get("phase")!="pilot" or x.get("K")!=12 for x in rows):raise ValueError("Phase/K mismatch")
    names=sorted(next(iter(rows))["scalars"] if "scalars" in rows[0] else scalar_metrics(rows[0]["metrics"]))
    values={}
    for model in MODELS:
        rr=[]
        for u in units:
            for seed in seeds:
                row=idx[model,u["parent"],u["patch"],seed]
                sc=row.get("scalars")
                if sc is None:sc=scalar_metrics(row["metrics"])
                if set(sc)!=set(names):raise ValueError("Inconsistent scalar metric keys")
                rr.append(dict(parent=u["parent"],patch=u["patch"],unit_id=u["unit_id"],
                    patch_group=u["patch_group"],seed=seed,**sc))
        values[model]=rr
    effects={}
    for effect,coeff in EFFECTS.items():
        rr=[]
        for n,reference in enumerate(values["L0"]):
            v={k:reference[k] for k in ("parent","patch","unit_id","patch_group","seed")}
            for key in names:
                parts=[(c,values[m][n][key]) for m,c in coeff.items()]
                v[key]=sum(c*x for c,x in parts) if all(x is not None for _,x in parts) else None
            rr.append(v)
        effects[effect]=rr
    result=dict(schema="edge_factorial_summary_v1",phase="pilot",models=list(MODELS),seeds=list(seeds),
        units=units,metric_names=names,scopes={},bindings={},effect_coefficients=EFFECTS,
        protocol=PROTOCOL,automatic_winner="NOT_DEFINED",model_calls=0,geometry_calls=0)
    for group in ("old","new"):
        chosen=[u for u in units if u["patch_group"]==group]
        result["scopes"][group]=dict(units=chosen,
            models={m:_scope(values[m],chosen,seeds,names) for m in MODELS},
            effects={e:_scope(v,chosen,seeds,names,paired=True) for e,v in effects.items()})
    for j in sorted({u["patch"] for u in units}):
        parents={u["parent"] for u in units if u["patch"]==j}
        if parents!={0,1}:
            result["bindings"][str(j)]=dict(status="NOT_RUN_MISSING_COUNTERPART",parents=sorted(parents))
            continue
        block=dict(status="PASS",patch_group="old" if j<3 else "new",models={},effects={})
        for m in MODELS:
            b=[]
            for seed in seeds:
                a=idx[m,0,j,seed];c=idx[m,1,j,seed]
                if "metrics" not in a or "metrics" not in c:
                    raise ValueError("Binding requires full saved metrics during aggregation")
                b.append(dict(seed=seed,**cross_binding(a["metrics"],c["metrics"])))
            block["models"][m]=dict(per_seed=b,metrics={k:_stat(b,k) for k in ("Delta0","Delta1","B")},
                both_positive_count=sum(x["both_directions_positive"] for x in b),concentration=concentration(b))
        for e,coeff in EFFECTS.items():
            b=[dict(seed=seed,**{k:sum(c*block["models"][m]["per_seed"][n][k] for m,c in coeff.items())
                for k in ("Delta0","Delta1","B")}) for n,seed in enumerate(seeds)]
            block["effects"][e]=dict(per_seed=b,metrics={k:_stat(b,k,paired=True) for k in ("Delta0","Delta1","B")},
                concentration=concentration(b))
        result["bindings"][str(j)]=block
    json.dumps(result,allow_nan=False)
    return result

def _csv(path,fields,rows):
    with _path(path).open("x",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    return dict(path=str(_path(path).resolve()),bytes=_path(path).stat().st_size,sha256=_sha(path),rows=len(rows))

def export_tables(r,out):
    out=_path(out);summary=r["summaries"]["pilot"];names=summary["metric_names"]
    identity=["phase","model","parent","patch","patch_group","unit_id","seed","K","raw_path","metrics_path"]
    rows=[dict({k:x.get(k) for k in identity},**x["scalars"]) for x in r["scores"]]
    records=dict(per_output=_csv(out/"compact_per_output.csv",identity+names,rows))
    effect_rows=[]
    for group,scope in summary["scopes"].items():
        for effect,block in scope["effects"].items():
            if block.get("status")!="PASS":continue
            for cell in block["cells"].values():
                for row in cell["per_seed"]:effect_rows.append(dict(effect=effect,**row))
    records["paired_factorial"]=_csv(out/"compact_per_seed_effects.csv",
        ["effect","parent","patch","unit_id","patch_group","seed"]+names,effect_rows)
    return records

def evaluate(r,out):
    """Root-only offline entry. Fail closed before any audit if raw matrix partial."""
    tick=time.perf_counter();out=_path(out)
    if r.get("summaries",{}).get("pilot"):raise RuntimeError("No automatic audit repeat")
    config=r["protocol"]["evaluation"];units=_units(config["units"]);seeds=tuple(map(int,config["seeds"]))
    if tuple(config["models"])!=MODELS or len(seeds)!=4 or len(set(seeds))!=4:
        raise ValueError("Frozen models/seeds differ")
    rows=list(r["generated"]);expected=_expected(units,seeds)
    if len(rows)!=len(expected) or {_identity(x) for x in rows}!=expected:raise ValueError("All registered raw outputs required before geometry")
    if any(x.get("phase")!="pilot" or x.get("K")!=12 for x in rows):raise ValueError("Unexpected output phase/K")
    phase=r["sampling_phases"]["pilot"]
    for m in MODELS:
        if phase["models_audit"][m]["status"]!="PASS":raise ValueError("Frozen model sampling did not pass: "+m)
    rec=r["assets"]["evaluation_inputs"];banks=load_bank(rec["path"],rec["sha256"])
    actual={(p,j) for j,b in banks.items() for p in b["actual_parents"]}
    if actual!={(u["parent"],u["patch"]) for u in units}:raise ValueError("Registered cases and formal units differ")
    directory=out/"audit";directory.mkdir(exist_ok=False)
    ledger=dict(model_calls=0,OT_calls=0,geometry_autograd_calls=0,optimizer_updates=0,repairs=0)
    inputs=[]
    for j,bank in banks.items():
        bank["input_topology"]=_input_audit(bank,j,ledger)
        bank["context_audits"]={p:audit_context(bank["cases_by_key"][p,12]["constraints"],
            bank["cases_by_key"][p,12]["bbox_diagonal"],ledger) for p in bank["actual_parents"]}
        inputs.append(_write(directory/f"input_patch{j}.json",dict(input_topology=bank["input_topology"],
            context_audits=bank["context_audits"],reference_audit=bank["reference_audit"])))
    full_rows=[];compact=[]
    for row in rows:
        started=time.perf_counter();m,p,j,s=_identity(row);bank=banks[j]
        full=_raw(row,bank,p);before=full.tobytes();metrics=score_output(full,bank,p,ledger=ledger)
        _metric_reference_checks(metrics,bank,p,j)
        metrics["edge_factorial_boundary"]=_boundary_extension(full,bank,p,j,metrics,ledger)
        for missing in set((0,1))-set(bank["actual_parents"]):
            metrics["remainder"]["cross"][missing]=None
            metrics["common_unknown_R12"]["cross"][missing]=None
            metrics["cross_binding_status"]="NOT_RUN_MISSING_COUNTERPART"
        if before!=full.tobytes():raise AssertionError("Read-only evaluator modified raw")
        values=scalar_metrics(metrics)
        identity=dict(phase="pilot",model=m,parent=p,patch=j,patch_group="old" if j<3 else "new",
            unit_id=f"P{p}_patch{j}",seed=s,K=12,raw_path=row["raw_path"],raw_sha256=row["raw_sha256"])
        artifact=_write(directory/f"{m}_P{p}_patch{j}_seed{s}.json",
            dict(identity,metrics=metrics,scoring_seconds=time.perf_counter()-started))
        full_rows.append(dict(identity,metrics=metrics,scalars=values))
        compact.append(dict(identity,scalars=values,metrics_path=artifact["path"],
            metrics_sha256=artifact["sha256"],metrics_bytes=artifact["bytes"],
            scoring_seconds=time.perf_counter()-started))
        print(json.dumps(dict(event="edge_factorial_score",model=m,parent=p,patch=j,seed=s,
            completed=len(compact),total=len(rows))),flush=True)
    r["scores"]=compact;r.setdefault("summaries",{})["pilot"]=aggregate_results(full_rows,units,seeds)
    r["evaluation_ledger"]=dict(status="PASS",counts=ledger,seconds=time.perf_counter()-tick,
        input_audits=inputs,scored_outputs=len(rows),immutable_context_audits=len(units),
        complete_pairs_per_output=dict(CF=1200,FF=4950,CC_cached=66),historical_scores_reused=0,
        protocol=PROTOCOL)
    r["evaluation_ledger"]["tables"]=export_tables(r,out)
    return r["evaluation_ledger"]

def run_cpu_tests():
    """Synthetic scalar arithmetic only; no raw files, geometry or torch work."""
    seeds=(10201,10202,10203,10204)
    units=[dict(parent=p,patch=j,patch_group="old" if j<3 else "new") for p in (0,1) for j in range(5)]
    key="remainder_symmetric_RMS_bbox_pct";coverage="Gamma_to_free_edges_coverage_0.001_fraction"
    shifts={"L0":0.,"Edge_only":-2.,"L5":-1.,"JEdge5":-4.}
    rows=[]
    for m in MODELS:
        for u in units:
            for n,s in enumerate(seeds):
                value=10+u["parent"]+u["patch"]+n*.25+shifts[m]
                cross=[dict(rms_bbox_pct=value+(q!=u["parent"])*2) for q in (0,1)]
                metrics=dict(K=12,variant="raw",parent_index=u["parent"],
                    remainder=dict(cross=[dict(symmetric_rms_bbox_pct=x["rms_bbox_pct"]) for x in cross]),
                    common_unknown_R12=dict(cross=cross,own=cross[u["parent"]]))
                rows.append(dict(phase="pilot",model=m,parent=u["parent"],patch=u["patch"],seed=s,K=12,
                    metrics=metrics,scalars={key:value,coverage:.5-.01*shifts[m],"free_total_surface_area":20.}))
    summary=aggregate_results(rows,units,seeds)
    assert len(rows)==160
    assert summary["scopes"]["old"]["models"]["L0"]["cell_count"]==6
    assert summary["scopes"]["new"]["models"]["L0"]["cell_count"]==4
    assert np.isclose(summary["scopes"]["new"]["effects"]["I"]["macro_mean"][key],-1.)
    assert summary["scopes"]["old"]["effects"]["Edge_off"]["metrics"][key]["improvement_counts"]["better"]==24
    assert summary["scopes"]["new"]["effects"]["Edge_off"]["metrics"][coverage]["improvement_counts"]["better"]==16
    assert len(summary["scopes"]["new"]["effects"]["I"]["leave_one_seed_out"])==4
    assert all(np.isclose(v["macro_mean"][key],-1.) for v in summary["scopes"]["new"]["effects"]["I"]["leave_one_seed_out"].values())
    assert summary["scopes"]["new"]["effects"]["Edge_off"]["metrics"]["free_total_surface_area"]["worst"] is None
    assert summary["bindings"]["4"]["models"]["L0"]["both_positive_count"]==4
    for bad in (rows+[rows[0]],rows[:-1]):
        try:aggregate_results(bad,units,seeds)
        except ValueError:pass
        else:raise AssertionError("Invalid matrix accepted")
    a=[0.,0.,100.,100.];b=[0.,100.,100.,100.]
    differences=[dict(seed=s,value=y-x) for s,x,y in zip(seeds,a,b)]
    assert _stat(differences,"value",paired=True)["median"]==0. and np.median(b)-np.median(a)==50.
    fewer=[u for u in units if not (u["parent"]==1 and u["patch"]==4)]
    partial=[x for x in rows if not (x["parent"]==1 and x["patch"]==4)]
    short=aggregate_results(partial,fewer,seeds)
    assert short["scopes"]["new"]["models"]["L0"]["cell_count"]==3
    assert short["bindings"]["4"]["status"]=="NOT_RUN_MISSING_COUNTERPART"
    return dict(status="PASS",checks=15,model_calls=0,geometry_calls=0,OT_calls=0,autograd_calls=0,
        scope="Scalar matrix, pairing-before-median, five factorial formulas, equal cells, coverage sign, LOSO, incomplete new-unit handling")
