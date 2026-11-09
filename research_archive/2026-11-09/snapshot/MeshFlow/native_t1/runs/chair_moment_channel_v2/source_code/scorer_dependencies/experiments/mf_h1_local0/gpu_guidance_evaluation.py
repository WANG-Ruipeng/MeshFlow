"""Independent offline Raw/+W evaluation for the fixed GPU-guidance pilot.

The original CPU endpoint scorer is reused unchanged. This module never calls a
model or a GPU energy, modifies coordinates, welds, or prunes triangles.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import itertools
import hashlib
import json
import math
import time

import numpy as np

from .guided_native_geometry_audit import score_endpoint, pair_scan
from .native_patch_metrics import patch_binding_four_grid


Q_CUT=10.
AREA_FLOOR=1e-10
Q_FLOOR=1e-12
NEAR_AREA=1e-12
PROPER="proper_transverse_penetration"
COPLANAR="coplanar_positive_area_overlap"
ILLEGAL=(PROPER,COPLANAR)


def _tri(value):
    value=np.asarray(value)
    if value.ndim==2 and value.shape[-1]==9:value=value.reshape(-1,3,3)
    if value.ndim!=3 or value.shape[1:]!=(3,3):raise ValueError("Expected triangle coordinates")
    return value


def _distribution(values):
    values=np.asarray(values,dtype=np.float64).reshape(-1)
    finite=values[np.isfinite(values)]
    return dict(count=len(values),finite_count=len(finite),positive_inf_count=int(np.isposinf(values).sum()),
        negative_inf_count=int(np.isneginf(values).sum()),nan_count=int(np.isnan(values).sum()),
        mean=float(finite.mean()) if len(finite) else None,
        median=float(np.median(finite)) if len(finite) else None,
        minimum=float(finite.min()) if len(finite) else None,
        maximum=float(finite.max()) if len(finite) else None,
        p05=float(np.quantile(finite,.05)) if len(finite) else None,
        p95=float(np.quantile(finite,.95)) if len(finite) else None,
        finite_subset_statistics=True)


def shape_quality(triangles,L):
    """Actual shape q plus separately identified numerical floor behavior."""
    tri=_tri(triangles).astype(np.float64,copy=False)
    L=float(L)
    if not np.isfinite(tri).all() or not math.isfinite(L) or L<=0:
        raise ValueError("Require finite coordinates and positive existing bbox L")
    normalized=tri/L
    edges=np.stack([normalized[:,(i+1)%3]-normalized[:,i] for i in range(3)],axis=1)
    squared=(edges*edges).sum(-1)
    area=.5*np.linalg.norm(np.cross(normalized[:,1]-normalized[:,0],normalized[:,2]-normalized[:,0]),axis=-1)
    s=squared.sum(-1)
    q_guarded=np.maximum(s/(4*np.sqrt(3)*np.maximum(area,AREA_FLOOR)),Q_FLOOR)
    q_actual=np.divide(s,4*np.sqrt(3)*area,out=np.full_like(s,np.inf),where=area>0)
    q_actual[(area==0)&(s==0)]=np.nan
    angles=[]
    for corner in range(3):
        a=normalized[:,(corner+1)%3]-normalized[:,corner]
        b=normalized[:,(corner+2)%3]-normalized[:,corner]
        den=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
        cosine=np.divide((a*b).sum(-1),den,out=np.ones_like(den),where=den>0)
        angles.append(np.degrees(np.arccos(np.clip(cosine,-1,1))))
    angles=np.stack(angles,axis=-1)
    per_face=[]
    for index in range(len(tri)):
        q=float(q_actual[index])
        per_face.append(dict(face=index,area=float(area[index]*L*L),area_normalized=float(area[index]),
            area_exact_zero=bool(area[index]==0),area_near_degenerate=bool(area[index]<=NEAR_AREA),
            area_floor_triggered=bool(area[index]<AREA_FLOOR),
            q_guarded=float(q_guarded[index]),q_actual=q if math.isfinite(q) else None,
            q_actual_status="FINITE" if math.isfinite(q) else "INFINITE_COLLINEAR" if math.isinf(q) else "UNDEFINED_COINCIDENT",
            q_actual_gt10=bool(q>Q_CUT),q_guarded_gt10=bool(q_guarded[index]>Q_CUT),
            minimum_angle_degrees=float(angles[index].min()),minimum_angle_lt5=bool(angles[index].min()<5)))
    energy=.5*np.maximum(np.log(q_guarded/Q_CUT),0)**2
    return dict(schema="gpu_guidance_shape_quality_v1",faces=len(tri),bbox_diagonal=L,
        q_definition="sum squared edge lengths/(4*sqrt(3)*triangle area)",q_cut=Q_CUT,
        normalized_area_floor=AREA_FLOOR,numerical_q_floor=Q_FLOOR,near_degenerate_area_threshold=NEAR_AREA,
        per_face=per_face,area_normalized=_distribution(area),area_original=_distribution(area*L*L),
        total_area=float(area.sum()*L*L),q_actual=_distribution(q_actual),q_guarded=_distribution(q_guarded),
        q_actual_gt10_count=int((q_actual>Q_CUT).sum()),q_guarded_gt10_count=int((q_guarded>Q_CUT).sum()),
        q_actual_gt10_fraction=float((q_actual>Q_CUT).mean()) if len(tri) else None,
        q_guarded_gt10_fraction=float((q_guarded>Q_CUT).mean()) if len(tri) else None,
        exact_zero_area_count=int((area==0).sum()),near_degenerate_area_count=int((area<=NEAR_AREA).sum()),
        area_floor_trigger_count=int((area<AREA_FLOOR).sum()),
        min_angle_lt5_count=int((angles.min(-1)<5).sum()),
        min_angle_lt5_fraction=float((angles.min(-1)<5).mean()) if len(tri) else None,
        angle_p05_degrees=float(np.quantile(angles,.05)) if len(tri) else None,
        guarded_EA_CPU_formula_reference=float(energy.mean()) if len(tri) else None,
        EA_scope="Read-only NumPy formula reference; not an extra GPU energy or gradient evaluation",
        caveat="Finite guarded q/energy does not imply nondegenerate geometry; floor triggers are not all degeneracies")


def _detector_view(scan,which):
    """Recover constituent labels from saved verified-pair diagnostics only."""
    counts=Counter();contacts=[]
    for row in scan.get("contacts",[]):
        detector=row.get("detector",{})
        classification=row["classification"] if which=="verified" else detector.get(which,{}).get("classification",row["classification"])
        counts[classification]+=1
        if classification!="disjoint":contacts.append(dict(pair=list(row["pair"]),classification=classification))
    counts["disjoint"]+=int(scan["tested_pairs"])-len(scan.get("contacts",[]))
    sets={kind:[row["pair"] for row in contacts if row["classification"]==kind] for kind in ILLEGAL}
    unresolved=[row["pair"] for row in contacts if "unresolved" in row["classification"].lower() or "degenerate" in row["classification"].lower()]
    return dict(detector=which,tested_pairs=scan["tested_pairs"],classification_counts=dict(counts),
        proper_pairs=sets[PROPER],coplanar_overlap_pairs=sets[COPLANAR],
        proper_pair_count=len(sets[PROPER]),coplanar_pair_count=len(sets[COPLANAR]),
        illegal_pair_count=len(sets[PROPER])+len(sets[COPLANAR]),unresolved_pairs=unresolved,
        unresolved_count=len(unresolved),contacts=contacts,
        source="Derived from existing complete endpoint scan; no new geometry tests")


def _illegal_quality(quality,face_ids):
    ids=sorted(set(int(index) for index in face_ids))
    chosen=[quality["per_face"][index] for index in ids]
    q=np.array([row["q_actual"] if row["q_actual"] is not None else
                np.inf if row["q_actual_status"]=="INFINITE_COLLINEAR" else np.nan for row in chosen])
    return dict(face_ids=ids,count=len(ids),area_normalized=_distribution([row["area_normalized"] for row in chosen]),
        q_actual=_distribution(q),q_guarded=_distribution([row["q_guarded"] for row in chosen]),
        area_floor_trigger_count=sum(row["area_floor_triggered"] for row in chosen),
        exact_zero_area_count=sum(row["area_exact_zero"] for row in chosen),
        near_degenerate_count=sum(row["area_near_degenerate"] for row in chosen),
        q_actual_gt10_count=sum(row["q_actual_gt10"] for row in chosen),
        min_angle_lt5_count=sum(row["minimum_angle_lt5"] for row in chosen),
        q_at_most10_and_not_near_degenerate_face_ids=[row["face"] for row in chosen if row["q_actual"] is not None and row["q_actual"]<=10 and not row["area_near_degenerate"]],
        all_illegal_faces_are_q_gt10=None if not chosen else all(row["q_actual_gt10"] for row in chosen),
        all_illegal_faces_below_area_floor=None if not chosen else all(row["area_floor_triggered"] for row in chosen),
        interpretation="Distribution only; no learned small-face threshold, no pruning, and no guarantee deletion would be safe")


def audit_context(C,L,ledger=None):
    """One static66-pair C/C scan per parent, reusable because C is immutable."""
    C=_tri(C)
    if C.shape!=(12,3,3) or C.dtype!=np.float32:
        raise ValueError("Require original FP32 C12")
    scan=pair_scan(C,C,L,ledger=ledger,self_pairs=True,independent=True)
    views={which:_detector_view(scan,which) for which in ("primary","independent","verified")}
    scan["scope"]="All66 unordered fixed C/C pairs; static input-only audit"
    if ledger is not None:ledger["static_context_audits"]=ledger.get("static_context_audits",0)+1
    return dict(schema="gpu_guidance_static_context_v1",context_sha256=hashlib.sha256(C.tobytes()).hexdigest(),
        bbox_diagonal=float(L),pair_scan=scan,detector_views=views,computed_once=True,
        model_calls=0,geometry_autograd_calls=0,optimizer_updates=0)


def score_variant(tokens,bank,parent,ledger=None,topology=None):
    """Score Raw or W coordinates verbatim with identical independent CPU rules.

    topology may be an indexed audit or {indexing: construction audit, topology:
    indexed audit}. W and indexing are performed elsewhere, never in this call.
    """
    tick=time.perf_counter()
    full=_tri(tokens)
    case=bank["cases_by_key"][(int(parent),12)]
    metrics=score_endpoint(full,bank,parent,ledger=ledger)
    L=float(case["bbox_diagonal"])
    quality=shape_quality(full[12:],L)
    scans=metrics["guided_geometry_audit"]
    detectors={scope:{which:_detector_view(scans[key],which) for which in ("primary","independent","verified")}
               for scope,key in (("CF","C_free"),("FF","free_free"))}
    legacy=metrics["seam"]["context_free_intersections"]
    old={(x["context_face"],x["free_face"]) for x in legacy["contacts"] if x["classification"]==PROPER}
    if old!={tuple(x) for x in detectors["CF"]["primary"]["proper_pairs"]}:
        raise AssertionError("Primary CF reconstruction differs from historical classifier")
    groups={}
    for which in ("primary","independent","verified"):
        cf,ff=detectors["CF"][which],detectors["FF"][which]
        cf_faces={pair[1] for key in ("proper_pairs","coplanar_overlap_pairs") for pair in cf[key]}
        ff_faces={index for key in ("proper_pairs","coplanar_overlap_pairs") for pair in ff[key] for index in pair}
        groups[which]=dict(CF=_illegal_quality(quality,cf_faces),FF=_illegal_quality(quality,ff_faces),
            union=_illegal_quality(quality,cf_faces|ff_faces))
    topo=deepcopy(topology) if topology is not None else dict(status="NOT_RUN")
    indexed=topo.get("topology",topo)
    cf,ff=detectors["CF"]["verified"],detectors["FF"]["verified"]
    any_illegal=any(detectors[scope][which]["illegal_pair_count"]>0 for scope in ("CF","FF") for which in ("primary","independent"))
    unresolved=cf["unresolved_count"]+ff["unresolved_count"]
    static=bank.get("context_audits",{}).get(int(parent))
    if static is not None:
        if static["context_sha256"]!=hashlib.sha256(np.ascontiguousarray(case["constraints"]).tobytes()).hexdigest() or static["bbox_diagonal"]!=L:
            raise AssertionError("Static C/C cache belongs to a different context or scale")
        detectors["CC"]=deepcopy(static["detector_views"])
        any_illegal=any_illegal or any(detectors["CC"][which]["illegal_pair_count"]>0 for which in ("primary","independent"))
        unresolved+=detectors["CC"]["verified"]["unresolved_count"]
        if ledger is not None:ledger["static_context_audit_reuses"]=ledger.get("static_context_audit_reuses",0)+1
    embedded=("FAIL_DETECTED_ILLEGAL_INTERSECTIONS" if any_illegal else "UNRESOLVED" if unresolved else
              "NO_ILLEGAL_TRIANGLE_INTERSECTIONS_DETECTED" if static is not None else "NO_ILLEGAL_CF_FF_DETECTED_C_C_NOT_RUN")
    metrics["gpu_guidance_evaluation"]=dict(schema="gpu_guidance_evaluation_v1",input_coordinates="Evaluated verbatim; no scoring repairs",
        free_shape_quality=quality,detector_views=detectors,illegal_free_face_quality=groups,
        indexed_topology=topo,C_PRESERVED="PASS" if scans["context_bitwise_preserved"] else "FAIL",
        COMBINATORIAL_CLOSURE_STATUS=indexed.get("COMBINATORIAL_CLOSURE_STATUS","NOT_RUN"),
        EMBEDDED_GEOMETRY_VALIDITY_STATUS=embedded,
        embedded_scope="All static112-face unordered pairs: CF1200 + FF4950 + fixed CC66 reused" if static is not None else "CF1200 + FF4950; fixed CC NOT_RUN",
        static_context_audit_reused=static is not None,continuous_collision_safety="NOT_RUN",
        geometry_certification="Numerical diagnostic at declared tolerances, not an exact-predicate or continuous collision certificate",
        no_pruning=True,no_welding_in_scorer=True,whole_faces=112,free_faces=100,
        scoring_and_additional_audit_seconds=time.perf_counter()-tick,
        model_forwards=0,model_backwards=0,GPU_energy_evaluations=0,geometry_autograd_calls=0,optimizer_updates=0)
    if ledger is not None:ledger["gpu_guidance_variant_scores"]=ledger.get("gpu_guidance_variant_scores",0)+1
    json.dumps(metrics,allow_nan=False)
    return metrics


def compare_variants(before,after):
    """Same generated output before/after fixed W; use also for same-layer G0 comparisons."""
    a,b=before["gpu_guidance_evaluation"],after["gpu_guidance_evaluation"]
    if before["parent_index"]!=after["parent_index"] or before["K"]!=after["K"]:
        raise ValueError("Comparisons require the same parent and K")
    changes={}
    for scope in ("CF","FF"):
        changes[scope]={}
        for which in ("primary","independent","verified"):
            changes[scope][which]={}
            for kind in ("proper_pairs","coplanar_overlap_pairs","unresolved_pairs"):
                old={tuple(x) for x in a["detector_views"][scope][which][kind]}
                new={tuple(x) for x in b["detector_views"][scope][which][kind]}
                changes[scope][which][kind]=dict(before=len(old),after=len(new),added=[list(x) for x in sorted(new-old)],removed=[list(x) for x in sorted(old-new)])
    shape_a,shape_b=a["free_shape_quality"],b["free_shape_quality"]
    areas_a=np.array([x["area"] for x in shape_a["per_face"]]);areas_b=np.array([x["area"] for x in shape_b["per_face"]])
    ratios=np.divide(areas_b,areas_a,out=np.full_like(areas_b,np.nan),where=areas_a>0)
    ratios[(areas_a==0)&(areas_b>0)]=np.inf
    result=dict(pair_set_turnover=changes,area_ratio_after_over_before=_distribution(ratios),
        total_area_before=shape_a["total_area"],total_area_after=shape_b["total_area"],
        q_gt10_count_before=shape_a["q_actual_gt10_count"],q_gt10_count_after=shape_b["q_actual_gt10_count"],
        narrow_fraction_difference=shape_b["min_angle_lt5_fraction"]-shape_a["min_angle_lt5_fraction"],
        exact_zero_before=shape_a["exact_zero_area_count"],exact_zero_after=shape_b["exact_zero_area_count"],
        near_degenerate_before=shape_a["near_degenerate_area_count"],near_degenerate_after=shape_b["near_degenerate_area_count"],
        remainder_RMS_difference=after["remainder"]["own"]["symmetric_rms_bbox_pct"]-before["remainder"]["own"]["symmetric_rms_bbox_pct"],
        forward_RMS_difference=after["remainder"]["own"]["predicted_to_target_rms_bbox_pct"]-before["remainder"]["own"]["predicted_to_target_rms_bbox_pct"],
        reverse_RMS_difference=after["remainder"]["own"]["target_to_predicted_rms_bbox_pct"]-before["remainder"]["own"]["target_to_predicted_rms_bbox_pct"],
        boundary_RMS_difference=after["guided_geometry_audit"]["boundary"]["rms_bbox_pct"]-before["guided_geometry_audit"]["boundary"]["rms_bbox_pct"],
        comparison_scope="Fixed same-parent/seed face slots; no reassignment or output selection",geometry_evaluations=0,model_calls=0)
    json.dumps(result,allow_nan=False)
    return result


def cross_binding(context0_metrics,context1_metrics):
    return patch_binding_four_grid(context0_metrics,context1_metrics)


def run_cpu_tests():
    eq=np.array([[[0.,0.,0.],[1.,0.,0.],[.5,np.sqrt(3)/2,0.]]])
    reference=shape_quality(eq,1.)
    assert abs(reference["per_face"][0]["q_actual"]-1)<1e-12
    assert reference["guarded_EA_CPU_formula_reference"]==0
    for order in itertools.permutations(range(3)):
        value=shape_quality(eq[:,order],1.)
        assert abs(value["per_face"][0]["q_actual"]-1)<1e-12
    translated=eq+np.array([3.,-7.,2.])
    rotated=translated[:,:,[1,2,0]]
    assert abs(shape_quality(rotated,1.)["per_face"][0]["q_actual"]-1)<1e-12
    assert abs(shape_quality(eq*2,1.)["per_face"][0]["q_actual"]-1)<1e-12
    zero=shape_quality(np.zeros((1,3,3)),1.)
    assert zero["exact_zero_area_count"]==1 and zero["area_floor_trigger_count"]==1
    assert zero["q_actual"]["nan_count"]==1 and zero["guarded_EA_CPU_formula_reference"]==0
    sliver=shape_quality(np.array([[[0.,0.,0.],[1.,0.,0.],[.5,1e-6,0.]]]),1.)
    assert sliver["q_actual_gt10_count"]==1 and sliver["min_angle_lt5_count"]==1
    tiny=shape_quality(eq*1e-5,1.)
    assert tiny["area_floor_trigger_count"]==1 and tiny["near_degenerate_area_count"]==0
    assert abs(tiny["per_face"][0]["q_actual"]-1)<1e-12
    view=_detector_view(dict(tested_pairs=3,contacts=[dict(pair=[0,1],classification="UNRESOLVED",
        detector=dict(primary=dict(classification=PROPER),independent=dict(classification="UNRESOLVED")))]),"primary")
    assert view["proper_pair_count"]==1 and view["classification_counts"]["disjoint"]==2
    clean=_illegal_quality(reference,[0])
    assert clean["all_illegal_faces_are_q_gt10"] is False
    json.dumps([reference,zero,sliver,tiny,view,clean],allow_nan=False)
    return dict(status="PASS",fixture_families=9,vertex_permutations=6,model_calls=0,GPU_calls=0,
                geometry_tests=0,geometry_autograd_calls=0,optimizer_updates=0)
