"""Original CPU A rollout and an explicitly NEW, separately timed quality fallback."""
from __future__ import annotations
import time
import numpy as np
import torch
from ..exporter import RadialSupport
from ..legacy.common import sample_surface
from ..legacy.deform import deform
from ..layout_value.geometry import mesh_quality, audit_mesh
from ..io_utils import array_hash

QUALITY_EPS_DEG = 1e-6

def named_quality(vertices, faces):
    q = mesh_quality(vertices, faces)
    q["min_triangle_angle_deg"] = q.pop("min_angle_deg")
    q["triangle_min_angle_p05_deg"] = q.pop("angle_q05_deg")
    q["triangle_min_angle_median_deg"] = q.pop("angle_median_deg")
    q["triangle_min_angle_p05_definition"] = q.pop("angle_q05_definition")
    q["triangle_shape_quality_p05"] = q.pop("shape_quality_q05")
    return q

def qualification(quality, audit, floor, requested_faces=512):
    reasons=[]
    if quality["face_count"] != requested_faces: reasons.append("face_budget")
    if audit.get("valid") is not True: reasons.append("geometry_validity")
    if quality["degenerate_count"]: reasons.append("degenerate")
    for key in ("min_triangle_angle_deg", "triangle_min_angle_p05_deg"):
        if not np.isfinite(quality[key]) or quality[key]+QUALITY_EPS_DEG < floor[key]:
            reasons.append(key)
    return {"qualified":not reasons, "failure_reasons":reasons}

def choose_delivery(raw_mesh, fixed_mesh, raw_qualification):
    """No RMS or teacher label is accepted by this deploy-time decision."""
    fallback = not raw_qualification["qualified"]
    return (fixed_mesh if fallback else raw_mesh), {
        "fallback_to_fixed":fallback,
        "fallback_reasons":raw_qualification["failure_reasons"] if fallback else [],
        "policy":"endpoint quality+validity+budget gate, else Fixed; no error-based selection",
    }

def make_fixed(support, logr, base, faces):
    begin=time.perf_counter()
    u=np.asarray(base,np.float64).copy()
    r=support.query(logr,u)
    mesh={"vertices":u*r[:,None],"faces":np.asarray(faces,np.int64),"u":u,"r":r}
    return mesh,time.perf_counter()-begin

def predict_raw(flow, logr, support, base, faces):
    """Network reads ONLY original proxy256 features; final radius is re-queried."""
    if next(flow.parameters()).device.type!="cpu": raise ValueError("Deployment retains original CPU path")
    begin=time.perf_counter()
    pv,pf=support.mesh(logr)
    proxy=sample_surface(pv,pf,256,np.random.default_rng(4501))[0]
    proxy_seconds=time.perf_counter()-begin
    u,path,alphas,rollout_seconds=deform(flow,proxy,base,faces,True,32)
    query_begin=time.perf_counter()
    r=support.query(logr,u)
    mesh={"vertices":u*r[:,None],"faces":np.asarray(faces,np.int64),"u":u,"r":r,"alpha":alphas}
    return mesh, {
        "generation_seconds":time.perf_counter()-begin,
        "proxy_seconds":proxy_seconds,
        "rollout_seconds":rollout_seconds,
        "final_query_seconds":time.perf_counter()-query_begin,
        "proxy_points_hash":array_hash(proxy),
        "min_alpha":float(alphas.min()),
        "shrunk_steps":int((alphas<1).sum()),
        "network_device":"cpu","network_dtype":"float32","guard_geometry_dtype":"float64",
        "steps":32, "final_accepted_direction_requery":True,
    }

def deploy(flow, logr, support, base, faces, *, fixed_only=False):
    """Timings include actual endpoint checks; no evaluation/teacher geometry used."""
    begin=time.perf_counter()
    fixed, fixed_generation_seconds=make_fixed(support,logr,base,faces)
    fq_start=time.perf_counter()
    floor=named_quality(fixed["vertices"],faces)
    fixed_audit=audit_mesh(fixed["vertices"],faces,exact=True)
    floor_seconds=time.perf_counter()-fq_start
    if not fixed_audit["valid"]: raise ValueError("Fixed fallback is not geometrically valid")
    if fixed_only:
        timing={"generation_seconds":fixed_generation_seconds,"quality_and_audit_seconds":floor_seconds,
                "raw_with_audit_seconds":time.perf_counter()-begin,"deployment_device":"cpu"}
        return {"raw_mesh":fixed,"delivered_mesh":fixed,"raw_quality":floor,"raw_audit":fixed_audit,
                "delivered_quality":floor,"delivered_audit":fixed_audit,"floor":floor,
                "raw_qualification":qualification(floor,fixed_audit,floor),
                "delivery":{"fallback_to_fixed":False,"policy":"Fixed baseline","fallback_reasons":[]},
                "timing":timing}
    raw, meta=predict_raw(flow,logr,support,base,faces)
    q_start=time.perf_counter()
    quality=named_quality(raw["vertices"],faces)
    raw_audit=audit_mesh(raw["vertices"],faces,exact=True)
    q=qualification(quality,raw_audit,floor)
    qa_seconds=time.perf_counter()-q_start
    select_begin=time.perf_counter()
    delivered, delivery=choose_delivery(raw,fixed,q)
    select_seconds=time.perf_counter()-select_begin
    total=time.perf_counter()-begin
    return {"raw_mesh":raw,"delivered_mesh":delivered,"raw_quality":quality,"raw_audit":raw_audit,
            "delivered_quality":floor if delivery["fallback_to_fixed"] else quality,
            "delivered_audit":fixed_audit if delivery["fallback_to_fixed"] else raw_audit,
            "floor":floor,"raw_qualification":q,"delivery":delivery,
            "timing":{**meta,"quality_and_audit_seconds":qa_seconds,
                "raw_with_audit_seconds":meta["generation_seconds"]+qa_seconds,
                "fixed_generation_seconds":fixed_generation_seconds,"fixed_quality_and_audit_seconds":floor_seconds,
                "selection_seconds":select_seconds,"quality_fallback_total_seconds":total,
                "metric_evaluation_excluded":True,"model_load_excluded":True}}
