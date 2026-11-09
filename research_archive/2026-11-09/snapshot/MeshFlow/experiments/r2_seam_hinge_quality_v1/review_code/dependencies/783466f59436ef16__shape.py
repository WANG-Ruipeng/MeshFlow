from collections import Counter
import math
import numpy as np
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
