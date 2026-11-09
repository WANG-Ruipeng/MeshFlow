"""CPU-only fixed complete-seat evaluation panel; no model import on import.
Selection is supplied after input review, never chosen using generated outputs.
"""
from __future__ import annotations
import argparse, collections, hashlib, sys, time
from pathlib import Path
import numpy as np
from .common import ROOT, OUT, read, resolve, sha, resources
from r2.pilots.bench_parts_convergence_v1 import panel as P
VERSION = "BENCH_FULL_SEAT_CURRICULUM_FIXED24_V1"
LABELS = ("COMMON_S12000", "NATIVE_S6000", "SEAM_S3000")
L = float(np.sqrt(3) * 2 / .3762)

def arrays(path):
    with np.load(resolve(path), allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}

def verify(files):
    for p,h in files.items():
        if sha(p)!=h: raise ValueError("Frozen asset differs: " + str(p))

def seed(cid, noise_index):
    namespace = VERSION + "|EVAL|" + cid + "|" + str(noise_index)
    return int.from_bytes(hashlib.sha256(namespace.encode("utf-8")).digest()[:4], "little")

def protect(files, path, expected=None):
    path = resolve(path)
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError("Source hash differs: " + str(path))
    key = str(path)
    if key in files and files[key] != digest:
        raise ValueError("Asset changed during preparation")
    files[key] = digest
    return digest

def new_bundle(row, protected):
    from meshflow_control.source_winding import capture_source, validate_source
    from r2.pilots.user_condition_finetune_v1 import evaluation as old
    from r2.tools.avmr_pmr_v1 import run as A
    from r2.tools.cref_layer_audit_v1 import render as H
    from r2.pilots.chair_multipart_seam_v1 import gallery as G
    cid = row["condition_id"]
    for key in ("input", "paired", "case"):
        protect(protected, row[key + "_path"], row[key + "_sha256"])
    inp, pair = arrays(row["input_path"]), arrays(row["paired_path"])
    core = {key: inp[key] for key in ("C_vertices", "C_faces_original", "C_faces_model", "N", "K")}
    meta = {key: pair[key] for key in ("C_source_vertex_ids", "known_ids")}
    original, canonical = core["C_faces_original"], core["C_faces_model"]
    cc, co = core["C_vertices"][canonical], core["C_vertices"][original]
    if cc.dtype != np.float32 or not np.array_equal(cc, pair["C"]) or cc.shape != (row["K"], 3, 3):
        raise ValueError("Input and supervised C differ")
    if (int(core["N"]), int(core["K"])) != (row["N"], row["K"]):
        raise ValueError("Selected input count differs")
    permutation = np.asarray([[np.flatnonzero(original[i] == vertex).item() for vertex in face]
                              for i, face in enumerate(canonical)], np.int64)
    source = capture_source(co, source_vertex_ids=meta["C_source_vertex_ids"][original],
                            corner_permutation=permutation, source_face_ids=meta["known_ids"])
    validate_source(source, cc)
    dest = OUT / "evaluation_inputs/conditions" / cid
    paths = dict(inference_npz=dest / "condition.npz", metadata_npz=dest / "metadata.npz", source_npz=dest / "source_condition.npz")
    for name, values in (("inference_npz", core), ("metadata_npz", meta), ("source_npz", source)):
        P.array_file(paths[name], values)
    _, _, topology = A.topology(co, meta["C_source_vertex_ids"][original])
    if int(topology["boundary_edge_count"]) != int(row["boundary_edges"]):
        raise ValueError("C source boundary count changed")
    bank, query_seed = old.query_bank(cc, cid)
    qp = OUT / "evaluation/queries" / (cid + ".npz")
    P.array_file(qp, bank)
    descriptor = dict(condition_id=cid, parent_id=row["parent_id"], path=str(qp), sha256=protect(protected, qp),
        array_sha256={k: old.array_sha(v) for k, v in bank.items()},
        identity=dict(condition_id=cid, C_seed=int(query_seed), C_sha256=old.array_sha(cc), L0=L),
        query_count=2048, weight_denominator=float(bank["C_weights"].sum()), source_boundary_audit=topology)
    # Source-derived framing is frozen before any generation. This is display
    # metadata only, not a unique-style scoring target or model input.
    g = dict(C=co, F=pair["Y"], C_original=co, tri=np.concatenate((co, pair["Y"])))
    bounds = H.bounds_for_condition([g], None, L)
    points = g["tri"].reshape(-1, 3)
    center = (points.min(0) + points.max(0)) / 2
    span = max(np.ptp((points - center) @ G.D.basis(az, 23), axis=0)[:2].max() for az in (-60, 120)) * 1.25
    depth = dict(center=center.tolist(), span=float(span))
    result = {**row, **{k: str(p) for k, p in paths.items()}, "query_descriptor": descriptor,
              "gallery_bounds": bounds, "gallery_depth_bounds": depth,
              "gallery_frame_source": "FROZEN_SOURCE_COMPLETION_DISPLAY_ONLY",
              "source_labels_available_to_models": False, "closed_C": row["boundary_edges"] == 0,
              "part_family": row["part_type"] if row["part_type"] in P.FAMILIES else "COMBINATION"}
    for key, path in paths.items():
        result[key + "_sha256"] = protect(protected, path)
    result.update(metadata_sha256=result["metadata_npz_sha256"], source_sha256=result["source_npz_sha256"])
    return result

def prepare(selection_path=None):
    done = OUT / 'eval_assets_complete.json'
    if done.exists():
        result=read(done);verify(result['files']);validate_assets();return result
    selection_path=resolve(selection_path or OUT/'eval_selection.json')
    selection=read(selection_path)
    if selection.get('status')!='APPROVED_INPUT_REVIEW':
        raise ValueError('Input-reviewed selection is required before noise/query creation')
    chosen=selection['conditions']
    if len(chosen)!=12 or len({r['parent_id'] for r in chosen})!=12:
        raise ValueError('12 distinct parents with one full-seat C each required')
    if collections.Counter(r['eval_group'] for r in chosen)!=dict(CURRENT_TRAIN=8,DEV=4):
        raise ValueError('Fixed 8 training plus 4 held-out development parents required')
    data=read(OUT/'data/manifest.json')
    train=data.get('train',data.get('rows',[]));train_ids={r['condition_id'] for r in train}
    train_parents={r['parent_id'] for r in train}
    for r in chosen:
        if r['eval_group']=='CURRENT_TRAIN' and r['condition_id'] not in train_ids:raise ValueError('Not actual training C')
        if r['eval_group']=='DEV' and r['parent_id'] in train_parents:raise ValueError('DEV parent leaked into curriculum')
        if r.get('full_seat_review') not in ('ACCEPT','PASS'):raise ValueError('Complete seat input not approved')
    if (OUT/'evaluation_inputs').exists():raise RuntimeError('Partial panel retained; explicit audit required')
    protected={};tick=time.perf_counter()
    for path in (selection_path,OUT/'data/manifest.json',Path(__file__)):
        protect(protected,path)
    from r2.pilots.user_condition_finetune_v1 import evaluation as original
    for path,digest in original.scorer_sources().items():protect(protected,path,digest)
    rows=[];conditions=[]
    for source in chosen:
        r=dict(source,split=source['eval_group'],part_type='SEAT_SURFACE',baseline_reuse=False,
               planned_current_training_C=source['eval_group']=='CURRENT_TRAIN')
        base=new_bundle(r,protected);conditions.append({k:base[k] for k in ('condition_id','parent_id','N','K','boundary_edges','closed_C','part_type','eval_group','gallery_bounds','gallery_depth_bounds')})
        for ni in (0,1):
            cid=r['condition_id'];rid=cid+'__n'+str(ni);frozen_seed=seed(cid,ni)
            z=np.random.RandomState(frozen_seed).standard_normal((r['N']-r['K'],3,3)).astype(np.float32)
            path=OUT/'evaluation_inputs/requests'/rid/'noise.npz'
            P.array_file(path,dict(z=z,seed=np.asarray(frozen_seed,np.int64)))
            digest=protect(protected,path)
            rows.append(dict(base,request_id=rid,noise_index=ni,seed=frozen_seed,noise_npz=str(path),noise_sha256=digest,noise_npz_sha256=digest))
    manifest=dict(schema=VERSION,status='FROZEN',requests=rows,conditions=conditions,labels=list(LABELS),
        stages=['RAW','BASE_POST','CLEAN_FINAL_POST'],protected=protected,requests_per_model=24,
        new_noise_arrays=24,new_query_arrays=12,new_RAW=72,new_D0=72,generator_forwards=3600,
        new_scoring_rows=216,reused_scoring_rows=0,C_query_count=2048,L=L,
        group_counts=dict(collections.Counter(q['eval_group'] for q in rows)),
        inference_policy='COMMON/NATIVE: all free, zero head; SEAM: frozen OLD_S15000 C-only head, source-owner never read',
        maximum_head_forwards=12,all_RAW_before_D0=True,no_unique_GT_style_success=True,
        metric_contract='Original scorers unchanged. No source reconstruction success; seam-specific fields NA for Native; all-output C/F and quality metrics retained.',
        gallery_policy='Source-fixed azimuth -60/120 elevation23 and central front/side sections; shared across all models/stages',
        model_calls=0,head_calls=0,scoring_calls=0)
    P.immutable(OUT/'eval_manifest.json',manifest)
    checks=validate_assets();P.immutable(OUT/'eval_cpu_checks.json',checks)
    if 'torch' in sys.modules:raise RuntimeError('CPU panel preparation imported neural framework')
    verify(protected)
    result=dict(status='COMPLETE_CPU_PREPARATION',conditions=12,requests_per_model=24,
        model_calls=0,GPU_calls=0,scoring_calls=0,seconds=time.perf_counter()-tick,
        files={str(p):sha(p) for p in (OUT/'eval_manifest.json',OUT/'eval_cpu_checks.json')})
    P.immutable(done,result);return result

def requests(label=None):
    if label is not None and label not in LABELS:raise ValueError('Unknown label')
    m=read(OUT/'eval_manifest.json');verify(m['protected']);rows=m['requests']
    if len(rows)!=24 or len({q['request_id'] for q in rows})!=24:raise ValueError('24 unique requests required')
    c=collections.Counter(q['condition_id'] for q in rows)
    if len(c)!=12 or set(c.values())!={2}:raise ValueError('12C x2 noises required')
    return rows

def validate_assets():
    from meshflow_control.source_winding import validate_source
    checks=[]
    for q in requests():
        for key in ('inference_npz','metadata_npz','source_npz','noise_npz'):
            if sha(q[key])!=q[key+'_sha256']:raise ValueError('Changed input '+key)
        a=arrays(q['inference_npz']);z=arrays(q['noise_npz'])['z']
        if set(a)!={'C_vertices','C_faces_original','C_faces_model','N','K'}:raise ValueError('Model bundle includes non-C fields')
        if z.dtype!=np.float32 or z.shape!=(q['N']-q['K'],3,3) or not np.isfinite(z).all():raise ValueError('Bad noise')
        validate_source(arrays(q['source_npz']),a['C_vertices'][a['C_faces_model']])
        d=q['query_descriptor'];bank=arrays(d['path'])
        if sha(d['path'])!=d['sha256'] or len(bank['C_points'])!=2048 or not np.isclose(bank['C_weights'].sum(),1):raise ValueError('Query differs')
        checks.append(dict(request_id=q['request_id'],source_identity=True,noise_identity=True,query_identity=True))
    return dict(status='PASS',requests=24,conditions=12,checks=checks,model_calls=0,geometry_scores=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','verify'));p.add_argument('--selection');a=p.parse_args()
    result=prepare(a.selection) if a.action=='prepare' else validate_assets()
    print({k:v for k,v in result.items() if k not in ('files','checks')})
