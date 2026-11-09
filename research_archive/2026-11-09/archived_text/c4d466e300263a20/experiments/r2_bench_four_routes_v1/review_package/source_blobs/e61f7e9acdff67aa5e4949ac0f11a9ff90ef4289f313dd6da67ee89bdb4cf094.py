"""Read-only asset checks and new immutable SEAM registration inputs; no model."""
from __future__ import annotations
import collections, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'r2/src'))
import numpy as np
from .common import *
from r2.pilots.bench_parts_convergence_v1.data import load_case, boundary_from_C


def owner_contract(row):
    case=load_case(row)
    labels=np.asarray(case['occupied_labels'],bool)
    edges=boundary_from_C(case['C_vertices'],case['C_faces_original'])
    errors=[]
    if len(labels)!=len(edges):errors.append('LABEL_BOUNDARY_COUNT')
    if len(edges)>int(row['N'])-int(row['K']):errors.append('BOUNDARY_SLOTS_EXCEED_FREE_BUDGET')
    chosen=[]
    for ei in np.flatnonzero(labels):
        fi=int(case['edge_owner_face'][ei]);triple=case['edge_source_triple'][ei]
        if fi not in set(map(int,case['free_ids'])):errors.append('OWNER_NOT_FREE')
        if fi in chosen:errors.append('DUPLICATE_OWNER')
        chosen.append(fi)
        source_face=case['source_face_vertex_ids'][fi]
        if any(np.count_nonzero(source_face==vi)!=1 for vi in triple):errors.append('SOURCE_CORNER_IDENTITY')
        else:
            cp=np.asarray([np.flatnonzero(source_face==vi).item() for vi in triple])
            target=case['full_target'][fi,cp]
            _,u,v=edges[ei]
            if not np.array_equal(target[:2],case['C_vertices'][[v,u]]):errors.append('OWNER_ANCHOR_COORDINATE')
    # All non-owner Y faces remain bulk. No deletion, duplicate seam, or GT tip anchor.
    return dict(condition_id=row['condition_id'],parent_id=row['parent_id'],N=int(row['N']),K=int(row['K']),
        boundary_edges=len(edges),owners=int(labels.sum()),distinct_owner_faces=len(set(chosen)),
        closed_C=len(edges)==0,case_path=row['case_path'],case_sha256=sha(row['case_path']),
        status='PASS' if not errors else 'BLOCKED',errors=sorted(set(errors)),
        third_corner_in_C_count=sum(int(case['edge_source_triple'][ei,2]) in set(map(int,case['C_source_vertex_ids'])) for ei in np.flatnonzero(labels)),
        contract='Owner-only existing whole-face target; source third-C vertices remain generated; aliases/Null return to bulk')


def prepare():
    src=SOURCE_ROOT/'data/manifest.json';m=read(src)
    rows=[owner_contract(r) for r in m['train']]
    if len(rows)!=26:raise ValueError('Exactly the original26 training parents required')
    summary=dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'BLOCKED',rows=rows,
        source_manifest=str(src),source_manifest_sha256=sha(src),model_forwards=0,OT_calls=0,
        source_checkpoint_exists=SOURCE.is_file(),source_checkpoint_sidecar=read(SOURCE.with_suffix('.json')))
    write_new(OUT/'data/owner_compatibility.json',summary)
    if summary['status']!='PASS':
        print(json.dumps(summary));return summary
    for row in m['train']:
        if sha(row['case_path'])!=row['case_sha256']:raise ValueError('Original case changed')
    ev=read(EVAL_SOURCE)
    if len(ev['requests'])!=46:raise ValueError('Fixed existing46 requests required')
    # Preserve request dictionaries/noise/cameras/query; merely cite them in a new manifest.
    write_new(OUT/'data/manifest.json',m)
    write_new(OUT/'data/eval_manifest.json',ev)
    from . import stream
    schedule=stream.make_schedule(m['train'],'SEAM1500')
    write_new(OUT/'data/schedule_SEAM1500.json',schedule)
    write_new(OUT/'data/stream_gate.json',stream.validate(schedule,m['train'],1500,'SEAM1500'))
    print(json.dumps({k:summary[k] for k in ('status','model_forwards','OT_calls')}))
    return summary

if __name__=='__main__':prepare()
