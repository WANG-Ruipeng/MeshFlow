"""Map cached GEO deletion reasons to existing RAW/PRE_GEO source slots.

No GEO execution, pair detector, scoring, model or random number generation.
Areas are triangle-area sums, not union areas or a source-GT quality criterion.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from legacy import ROOT, OUT, OLD, REASONS, sha, save, table

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}

def measures(tri):
    t = np.asarray(tri, dtype=np.float64)
    cross = np.cross(t[:,1]-t[:,0], t[:,2]-t[:,0])
    area = np.linalg.norm(cross, axis=1)*.5
    lengths = np.stack([np.linalg.norm(t[:,(i+1)%3]-t[:,i],axis=1) for i in range(3)],axis=1)
    repeated = np.any(np.stack([np.all(t[:,(i+1)%3]==t[:,i],axis=1) for i in range(3)],axis=1),axis=1)
    collinear = np.all(cross==0,axis=1) & ~repeated
    reasons = np.where(repeated,'exact_repeated_vertex',np.where(collinear,'exact_collinear','NONDEGENERATE'))
    return area,lengths.min(axis=1),reasons

def median(values):
    return float(np.median(values)) if values else 'NA_EMPTY'

def run(stage):
    source = OLD if stage=='old' else OUT.parent
    labels = ('ORIGINAL_C200','DEDUP_C200') if stage=='old' else ('ORIGINAL_LOW_N','DEDUP_LOW_N')
    regpath = source/'registration.json'
    reg = read(regpath)
    requests = {q['request_id']:q for q in reg['requests']}
    assert len(requests)==24
    hashes={str(regpath):sha(regpath)}
    face_rows, output_rows, summary = [], [], []
    for label in labels:
        paths=sorted((source/'evaluation/post'/label).glob('*/receipt.json'))
        assert len(paths)==24, (label,len(paths))
        assert {p.parent.name for p in paths}==set(requests)
        for p in paths:
            receipt=read(p);q=requests[p.parent.name]
            assert receipt['status']=='COMPLETE' and receipt['request_id']==q['request_id']
            assert receipt['label']==label
            files=receipt['files']
            geometry_refs=[f for f in files if Path(f).name=='geometry.json']
            assert len(geometry_refs)==1, 'Missing/ambiguous cached geometry reference'
            gp=Path(geometry_refs[0]);pp=Path(receipt['pre_geo_path']);rp=Path(receipt['raw_path'])
            assert str(pp) in files, 'PRE_GEO hash is missing from receipt.files'
            for f,h in ((gp,files[str(gp)]),(pp,files[str(pp)]),(rp,receipt['raw_sha256'])):
                assert sha(f)==h, f
                hashes[str(f)]=h
            hashes[str(p)]=sha(p)
            g=read(gp);stats=g['stats'];raw=arrays(rp)['raw'];pre=arrays(pp)['output']
            n,k=int(q['N']),int(q['K']);radius=float(stats['radius_model'])
            assert raw.shape in ((n,9),(n,3,3)), raw.shape
            raw=raw.reshape(n,3,3)
            assert raw.shape==pre.shape==(n,3,3)
            assert raw.dtype==pre.dtype==np.float32
            assert raw[:k].tobytes()==pre[:k].tobytes()
            assert stats['input_faces']==n and stats['known_faces']==k
            records=g['face_records']
            assert sorted(f['source_face_id'] for f in records)==list(range(n))
            ra,rl,rd=measures(raw);pa,pl,pd=measures(pre)
            denominator=float(pa[k:].sum());rawden=float(ra[k:].sum())
            assert denominator>0 and rawden>0
            counter=Counter(f['reason'] for f in records if not f['kept'])
            assert counter==Counter(stats['deleted_by_reason'])
            assert sum(counter.values())==stats['deleted_free_faces']
            for f in records:
                if f['kept']:continue
                i=int(f['source_face_id'])
                assert i>=k and not f['known']
                origin='NOT_REPEATED_VERTEX_CLASS'
                if f['reason']=='exact_repeated_vertex':
                    origin=('PRE_GEO_ALREADY_REPEATED_VERTEX' if pd[i]=='exact_repeated_vertex' else
                            'PRE_GEO_ALREADY_COLLINEAR' if pd[i]=='exact_collinear' else 'GEO_NEW_EXACT_DEGENERATION')
                face_rows.append(dict(model=label,request_id=q['request_id'],parent_id=q['parent_id'],split=q['split'],
                    candidate=q['candidate'],source_face_id=i,identity_domain='GENERATION_SOURCE_SLOT_NOT_TRAINING_FACE_ID',
                    reason=f['reason'],duplicate_of_source_face=f['duplicate_of_source_face'],
                    raw_exact_degeneracy=str(rd[i]),pre_geo_exact_degeneracy=str(pd[i]),repeated_vertex_origin=origin,
                    GEO_new_exact_degeneracy=bool(f['reason'] in ('exact_repeated_vertex','exact_collinear') and pd[i]=='NONDEGENERATE'),
                    RAW_area=float(ra[i]),PRE_GEO_area=float(pa[i]),
                    PRE_GEO_area_fraction_of_all_free=float(pa[i]/denominator),
                    RAW_shortest_edge=float(rl[i]),PRE_GEO_shortest_edge=float(pl[i]),fixed_GEO_radius=radius,
                    RAW_shortest_edge_over_radius=float(rl[i]/radius),PRE_GEO_shortest_edge_over_radius=float(pl[i]/radius)))
            local=[x for x in face_rows if x['model']==label and x['request_id']==q['request_id']]
            base=dict(model=label,request_id=q['request_id'],parent_id=q['parent_id'],split=q['split'],candidate=q['candidate'],
                N=n,K=k,input_free=n-k,RAW_all_free_area_sum=rawden,PRE_GEO_all_free_area_sum=denominator)
            for reason in REASONS+('ALL_DELETED',):
                subset=local if reason=='ALL_DELETED' else [x for x in local if x['reason']==reason]
                output_rows.append(dict(**base,reason=reason,deleted_count=len(subset),
                    PRE_GEO_area_sum=sum(x['PRE_GEO_area'] for x in subset),
                    PRE_GEO_area_fraction=sum(x['PRE_GEO_area'] for x in subset)/denominator,
                    RAW_area_sum=sum(x['RAW_area'] for x in subset),
                    RAW_shortest_edge_over_radius_median=median([x['RAW_shortest_edge_over_radius'] for x in subset]),
                    PRE_GEO_shortest_edge_over_radius_median=median([x['PRE_GEO_shortest_edge_over_radius'] for x in subset]),
                    PRE_GEO_already_exact_degenerate=sum(x['pre_geo_exact_degeneracy']!='NONDEGENERATE' for x in subset),
                    GEO_new_exact_degeneration=sum(x['GEO_new_exact_degeneracy'] for x in subset)))
        for reason in REASONS+('ALL_DELETED',):
            subset=[x for x in face_rows if x['model']==label and (reason=='ALL_DELETED' or x['reason']==reason)]
            outs=[x for x in output_rows if x['model']==label and x['reason']==reason]
            totalarea=sum(x['PRE_GEO_all_free_area_sum'] for x in outs)
            summary.append(dict(model=label,reason=reason,requests=len(outs),deleted_count=len(subset),
                PRE_GEO_deleted_area_sum=sum(x['PRE_GEO_area'] for x in subset),
                PRE_GEO_all_free_area_sum=totalarea,
                PRE_GEO_deleted_area_fraction_pooled=sum(x['PRE_GEO_area'] for x in subset)/totalarea,
                PRE_GEO_deleted_area_fraction_macro=float(np.mean([x['PRE_GEO_area_fraction'] for x in outs])),
                RAW_shortest_edge_over_radius_median=median([x['RAW_shortest_edge_over_radius'] for x in subset]),
                PRE_GEO_shortest_edge_over_radius_median=median([x['PRE_GEO_shortest_edge_over_radius'] for x in subset]),
                RAW_already_exact_degenerate=sum(x['raw_exact_degeneracy']!='NONDEGENERATE' for x in subset),
                PRE_GEO_already_repeated_vertex=sum(x['pre_geo_exact_degeneracy']=='exact_repeated_vertex' for x in subset),
                PRE_GEO_already_collinear=sum(x['pre_geo_exact_degeneracy']=='exact_collinear' for x in subset),
                GEO_new_exact_degeneration=sum(x['GEO_new_exact_degeneracy'] for x in subset)))
    for f,h in hashes.items():assert sha(f)==h, f
    prefix=stage+'_deletion_mechanism_v2'
    table(prefix+'_per_face.csv',face_rows)
    table(prefix+'_per_output.csv',output_rows)
    table(prefix+'_summary.csv',summary)
    result=dict(status='PASS',version='cached_deletion_mechanism_v2',stage=stage,source_sha256=sha(__file__),request_count=48,deleted_faces=len(face_rows),
        preserved_inputs=hashes,summary=summary,
        method='Cached face_records.source_face_id directly indexes RAW and PRE_GEO generation slot. Exact degeneracy independently duplicates the source detector order: repeated numeric coordinate, then exact FP64 cross-product zero. No tolerance or geometry reprocessing.',
        area_caveat='Triangle areas are summed with multiplicity. They are not union area, visible area, source-GT reconstruction error or proof of missing body. Macro fractions weight all 24 outputs equally; pooled fractions weight PRE_GEO free surface area.',
        identity_caveat='Generation source slot identities preserve row correspondence across RAW/PRE_GEO/GEO within an output; they are not source training target IDs or guaranteed corresponding physical faces between different model outputs.',
        new_model_calls=0,new_geometry_processing=0,new_intersection_scans=0,new_quality_scores=0)
    if stage=='old':
        result['engineering_history']=['First CPU attempt stopped before output at a shape assertion: RAW stored as N x 9; matched original source reshape to N x 3 x 3.',
            'Initial v1 aggregate GEO_new_exact_degeneration only counted repeated-vertex category. v2 includes collinear degeneracy as well; all original v1 outputs retained. Repeat-vertex +430 conclusion unchanged.']
    save(prefix+'.json',result)
    print(json.dumps(dict(status='PASS',stage=stage,deleted_faces=len(face_rows),summary=summary),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('old','low'),required=True)
    run(p.parse_args().stage)
