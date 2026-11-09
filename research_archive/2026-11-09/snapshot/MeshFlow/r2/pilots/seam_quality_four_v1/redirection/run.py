"""Registered 16-output CPU seam-tip redirection; no models, noise or retries."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import argparse, collections, concurrent.futures, importlib, json, sys, time
import numpy as np
from r2.pilots.seam_slot_c200_v1 import common as C, generate as G, evaluate as E
from r2.pilots.seam_geometry_guard_v1 import capacity_guard as CG, summarize_capacity as SC
from meshflow_control import geo_io
from . import engine

OUT=C.ROOT/'experiments/r2_seam_quality_four_v1/redirection'
OLD=C.ROOT/'experiments/r2_seam_slot_c200_v1'
CAP=C.ROOT/'experiments/r2_seam_geometry_guard_v1/capacity'
ARM='SEAM_SLOT_C200'


def registration():
    dest=OUT/'registration.json'
    if dest.exists():
        reg=C.read(dest);E.verify(reg['protected']);E.verify(reg['source_hashes']);return reg
    resource=C.ROOT/'experiments/r2_seam_quality_four_v1/resources_initial.json'
    if not resource.exists(): raise ValueError('Root resource record required before dispatch')
    manifest=C.read(OLD/'eval_manifest.json')
    assert len(manifest['requests'])==16
    protected={str(resource):C.sha(resource),str(OLD/'eval_manifest.json'):C.sha(OLD/'eval_manifest.json')}
    for q in manifest['requests']:
        old=OLD/'evaluation'/ARM/q['request_id'];cap=CAP/'evaluation'/ARM/q['request_id']
        files=[old/'raw/raw.npz',old/'raw/source_condition.npz',old/'post/pre_geo.npz',old/'post/post.npz',
               cap/'post/post.npz',cap/'post/post_canonical.npz',cap/'post/merge_ledger.json']
        files += [C.resolve(q[x]) for x in ('inference_npz','metadata_npz')]
        files += [C.resolve(q['query_descriptor']['path'])]
        for root in (OLD,CAP):
            folder=root/'evaluation/scores'/ARM/q['request_id']/'POST'
            files += [folder/name for name in ('score.json','pairs.json','valid_edges.csv')]
        for p in files: protected[str(p)]=C.sha(p)
    sources=E.source_hashes()
    for p in [Path(engine.__file__),Path(__file__),Path(CG.__file__),Path(SC.__file__),Path(geo_io.__file__),Path(engine.O.__file__)]:
        sources[str(p)]=C.sha(p)
    reg=dict(version=engine.VERSION,status='FROZEN',requests=16,CPU_process_limit=2,threads_per_process=1,
        generator_calls=0,D0_calls=0,training_updates=0,new_noise=0,new_queries=0,new_POST=16,new_quality_scans=16,
        original_radius=engine.O.RADIUS,no_radius_sweep=True,protected=protected,source_hashes=sources,
        contract=['Use saved PREDICTED PRE_GEO and existing CAPACITY_POST, not source occupancy.',
        'Candidates are capacity-rejected bulk corners and only the rejected occupied C edge owner tips.',
        'Freeze actual CAPACITY clusters, assign the complete source cluster to the current tip representative. No reclustering.',
        'Every source-cluster corner is within the original radius of the target from its PRE_GEO position.',
        'Preserve C and seam hard anchors, current assigned cluster identities, all retained faces and all existing deleted-face identities.',
        'Reject new exact degeneracy/duplicates, new or worse nonmanifold edges, new illegal CF/FF pairs and new unresolved classifications.',
        'Original GEO and capacity are endpoint references; comparison does not prove a temporal cause or bulk intent.',
        'Order by original capacity rejection sequence, then owner RAW slot. No performance-based threshold selection.'],
        closest_prior='R10 terminal partner closure; partial_edge_binding_v1 whole-edge reassignment; SEAM capacity guard',
        substantive_change='Existing occupied owner tip is the only destination; actual capacity clusters move inside original radius and zero-new-conflict checks.',
        stopping='Exactly 16 CPU candidates and original scores, paired report and inspection. No automatic training/default/commit/push.')
    C.write(dest,reg);C.write(OUT/'eval_manifest.json',manifest)
    return reg


def geometry():
    reg=registration();manifest=C.read(OLD/'eval_manifest.json');rows=[]
    maths=E.helpers()[0].cpu_math()
    classifier=importlib.import_module(maths['geometry'].__package__+'.intersections').verified_pair
    for q in manifest['requests']:
        rid=q['request_id'];dest=OUT/'evaluation'/ARM/rid/'post';receipt=dest/'receipt.json'
        if receipt.exists():
            rr=C.read(receipt);E.verify(rr['files']);rows.append(rr['redirection']);continue
        if dest.exists():raise ValueError('Incomplete geometry attempt: explicit recovery required '+rid)
        C.write(dest/'started.json',dict(request_id=rid,utc=C.utc()))
        start=time.perf_counter();old=OLD/'evaluation'/ARM/rid;cap=CAP/'evaluation'/ARM/rid
        raw=C.arrays(old/'raw/raw.npz');source=C.arrays(old/'raw/source_condition.npz')
        pre=C.arrays(old/'post/pre_geo.npz')['output'];base=C.arrays(cap/'post/post_canonical.npz')
        rejected=C.read(cap/'post/merge_ledger.json')
        canonical,final,summary,ledger,orient,anchors=engine.execute(pre,raw,base,rejected,source,classifier)
        summary.update(request_id=rid,parent_id=q['parent_id'],noise_index=q['noise_index'],seconds=time.perf_counter()-start)
        G.save_npz(dest/'post.npz',final);G.save_npz(dest/'post_canonical.npz',canonical)
        geo_io.write_indexed_ply(dest/'post.ply',final['vertices'],final['faces'])
        C.write(dest/'candidate_ledger.json',ledger);C.csvwrite(dest/'candidate_ledger.csv',ledger)
        C.write(dest/'redirection.json',summary);C.write(dest/'anchor_identity.json',anchors)
        C.write(dest/'source_restoration.json',orient['restoration_summary'])
        C.write(dest/'orientation.json',orient['orientation_result']['summary'])
        C.write(receipt,dict(status='COMPLETE',arm=ARM,request_id=rid,redirection=summary,model_calls=0,
            registration_sha256=C.sha(OUT/'registration.json'),files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file()}))
        rows.append(summary);print('REDIRECTION_GEOMETRY',len(rows),rid,summary['accepted'],summary['funnel'],flush=True)
    C.csvwrite(OUT/'geometry_summary.csv',rows)
    C.write(OUT/'geometry_completion.json',dict(status='COMPLETE',rows=rows,model_calls=0))
    return rows


def score(workers=2):
    reg=registration();assert len(C.read(OUT/'geometry_completion.json')['rows'])==16
    manifest=C.read(OLD/'eval_manifest.json');jobs=[(str(OUT),ARM,q,'POST',C.sha(OUT/'registration.json')) for q in manifest['requests']]
    started=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool: rows=list(pool.map(E.score_one,jobs))
    C.write(OUT/'score_completion.json',dict(status='COMPLETE',cells=16,workers=workers,threads_per_worker=1,
        seconds=time.perf_counter()-started,model_calls=0,receipts=rows))
    return rows


def aggregate():
    registration();rows=[];paired=[];edges=[];pairs=[];funnel=collections.Counter()
    for q in C.read(OLD/'eval_manifest.json')['requests']:
        rid=q['request_id'];bundle={}
        for label,root in (('ORIGINAL_GEO',OLD),('CAPACITY_GUARD',CAP),('TIP_REDIRECT',OUT)):
            mesh=C.arrays(root/'evaluation'/ARM/rid/'post/post.npz')
            folder=root/'evaluation/scores'/ARM/rid/'POST';s=C.read(folder/'score.json')
            metrics={**s['metrics'],**s['topology'],**s['seam_to_bulk']}
            er,all_er=SC.read_edges(folder/'valid_edges.csv',mesh);pm=SC.pair_map(folder/'pairs.json',mesh)
            bundle[label]=dict(mesh=mesh,metrics=metrics,edges=er,all_edges=all_er,pairs=pm)
            row={k:q[k] for k in ('request_id','parent_id','noise_index','N','K','split')};row['label']=label
            row.update({k:v for k,v in metrics.items() if v is None or isinstance(v,(int,float,bool,str))});rows.append(row)
        for label in ('ORIGINAL_GEO','CAPACITY_GUARD'):
            a,b=bundle[label],bundle['TIP_REDIRECT']
            for key in sorted(a['metrics'].keys() & b['metrics'].keys()):
                av,bv=a['metrics'][key],b['metrics'][key]
                if isinstance(av,(int,float)) and isinstance(bv,(int,float)):
                    paired.append(dict(request_id=rid,parent_id=q['parent_id'],reference=label,metric=key,
                                       reference_value=av,candidate_value=bv,delta=bv-av))
            for key in sorted(a['edges'].keys()|b['edges'].keys()):
                ae,be=a['edges'].get(key),b['edges'].get(key)
                edges.append(dict(request_id=rid,reference=label,source_slot_corner_identity=key,
                    edge_kind='CF' if any(s<q['K'] for s,_ in key) else 'FF',
                    state='RETAINED' if ae and be else 'LOST' if ae else 'NEW',
                    reference_length=ae['length_model'] if ae else 0,candidate_length=be['length_model'] if be else 0))
            for key in sorted(a['pairs'].keys()|b['pairs'].keys()):
                ac,bc=a['pairs'].get(key),b['pairs'].get(key)
                pairs.append(dict(request_id=rid,reference=label,pair_kind=key[0],source_slot_pair=key[1],
                    role_pair=SC.role_pair(key[1],q['K'],a['mesh']['raw_seam_slot_mask']),
                    state='RETAINED_ILLEGAL' if ac and bc else 'ELIMINATED' if ac else 'NEW_ILLEGAL'))
        summary=C.read(OUT/'evaluation'/ARM/rid/'post/redirection.json');funnel.update(summary['funnel'])
        # Whole-endpoint safeguards complement incremental tests, never hide failures.
        a,b=bundle['CAPACITY_GUARD'],bundle['TIP_REDIRECT']
        assert not (b['pairs'].keys()-a['pairs'].keys()),rid
        assert np.array_equal(a['mesh']['source_face_ids'],b['mesh']['source_face_ids']),rid
        assert b['metrics']['C_F_nonmanifold_edge_count']<=a['metrics']['C_F_nonmanifold_edge_count'],rid
    C.csvwrite(OUT/'per_output.csv',rows);C.csvwrite(OUT/'paired_per_noise.csv',paired)
    C.csvwrite(OUT/'legal_edge_source_identity_ledger.csv',edges);C.csvwrite(OUT/'illegal_pair_source_identity_ledger.csv',pairs)
    summary=[]
    for ref,metric in sorted({(r['reference'],r['metric']) for r in paired}):
        sub=[r for r in paired if (r['reference'],r['metric'])==(ref,metric)]
        summary.append(dict(reference=ref,metric=metric,n=len(sub),reference_mean=float(np.mean([r['reference_value'] for r in sub])),
            candidate_mean=float(np.mean([r['candidate_value'] for r in sub])),mean_delta=float(np.mean([r['delta'] for r in sub])),
            improved_numeric_down=sum(r['delta']<0 for r in sub),increased=sum(r['delta']>0 for r in sub)))
    C.csvwrite(OUT/'summary.csv',summary);C.write(OUT/'rejection_funnel.json',dict(funnel))
    C.write(OUT/'numerical_completion.json',dict(status='COMPLETE',outputs=16,new_scores=16,model_calls=0,
        complete_endpoint_safety_checks='PASS',funnel=dict(funnel)))
    return summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('register','geometry','score','aggregate','all'),required=True)
    a=ap.parse_args()
    for fn in ([registration,geometry,score,aggregate] if a.stage=='all' else [dict(register=registration,geometry=geometry,score=score,aggregate=aggregate)[a.stage]]):fn()
    if 'torch' in sys.modules:raise AssertionError('CPU-only stage imported torch')


if __name__=='__main__':main()
