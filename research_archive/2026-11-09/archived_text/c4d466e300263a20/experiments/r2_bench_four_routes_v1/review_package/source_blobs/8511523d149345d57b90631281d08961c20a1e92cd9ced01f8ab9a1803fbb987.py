"""Single frozen 46-request CPU selector + unchanged full postprocessing."""
from __future__ import annotations
from . import common as C,selector as S
import argparse, concurrent.futures, multiprocessing, resource, sys, time, traceback
import numpy as np

LABEL='HYBRID'
def modules():
    if 'torch' in sys.modules:raise RuntimeError('CPU-only import boundary violated')
    from meshflow_control import geo_c,geo_io,source_winding
    from r2.pilots.seam_slot_c200_v1 import postprocess as P
    from r2.pilots.bench_clean_multipart_seam_v1 import post_cpu as CPU
    from r2.pilots.bench_group_weight_match_v1 import evaluation as E
    return geo_c,P,CPU,E

def prepare():
    done=C.OUT/'registration.json'
    if done.exists():
        r=C.read(done);C.verify(r['protected']);C.verify(r['source_hashes']);return r
    from . import checks
    gate=checks.run()
    if gate['status']!='PASS':raise RuntimeError('CPU checks not passed')
    geo,P,CPU,E=modules();qs=E.requests();tasks=[]
    protected={str(E.MANIFEST):C.sha(E.MANIFEST)}
    for q in qs:
        raw=E.producer('M',q,'raw');post=E.producer('M',q,'post')
        paths=dict(raw=raw/'raw.npz',source=raw/'source_condition.npz',pre=post/'pre_geo.npz',
            original_base=post/'post.npz',original_base_pairs=E.score_dir('M',q,'BASE_POST')/'pairs.json',
            original_final=C.OLD/'cpu/M'/q['request_id']/'CLEAN_FINAL_POST/post.npz',
            original_final_pairs=E.score_dir('M',q,'CLEAN_FINAL_POST')/'pairs.json')
        for stage in ('RAW','BASE_POST','CLEAN_FINAL_POST'):
            paths[stage+'_row']=E.score_dir('M',q,stage)/'row.json'
            paths[stage+'_score']=E.score_dir('M',q,stage)/'score.json'
        paths.update(input=C.resolve(q['inference_npz']),metadata=C.resolve(q['metadata_npz']),query=C.resolve(q['query_descriptor']['path']))
        for p in paths.values():protected[str(p)]=C.sha(p)
        r=C.arrays(paths['raw']);pre=C.arrays(paths['pre'])['output'];k=int(q['K'])
        if r['raw'].shape!=(int(q['N']),3,3) or pre.shape!=r['raw'].shape or pre[:k].tobytes()!=r['raw'][:k].tobytes():raise ValueError('RAW/D0 canonical identity differs')
        if r['raw_seam_slot_mask'].any() or (r['raw_anchor_ids'][k:]>=0).any():raise ValueError('This candidate is Native only')
        tasks.append(dict(request=q,paths={k:str(v) for k,v in paths.items()},request_id=q['request_id']))
    code={str(p):C.sha(p) for p in C.CODE.glob('*.py')}
    code.update({str(p):C.sha(p) for p in (P.__file__,geo.__file__,CPU.__file__,CPU.H.__file__,CPU.alpha.__file__,CPU.holes.__file__,CPU.M.__file__)})
    code.update(CPU.H.Q.definition_hashes())
    reg=dict(status='FROZEN',schema=S.VERSION,utc=C.utc(),tasks=tasks,selector=S.CONFIG,L=C.L,
        source_hashes=code,protected=protected,baseline='Completed M, same 46 requests and full postprocessing',
        budget=dict(requests=46,RAW_GEO_mapping_calls=46,hybrid_GEO_calls=46,full_cascades=46,new_score_rows=92,
                    reference_score_rows=138,model_calls=0,D0_calls=0,new_noise=0,training_updates=0),
        cluster_mapping_only_not_RAW_GEO_scored_control=True,maximum_workers=2,threads_per_worker=1,
        graph_pairing='Generated RAW slots/corners; synthetic bridge identities excluded from cross-backend correspondence',
        no_GT_success=True,no_outcome_fallback=True)
    C.write(done,reg);return reg

def original_ledger(mesh,pairs,k,CPU):
    full=CPU.ledger(mesh,pairs,k)
    edges={CPU.H.Q._edge_key(r):r for r in full['edges'] if r['valid_edge'] and all(x['source_face_id']>=0 for x in r['source_incidents'])}
    bad=CPU.H.Q.illegal_pair_sets(pairs,full['source_face_ids'],k)
    bad={kind:{x for x in value if min(x)>=0} for kind,value in bad.items()}
    return full,edges,bad

def ledger_compare(old,new,oldpairs,newpairs,k,CPU):
    ol,oe,ob=original_ledger(old,oldpairs,k,CPU);nl,ne,nb=original_ledger(new,newpairs,k,CPU)
    edges=[];faces=[];pairs=[];summary={}
    for kind in ('CF','FF'):
        a={key:r for key,r in oe.items() if r['kind']==kind};b={key:r for key,r in ne.items() if r['kind']==kind}
        for key in sorted(set(a)|set(b)):
            edges.append(dict(kind=kind,source_edge_key=key,fate='RETAINED' if key in a and key in b else 'LOST' if key in a else 'NEW',
                old_length=a[key]['length_model'] if key in a else None,new_length=b[key]['length_model'] if key in b else None))
        oldids={i for i in ol['source_face_ids'] if i>=0};newids={i for i in nl['source_face_ids'] if i>=0}
        for pair in sorted(ob[kind]|nb[kind]):
            fate='PERSISTENT' if pair in ob[kind]&nb[kind] else 'NEW' if pair in nb[kind] else 'FACE_REMOVED' if not set(pair)<=newids else 'RESOLVED_RETAINED'
            pairs.append(dict(kind=kind,raw_slot_a=pair[0],raw_slot_b=pair[1],fate=fate))
        summary.update({kind+'_old_legal_edges':len(a),kind+'_retained_legal_edges':len(set(a)&set(b)),kind+'_lost_legal_edges':len(set(a)-set(b)),
            kind+'_new_legal_edges':len(set(b)-set(a)),kind+'_lost_length':sum(a[x]['length_model'] for x in set(a)-set(b)),
            kind+'_new_illegal_pairs':len(nb[kind]-ob[kind]),kind+'_resolved_illegal_pairs_retained':sum(p['kind']==kind and p['fate']=='RESOLVED_RETAINED' for p in pairs),
            kind+'_illegal_no_longer_observable_deleted':sum(p['kind']==kind and p['fate']=='FACE_REMOVED' for p in pairs)})
    oa={int(i):float(a) for i,a in ol['source_face_area'].items() if int(i)>=0};na={int(i):float(a) for i,a in nl['source_face_area'].items() if int(i)>=0}
    for i in sorted(set(oa)|set(na)):
        faces.append(dict(raw_slot=i,known=i<k,fate='RETAINED' if i in oa and i in na else 'LOST' if i in oa else 'NEWLY_RETAINED',old_area=oa.get(i),new_area=na.get(i)))
    summary.update(original_faces_lost=len(set(oa)-set(na)),original_faces_newly_retained=len(set(na)-set(oa)),
        old_bridge_faces=sum(i<0 for i in ol['source_face_ids']),new_bridge_faces=sum(i<0 for i in nl['source_face_ids']))
    return dict(summary=summary,edge_fates=edges,face_fates=faces,illegal_pair_fates=pairs,
        identity='Same generated RAW source slot/corner; BRIDGE excluded from cross-branch correspondence, still included in full scorer')

def one(task,regsha):
    geo,P,CPU,E=modules();q=task['request'];rid=q['request_id'];dest=C.OUT/'outputs'/rid;done=dest/'complete.json'
    if done.exists():
        r=C.read(done)
        if r['registration_sha256']!=regsha:raise ValueError('Registration differs')
        C.verify(r['files']);return r
    if dest.exists():raise RuntimeError('Partial geometry requires explicit engineering recovery: '+str(dest))
    C.write(dest/'started.json',dict(utc=C.utc(),registration_sha256=regsha,model_calls=0))
    tick=time.perf_counter();p=task['paths'];raw=C.arrays(p['raw']);pre=C.arrays(p['pre'])['output'];source=C.arrays(p['source']);k=int(q['K'])
    try:
        mapping=geo.process(raw['raw'],k)
        result=S.select(raw['raw'],pre,k,mapping,L=C.L)
        C.npz(dest/'selection.npz',dict(raw=raw['raw'],d0_pre=pre,output=result['output'],seed=result['seed'],
            selected_RAW=result['selected_RAW'],cluster_ids=result['cluster_ids'],region_ids=result['region_ids'],
            raw_corner_target_input_ids=mapping['corner_target_input_ids'],raw_corner_vertex_ids=mapping['corner_vertex_ids'],
            raw_slot_ids=np.arange(len(pre)),source_corner_ids=np.tile(np.arange(3),(len(pre),1))))
        C.csvwrite(dest/'per_face_selection.csv',result['per_face']);C.write(dest/'mixed_boundary.json',result['boundary'])
        C.write(dest/'selection_summary.json',result['summary'])
        output,canonical,post,g,orientation,anchors=P.geometry_from_proposal(raw,result['output'],source)
        if output.tobytes()!=result['output'].tobytes():raise AssertionError('Canonical selection changed in anchor restore')
        C.npz(dest/'pre_geo.npz',dict(output=output,known_faces=np.asarray(k),raw_anchor_ids=raw['raw_anchor_ids'],raw_seam_slot_mask=raw['raw_seam_slot_mask']))
        C.npz(dest/'post_canonical.npz',canonical);C.npz(dest/'post.npz',post)
        C.write(dest/'geometry.json',dict(stats=g['stats'],face_records=g['face_records'],RAW_GEO_mapping_stats=mapping['stats']))
        C.write(dest/'source_restoration.json',orientation['restoration_summary']);C.write(dest/'anchor_identity.json',anchors)
        pairs=CPU.H.scan(post['output'],k);C.write(dest/'pairs.json',pairs)
        cascade=CPU.process_one(q,LABEL,dest/'post.npz',dest/'pre_geo.npz',p['raw'],dest/'pairs.json',C.OUT/'cpu'/LABEL/rid,False,regsha)
        for stage,oldkey,oldpairkey,newpath,newpairpath in (
            ('BASE_POST','original_base','original_base_pairs',dest/'post.npz',dest/'pairs.json'),
            ('CLEAN_FINAL_POST','original_final','original_final_pairs',C.OUT/'cpu'/LABEL/rid/'CLEAN_FINAL_POST/post.npz',C.OUT/'cpu'/LABEL/rid/'CLEAN_FINAL_POST/pairs.json')):
            compare=ledger_compare(C.arrays(p[oldkey]),C.arrays(newpath),C.read(p[oldpairkey]),C.read(newpairpath),k,CPU)
            C.write(dest/('ledger_'+stage+'.json'),compare)
        if 'torch' in sys.modules:raise RuntimeError('Unexpected torch import')
        r=dict(status='COMPLETE',request_id=rid,registration_sha256=regsha,selection=result['summary'],
            seconds=time.perf_counter()-tick,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            model_calls=0,D0_calls=0,GEO_mapping_calls=1,GEO_selected_calls=1,cascade_scores=2,
            files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file()})
        cp=C.OUT/'cpu'/LABEL/rid/'complete.json';r['files'][str(cp)]=C.sha(cp);C.write(done,r)
        print('HYBRID_COMPLETE',rid,result['summary']['selected_RAW_faces'],round(r['seconds'],2),flush=True);return r
    except BaseException as error:
        C.write(dest/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),utc=C.utc(),automatic_retry=False));raise

def run():
    reg=prepare();path=C.OUT/'completions/geometry_scores.json'
    if path.exists():r=C.read(path);C.verify(r['files']);return r
    C.verify(reg['protected']);C.verify(reg['source_hashes']);rh=C.sha(C.OUT/'registration.json')
    tick=time.perf_counter();before=C.resources();C.write(C.OUT/'resources_before_dispatch.json',dict(utc=C.utc(),resources=before,maximum_workers=2,threads_per_worker=1,models=0))
    first=one(reg['tasks'][0],rh);after=C.resources()
    from r2.pilots.chair_back_transfer_v1.evaluate import worker_count
    workers=worker_count(after,first,2)
    C.write(C.OUT/'dispatch.json',dict(workers=workers,threads_per_worker=1,resources_after_first=after,first_peak_RSS_KiB=first['peak_RSS_KiB'],model_calls=0))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        records=[first,*pool.map(one,reg['tasks'][1:],[rh]*45)]
    C.verify(reg['protected']);C.verify(reg['source_hashes'])
    r=dict(status='COMPLETE',requests=len(records),new_scores=92,GEO_calls=92,model_calls=0,D0_calls=0,seconds=time.perf_counter()-tick,workers=workers,
        files={str(C.OUT/'outputs'/t['request_id']/'complete.json'):C.sha(C.OUT/'outputs'/t['request_id']/'complete.json') for t in reg['tasks']})
    C.write(path,r);return r

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','run'));args=parser.parse_args()
    print(prepare() if args.command=='prepare' else run())
