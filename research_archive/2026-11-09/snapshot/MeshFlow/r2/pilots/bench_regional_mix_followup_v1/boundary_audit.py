"""Cached ASSIST11 conflict/connection attribution; no new geometric tests.

Only set operations, original slot/corner indexing, and coordinate comparisons
are used. No GEO, renderer, scorer, intersection predicate, model or new labels.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv
import hashlib
import json
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'experiments/r2_bench_regional_mix_dual_v1'
OUT=ROOT/'experiments/r2_bench_regional_mix_followup_v1/boundary'
STAGES=('BASE_POST','CLEAN_FINAL_POST')


def resolve(path):
    text=str(path).replace('\\','/')
    for prefix in ('E:/MeshFlow/','/mnt/e/MeshFlow/'):
        if text.startswith(prefix):return ROOT/text[len(prefix):]
    p=Path(text);return p if p.is_absolute() else ROOT/p


def sha(path):
    h=hashlib.sha256()
    with resolve(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(resolve(path).read_text(encoding='utf-8-sig'))
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,set)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return x
def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(clean(value),f,ensure_ascii=False,indent=2,allow_nan=False)
def csvwrite(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(dict,list,tuple,set,np.ndarray)) else v for k,v in r.items()})


def state(ledger,new=True):
    edges={kind:set() for kind in ('CF','FF')};pairs={kind:set() for kind in ('CF','FF')}
    for row in ledger['edge_fates']:
        if row['fate'] in (('RETAINED','NEW') if new else ('RETAINED','LOST')):
            edges[row['kind']].add(tuple(tuple(x) for x in row['source_edge_key']))
    for row in ledger['illegal_pair_fates']:
        if row['fate'] in (('PERSISTENT','NEW') if new else ('PERSISTENT','FACE_REMOVED','RESOLVED_RETAINED')):
            pairs[row['kind']].add(tuple(sorted((row['raw_slot_a'],row['raw_slot_b']))))
    faces={r['raw_slot'] for r in ledger['face_fates'] if r['fate'] in (('RETAINED','NEWLY_RETAINED') if new else ('RETAINED','LOST'))}
    return dict(edges=edges,pairs=pairs,faces=faces)


def indexed_mesh(z):
    ids=z.get('parent_source_raw_slot',z['source_face_ids']);lookup={int(s):i for i,s in enumerate(ids) if s>=0}
    if len(lookup)!=int((ids>=0).sum()):raise ValueError('Repeated original slot')
    canonical={}
    for slot,index in lookup.items():
        perm=z['source_corner_ids'][index]
        if sorted(perm.tolist())!=[0,1,2]:raise ValueError('Corner identity not a permutation')
        arr=np.empty((3,3),dtype=z['output'].dtype);arr[perm]=z['output'][index];canonical[slot]=arr
    incidence=Counter(tuple(sorted((int(f[a]),int(f[b])))) for f in z['faces'] for a,b in ((0,1),(1,2),(2,0)))
    return dict(z=z,lookup=lookup,canonical=canonical,incidence=incidence)


def coordinates_differ(a,b,slot):
    if slot not in a['canonical'] or slot not in b['canonical']:return None,None
    x,y=a['canonical'][slot],b['canonical'][slot]
    return x.tobytes()!=y.tobytes(),float(np.linalg.norm(x.astype(float)-y.astype(float),axis=1).max())


def target_ids(mesh,slot):
    z=mesh['z'];arr=z.get('input_corner_target_input_ids',z.get('corner_target_input_ids'))
    return arr[slot].tolist() if arr is not None and slot<len(arr) else None


def edge_topology(mesh,key,pair_illegal):
    sides=[]
    for slot,c0,c1 in key:
        if slot not in mesh['lookup']:return dict(candidate_edge_state='FACE_REMOVED',candidate_edge_incidence=None,opposed=None)
        i=mesh['lookup'][slot];perm=mesh['z']['source_corner_ids'][i];face=mesh['z']['faces'][i]
        p0=int(np.flatnonzero(perm==c0)[0]);p1=int(np.flatnonzero(perm==c1)[0]);a,b=int(face[p0]),int(face[p1])
        if a==b:return dict(candidate_edge_state='COLLAPSED_ENDPOINTS',candidate_edge_incidence=None,opposed=None)
        directed=(a,b) if (p0+1)%3==p1 else (b,a)
        sides.append(directed)
    if set(sides[0])!=set(sides[1]):return dict(candidate_edge_state='NO_LONGER_SHARED_INDEX_EDGE',candidate_edge_incidence=None,opposed=None)
    edge=tuple(sorted(sides[0]));count=mesh['incidence'][edge];opposed=sides[0]==sides[1][::-1]
    if count!=2:reason='SHARED_EDGE_INCIDENCE_NOT_TWO'
    elif not opposed:reason='SHARED_EDGE_SAME_WINDING'
    elif pair_illegal:reason='SHARED_EDGE_BUT_CACHED_ILLEGAL_PAIR'
    else:reason='SHARED_EDGE_OTHER_LEGALITY_REASON_UNKNOWN'
    return dict(candidate_edge_state=reason,candidate_edge_incidence=count,opposed=opposed)


def run():
    if 'torch' in sys.modules:raise RuntimeError('CPU cached-only boundary')
    done=OUT/'COMPLETE.json'
    if done.exists():
        r=read(done)
        for p,h in r['files'].items():
            if sha(p)!=h:raise ValueError('Existing output changed')
        return r
    if OUT.exists():raise RuntimeError('Partial audit retained; no overwrite')
    inputs={}
    def checked(path,expected=None):
        p=resolve(path);h=sha(p)
        if expected is not None and h!=expected:raise ValueError('Source hash differs: '+str(p))
        inputs[str(p)]=h;return p
    def js(path,expected=None):return read(checked(path,expected))
    def arr(path,expected=None):
        with np.load(checked(path,expected),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    reg=js(SOURCE/'jobs.json');panel=js(SOURCE/'panel/manifest.json');rh=sha(SOURCE/'jobs.json')
    jobs={(j['request']['request_id'],j['label']):j for j in reg['jobs']}
    event_rows=[];face_rows=[];summary_rows=[]
    for item in panel['requests']:
        q=item['request'];rid=q['request_id'];k=int(q['K']);job=jobs[rid,'ASSIST11'];rawjob=jobs[rid,'RAW_ALL']
        cr=js(SOURCE/'runs'/job['job_id']/'complete.json');rr=js(SOURCE/'runs'/rawjob['job_id']/'complete.json')
        if any(c['status']!='COMPLETE' or c['registration_sha256']!=rh for c in (cr,rr)):raise ValueError('Incomplete source')
        selected=arr(job['proposal']['path'],job['proposal']['sha256'])['selected_RAW'].astype(bool)
        regions={slot:r['region_id'] for r in item['regions'] for slot in r['source_face_ids']}
        if set(np.flatnonzero(selected))!=set(regions):raise ValueError('ASSIST11 region union differs')
        def region(s):return 'C' if s<k else regions.get(s,'OUTSIDE')
        def origin(s):return 'C' if s<k else 'RAW' if selected[s] else 'D0'
        selection_path=ROOT/'experiments/r2_bench_four_routes_v1/hybrid/outputs'/rid/'selection.npz'
        olddone=js(selection_path.parent/'complete.json')
        clusters=arr(selection_path,olddone['files'][str(selection_path)])['cluster_ids']
        cluster_roles=defaultdict(set)
        for s,ids in enumerate(clusters):
            for cid in ids:cluster_roles[int(cid)].add(origin(s))
        mixed={c for c,roles in cluster_roles.items() if {'RAW','D0'}<=roles}
        rawcoords=arr(job['paths']['raw'],reg['protected'][str(resolve(job['paths']['raw']))])['raw']
        precoords=arr(job['paths']['pre'],reg['protected'][str(resolve(job['paths']['pre']))])['output']
        proposal=arr(job['proposal']['path'],job['proposal']['sha256'])['output']
        if proposal[~selected].tobytes()!=precoords[~selected].tobytes():raise ValueError('Nonselected input not exact D0')
        ledgers={};states={};meshes={}
        for stage in STAGES:
            ledgers[stage]=js(cr['artifacts']['ledger_'+stage],cr['files'][str(resolve(cr['artifacts']['ledger_'+stage]))])
            rawledger=js(rr['artifacts']['ledger_'+stage],rr['files'][str(resolve(rr['artifacts']['ledger_'+stage]))])
            states['A',stage]=state(ledgers[stage]);states['D',stage]=state(ledgers[stage],False);states['R',stage]=state(rawledger)
            if states['D',stage]!=state(rawledger,False):raise ValueError('Baseline ledger differs')
            key='base' if stage=='BASE_POST' else 'final'
            for label,record in [('A',cr),('R',rr)]:
                path=record['artifacts'][key];meshes[label,stage]=indexed_mesh(arr(path,record['files'][str(resolve(path))]))
            p=job['paths']['original_base' if stage=='BASE_POST' else 'original_final']
            meshes['D',stage]=indexed_mesh(arr(p,reg['protected'][str(resolve(p))]))
        def context(a,b,stage):
            origins=[origin(a),origin(b)];regions_=[region(a),region(b)]
            order={'C':0,'RAW':1,'D0':2};pair_origin='/'.join(sorted(origins,key=lambda t:order[t]))
            shared=set(clusters[a])&set(clusters[b]);ma=meshes['A',stage];md=meshes['D',stage];mr=meshes['R',stage]
            row=dict(raw_slot_a=a,raw_slot_b=b,origin_a=origins[0],origin_b=origins[1],origin_pair=pair_origin,
                region_a=regions_[0],region_b=regions_[1],region_pair='/'.join(sorted(regions_)),
                crosses_RAW_D0_selection_sources=set(origins)=={'RAW','D0'},
                shared_cached_RAW_GEO_clusters=sorted(shared),shared_cached_RAW_GEO_complete_edge=len(shared)>=2,
                a_touches_cached_mixed_cluster=bool(set(clusters[a])&mixed),b_touches_cached_mixed_cluster=bool(set(clusters[b])&mixed))
            for side,slot in [('a',a),('b',b)]:
                for label,name in [('D','D0'),('A','ASSIST11'),('R','RAW_ALL')]:row[f'{side}_survives_{name}']=slot in states[label,stage]['faces']
                changed,distance=coordinates_differ(md,ma,slot);row[f'{side}_output_differs_from_D0']=changed;row[f'{side}_max_shift_from_D0']=distance
                changed,distance=coordinates_differ(mr,ma,slot);row[f'{side}_output_differs_from_RAW_ALL']=changed;row[f'{side}_max_shift_from_RAW_ALL']=distance
                basechanged,dist=coordinates_differ(meshes['D','BASE_POST'],meshes['A','BASE_POST'],slot)
                oldid=target_ids(meshes['D','BASE_POST'],slot);newid=target_ids(meshes['A','BASE_POST'],slot)
                row[f'{side}_BASE_input_is_exact_D0']=not selected[slot]
                row[f'{side}_BASE_output_differs_from_D0']=basechanged
                row[f'{side}_BASE_representative_ids_D0']=oldid;row[f'{side}_BASE_representative_ids_ASSIST11']=newid
                row[f'{side}_BASE_representative_identity_changed']=None if oldid is None or newid is None else oldid!=newid
                row[f'{side}_same_D0_input_but_BASE_coordinates_changed']=(not selected[slot] and basechanged is True)
            return row
        for stage in STAGES:
            ledger=ledgers[stage];sa=states['A',stage];sr=states['R',stage];sd=states['D',stage]
            meta=dict(request_id=rid,parent_id=q['parent_id'],noise_index=q['noise_index'],stage=stage)
            local=[]
            for item_ in ledger['edge_fates']:
                if item_['fate']!='LOST':continue
                key=tuple(tuple(v) for v in item_['source_edge_key']);a,b=key[0][0],key[1][0];kind=item_['kind']
                pair=tuple(sorted((a,b)));row=dict(meta,entity='LOST_LEGAL_EDGE',kind=kind,source_edge_key=key,
                    old_length=item_['old_length'],**context(a,b,stage))
                row.update(edge_topology(meshes['A',stage],key,pair in sa['pairs'][kind]))
                row.update(RAW_ALL_edge_legal=key in sr['edges'][kind],RAW_ALL_pair_illegal=pair in sr['pairs'][kind],
                    both_faces_survive_ASSIST11=set(pair)<=sa['faces'],
                    edge_already_lost_at_BASE=key in states['D','BASE_POST']['edges'][kind] and key not in states['A','BASE_POST']['edges'][kind],
                    candidate_edge_legal_at_BASE=key in states['A','BASE_POST']['edges'][kind])
                local.append(row)
            for item_ in ledger['illegal_pair_fates']:
                if item_['fate']!='NEW':continue
                a,b=item_['raw_slot_a'],item_['raw_slot_b'];kind=item_['kind'];pair=tuple(sorted((a,b)))
                present=pair in sr['pairs'][kind];observable=set(pair)<=sr['faces']
                row=dict(meta,entity='NEW_ILLEGAL_PAIR',kind=kind,**context(a,b,stage),
                    RAW_ALL_pair_illegal=present,RAW_ALL_both_faces_survive=observable,
                    RAW_ALL_relation='ALREADY_ILLEGAL_IN_WHOLE_RAW_CHAIN' if present else 'BOTH_SURVIVE_NOT_ILLEGAL_IN_WHOLE_RAW_CHAIN' if observable else 'UNCOMPARABLE_RAW_ALL_FACE_REMOVED',
                    candidate_pair_illegal_at_BASE=pair in states['A','BASE_POST']['pairs'][kind],
                    D0_pair_illegal_at_BASE=pair in states['D','BASE_POST']['pairs'][kind],
                    pair_new_relative_D0_at_BASE=pair in states['A','BASE_POST']['pairs'][kind]-states['D','BASE_POST']['pairs'][kind],
                    both_faces_already_present_D0=set(pair)<=sd['faces'])
                local.append(row)
            for kind in ('CF','FF'):
                if sum(r['entity']=='LOST_LEGAL_EDGE' and r['kind']==kind for r in local)!=ledger['summary'][kind+'_lost_legal_edges']:raise AssertionError('Lost count mismatch')
                if sum(r['entity']=='NEW_ILLEGAL_PAIR' and r['kind']==kind for r in local)!=ledger['summary'][kind+'_new_illegal_pairs']:raise AssertionError('New pair count mismatch')
            event_rows.extend(local)
            summary_rows.append(dict(meta,**ledger['summary'],
                lost_edge_origins=dict(Counter(r['origin_pair'] for r in local if r['entity']=='LOST_LEGAL_EDGE')),
                new_FF_origins=dict(Counter(r['origin_pair'] for r in local if r['entity']=='NEW_ILLEGAL_PAIR' and r['kind']=='FF')),
                new_CF_origins=dict(Counter(r['origin_pair'] for r in local if r['entity']=='NEW_ILLEGAL_PAIR' and r['kind']=='CF'))))
            for f in ledger['face_fates']:
                face_rows.append(dict(meta,**f,origin=origin(f['raw_slot']),region=region(f['raw_slot'])))
    totals={}
    for stage in STAGES:
        subset=[r for r in event_rows if r['stage']==stage];groups={}
        for entity,kind in [('LOST_LEGAL_EDGE','FF'),('LOST_LEGAL_EDGE','CF'),('NEW_ILLEGAL_PAIR','FF'),('NEW_ILLEGAL_PAIR','CF')]:
            rr=[r for r in subset if r['entity']==entity and r['kind']==kind]
            groups[entity+'_'+kind]=dict(count=len(rr),by_origin=dict(Counter(r['origin_pair'] for r in rr)),
                by_region=dict(Counter(r['region_pair'] for r in rr)),by_parent=dict(Counter(r['parent_id'] for r in rr)),
                both_survive_ASSIST11=sum(r['a_survives_ASSIST11'] and r['b_survives_ASSIST11'] for r in rr),
                RAW_ALL_relation=dict(Counter(r.get('RAW_ALL_relation','NA_EDGE') for r in rr)),
                candidate_edge_state=dict(Counter(r.get('candidate_edge_state','NA_PAIR') for r in rr)),
                crosses_RAW_D0_selection_sources=sum(r['crosses_RAW_D0_selection_sources'] for r in rr),
                any_same_D0_input_but_BASE_coordinates_changed=sum(r['a_same_D0_input_but_BASE_coordinates_changed'] or r['b_same_D0_input_but_BASE_coordinates_changed'] for r in rr))
        totals[stage]=groups
    if [totals['CLEAN_FINAL_POST'][x]['count'] for x in ('LOST_LEGAL_EDGE_FF','NEW_ILLEGAL_PAIR_FF','NEW_ILLEGAL_PAIR_CF')]!=[23,37,7]:
        raise AssertionError('Prior report counts did not reproduce; stop rather than silently reinterpret')
    if 'torch' in sys.modules:raise RuntimeError('Unexpected model import')
    for p,h in inputs.items():
        if sha(p)!=h:raise ValueError('Input changed during audit')
    OUT.mkdir(parents=True)
    csvwrite(OUT/'per_event.csv',event_rows);csvwrite(OUT/'per_request.csv',summary_rows);csvwrite(OUT/'face_survival.csv',face_rows)
    definitions=dict(origin='Proposal source per original RAW slot. C is separate; ASSIST11 selects exactly P1 union P2.',
        boundary='RAW/D0 means a cross-source pair, not proof of local adjacency or a causal mechanism. Cached cluster/edge contact is separately reported.',
        RAW_ALL='Whole RAW after the unchanged complete backend at the same stage; not the unprocessed RAW generator output.',
        conflict_comparison='Existing illegal-pair ledgers only. Absent pair with both faces surviving is not claimed disjoint if the old detector had unknown relations.',
        propagation='An unselected D0 input may acquire different BASE coordinates/representative IDs because original GEO chooses representatives in the whole mixed soup. No new GEO call is used here.',
        retained_internal='RAW/RAW or D0/D0 classify coordinate origins. They can still touch a mixed cluster; no global causal isolation is claimed.',
        stage='FINAL and BASE are separately compared with their corresponding original D0 stage; FINAL membership at BASE is explicitly recorded.',
        excluded='Synthetic negative-ID bridge faces excluded from the inherited cross-branch original-slot ledger; original full scorers remain authoritative.')
    write(OUT/'summary.json',dict(status='COMPLETE',totals=totals,definitions=definitions,requests=6,parents=3))
    receipt=dict(status='COMPLETE',utc=datetime.now(timezone.utc).isoformat(),source_registration_sha256=rh,
        source_code_sha256=sha(__file__),inputs=inputs,requests=6,parents=3,event_rows=len(event_rows),
        original_FINAL_counts_reproduced=dict(lost_legal_FF=23,new_illegal_FF=37,new_illegal_CF=7),
        model_calls=0,backend_calls=0,GEO_calls=0,scorer_calls=0,intersection_calls=0,new_visual_labels=0,
        files={str(p):sha(p) for p in OUT.iterdir() if p.is_file()})
    write(done,receipt);return receipt


if __name__=='__main__':
    result=run();print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','files')},indent=2))
