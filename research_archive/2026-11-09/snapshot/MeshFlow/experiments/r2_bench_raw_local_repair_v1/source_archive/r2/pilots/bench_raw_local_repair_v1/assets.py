"""Read-only frozen 12-request assets and RAW-slot conflict attribution.

This module never invokes a model, geometry processing, or a scorer. All input
paths refer to existing frozen outputs. Derived indices are saved separately.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v]='1'
import argparse, csv, hashlib, json, time
from collections import Counter
from pathlib import Path
import numpy as np
from . import common as C

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_raw_local_repair_v1'
BASE_MANIFEST=ROOT/'experiments/r2_bench_geo_only_acceptance_v1/manifest.json'
SHAPE=ROOT/'experiments/r2_bench_backend_shape_preservation_v1'
RELDEF=ROOT/'experiments/r2_bench_backend_relative_deformation_v1'
PARENT_PREFIXES=('a8','27','838c','91e','fd355','2d')

def resolve(path):
    s=str(path).replace('\\','/')
    if os.name=='nt' and s.startswith('/mnt/e/'):s='E:/'+s[7:]
    elif os.name!='nt' and s.startswith('E:/'):s='/mnt/e/'+s[3:]
    p=Path(s)
    return p if p.is_absolute() else ROOT/p

def read(path):return json.loads(resolve(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with resolve(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def arrays(path):
    with np.load(resolve(path),allow_pickle=False) as z:
        a={k:z[k].copy() for k in z.files}
    for v in a.values():v.setflags(write=False)
    return a

def selected_tasks():
    source=read(BASE_MANIFEST)['tasks'];out=[]
    for prefix in PARENT_PREFIXES:
        tasks=sorted([t for t in source if t['parent_id'].startswith(prefix)],key=lambda t:t['noise_index'])
        assert len(tasks)==2 and [t['noise_index'] for t in tasks]==[0,1]
        assert len({t['parent_id'] for t in tasks})==1
        out.extend(tasks)
    assert len({t['request_id'] for t in out})==12
    return out

def asset_paths(t):
    rid=t['request_id'];o=t['original'];p={
        'RAW':t['new_raw_path'],'SOURCE_C':t['source_condition_path'],
        'INFERENCE':t['request']['inference_npz'],'QUERY':t['request']['query_descriptor']['path'],
        'OFFICIAL_PROPOSAL':o['proposal_path'],'OFFICIAL_PRE_GEO':o['pre_geo_path'],
        'OFFICIAL_CANONICAL':o['base_canonical_path'],'OFFICIAL_BASE':o['base_post_path'],
        'OFFICIAL_FINAL':o['final_post_path'],'OFFICIAL_GEOMETRY':o['geometry_path'],
        'OFFICIAL_RAW_SCORE':o['raw_score_path'],'OFFICIAL_BASE_SCORE':o['base_score_path'],
        'OFFICIAL_FINAL_SCORE':o['final_score_path'],
        'GEO_ONLY_PRE':t['new_pre_geo_path'],'GEO_ONLY_BASE':t['new_post_path'],
        'GEO_ONLY_GEOMETRY':t['new_geometry_path'],
    }
    p['RAW_PAIRS']=resolve(o['raw_score_path']).with_name('pairs.json')
    p['OFFICIAL_PAIRS']=resolve(o['base_score_path']).with_name('pairs.json')
    p['OFFICIAL_FINAL_PAIRS']=resolve(o['final_score_path']).with_name('pairs.json')
    gscore=BASE_MANIFEST.parent/'scores'/rid/'production'
    p['GEO_ONLY_PAIRS']=gscore/'pairs.json';p['GEO_ONLY_SCORE']=gscore/'score.json'
    for name,base,label in [('SHAPE',SHAPE,'SHAPE_D0'),('RELDEF',RELDEF,'RELDEF_D0')]:
        g=base/'geometry'/rid;c=base/'cpu'/label/rid
        p.update({name+'_PROPOSAL':base/'evaluation/CANDIDATE_RAW'/rid/'proposal.npz',
            name+'_PRE_GEO':g/'pre_geo.npz',name+'_CANONICAL':g/'post_canonical.npz',
            name+'_BASE':g/'post.npz',name+'_PAIRS':g/'pairs.json',
            name+'_GEOMETRY':g/'geometry.json',name+'_FINAL':c/'CLEAN_FINAL_POST/post.npz',
            name+'_BASE_SCORE':c/'BASE_POST/scores/score.json',
            name+'_FINAL_SCORE':c/'CLEAN_FINAL_POST/scores/score.json',
            name+'_FINAL_PAIRS':c/'CLEAN_FINAL_POST/scores/pairs.json'})
    return {k:resolve(v) for k,v in p.items()}

def stage_maps(mesh,n):
    """Output face/corner -> RAW identities; -1 means synthetic/unknown.

    This is provenance, not a claim that GEO retained the original topology.
    Complete original input-corner -> merged vertex arrays remain referenced.
    """
    out=mesh.get('output',mesh.get('raw'))
    if out is None:raise ValueError('No mesh triangles')
    m=len(out);sf=np.asarray(mesh.get('parent_source_raw_slot',mesh.get('raw_slot_ids',mesh.get('source_face_ids'))),np.int64)
    if sf.shape!=(m,):raise ValueError('Output to RAW-slot map missing or wrong size')
    sc=np.asarray(mesh.get('source_corner_ids'),np.int64)
    if sc.shape!=(m,3):raise ValueError('Output to RAW-corner map missing or wrong size')
    real=(sf>=0)&(sf<n)
    if len(set(map(int,sf[real])))!=int(real.sum()):raise ValueError('Nonunique original RAW slot')
    if not np.all(np.sort(sc[real],axis=1)==np.arange(3)):raise ValueError('Original corner mapping is not a permutation')
    flat=np.full((m,3),-1,np.int64);flat[real]=3*sf[real,None]+sc[real]
    kept=np.zeros(n,bool);kept[sf[real]]=True
    result=dict(output_to_raw_slot=sf,output_to_raw_corner=sc,output_to_raw_flat_corner=flat,raw_slot_kept=kept)
    iv=mesh.get('input_corner_vertex_ids',mesh.get('corner_vertex_ids'))
    if iv is not None:
        if iv.shape!=(n,3):raise ValueError('Original input-corner map dimension mismatch')
        result['raw_corner_to_merged_vertex']=iv
    if 'faces' in mesh:result['output_corner_to_vertex']=mesh['faces']
    return result

def _unknown(row):
    cls=row.get('classification','').lower()
    return any(s in cls for s in ('unknown','unresolved','degenerate')) or row.get('detector',{}).get('resolved') is False

def cached_contacts(mesh,pairs,kind,k,n):
    """Translate local detector indices to RAW slots; no geometry is tested."""
    sf=stage_maps(mesh,n)['output_to_raw_slot'];scan=pairs[kind]
    expected=k*(len(sf)-k) if kind=='CF' else (len(sf)-k)*(len(sf)-k-1)//2
    if scan['tested_pairs']!=expected or scan['expected_full_pairs']!=expected or scan.get('affected_subset') is not None:
        raise ValueError('Full original pair scan required')
    contacts={}
    for row in scan['contacts']:
        i,j=map(int,row['pair']);i=i if kind=='CF' else k+i;j=k+j
        if not(0<=i<len(sf) and 0<=j<len(sf)):raise ValueError('Local detector face index out of bounds')
        raw=(int(sf[i]),int(sf[j]));raw=raw if kind=='CF' else tuple(sorted(raw))
        if min(raw)<0:raise ValueError('BASE contact unexpectedly references synthetic face')
        if kind=='CF' and not(raw[0]<k<=raw[1]):raise ValueError('CF index convention differs')
        if kind=='FF' and min(raw)<k:raise ValueError('FF index convention differs')
        contacts[raw]=dict(classification=row['classification'],illegal=bool(row.get('detector',{}).get('illegal',False)),unknown=_unknown(row),post_face_i=i,post_face_j=j)
    return contacts

def pair_transition_rows(task,kind):
    """Union of illegal/UNKNOWN pair identities in SHAPE/RELDEF BASE caches."""
    a=task['assets'];k,n=task['K'],task['N']
    meshes={s:arrays(a[s+'_BASE']['path']) for s in ('SHAPE','RELDEF')}
    contacts={s:cached_contacts(meshes[s],read(a[s+'_PAIRS']['path']),kind,k,n) for s in meshes}
    kept={s:stage_maps(meshes[s],n)['raw_slot_kept'] for s in meshes}
    interesting=set()
    for d in contacts.values():interesting.update(p for p,v in d.items() if v['illegal'] or v['unknown'])
    out=[]
    for pair in sorted(interesting):
        oldkept=all(kept['SHAPE'][list(pair)]);newkept=all(kept['RELDEF'][list(pair)])
        old=contacts['SHAPE'].get(pair,dict(classification='disjoint',illegal=False,unknown=False)) if oldkept else dict(classification='NA_FACE_NOT_RETAINED',illegal=None,unknown=False)
        new=contacts['RELDEF'].get(pair,dict(classification='disjoint',illegal=False,unknown=False)) if newkept else dict(classification='NA_FACE_NOT_RETAINED',illegal=None,unknown=False)
        retention='COMMON_RETAINED' if oldkept and newkept else ('NEWLY_RETAINED' if newkept else ('NO_LONGER_RETAINED' if oldkept else 'UNKNOWN'))
        if old['unknown'] or new['unknown']:transition='UNKNOWN'
        elif old['illegal'] and new['illegal']:transition='ILLEGAL_BOTH'
        elif new['illegal']:transition='NEW_ILLEGAL_COMMON_FACES' if oldkept else 'ILLEGAL_NEWLY_RETAINED_FACES'
        elif old['illegal']:transition='RESOLVED_COMMON_FACES' if newkept else 'NO_LONGER_OBSERVABLE_FACE_REMOVED'
        else:transition='UNKNOWN'
        out.append(dict(request_id=task['request_id'],parent_id=task['parent_id'],noise_index=task['noise_index'],kind=kind,raw_slot_i=pair[0],raw_slot_j=pair[1],
            shape_i_retained=bool(kept['SHAPE'][pair[0]]),shape_j_retained=bool(kept['SHAPE'][pair[1]]),
            reldef_i_retained=bool(kept['RELDEF'][pair[0]]),reldef_j_retained=bool(kept['RELDEF'][pair[1]]),
            retention_attribution=retention,conflict_transition=transition,
            shape_classification=old['classification'],reldef_classification=new['classification'],
            shape_illegal=old['illegal'],reldef_illegal=new['illegal'],shape_unknown=old['unknown'],reldef_unknown=new['unknown']))
    return out

def load_manifest(verify=True):
    m=read(OUT/'data/manifest.json')
    if verify:
        C.verify(m['protected_assets']);C.verify(m['derived_files'])
    return m

def load_request(request_id,manifest=None,verify=True):
    """Read-only cached arrays. Caller must copy any array it intends to modify."""
    m=manifest or load_manifest(verify=verify)
    t=next(t for t in m['tasks'] if t['request_id']==request_id)
    if verify:C.verify({a['path']:a['sha256'] for a in t['assets'].values() if a['status']=='AVAILABLE'})
    data={name:arrays(a['path']) for name,a in t['assets'].items() if a['status']=='AVAILABLE' and resolve(a['path']).suffix=='.npz'}
    maps=arrays(t['slot_corner_maps']['path'])
    return dict(task=t,arrays=data,maps=maps)

def build_manifest():
    dest=OUT/'data';done=dest/'assets_COMPLETE.json'
    if done.exists():
        receipt=read(done);C.verify(receipt['files']);load_manifest(True);return receipt
    dest.mkdir(parents=True,exist_ok=True)
    attempt=dest/'assets_attempt.json'
    if attempt.exists():raise RuntimeError('Incomplete asset construction retained; no silent overwrite')
    C.write(attempt,dict(started_unix=time.time(),source_manifest=str(BASE_MANIFEST),source_sha256=sha(BASE_MANIFEST),cpu_threads=1,model_calls=0,scoring_calls=0,geometry_calls=0))
    tick=time.perf_counter();protected={str(BASE_MANIFEST):sha(BASE_MANIFEST)};tasks=[];pairs=[];mapping_records=[];derived={}
    for t in selected_tasks():
        task={key:t[key] for key in ('request_id','parent_id','condition_id','noise_index','eval_group','K','N')}
        task['request']=t['request'];task['assets']={};payloads={}
        for name,p in asset_paths(t).items():
            if not p.exists():
                task['assets'][name]=dict(path=str(p),status='BLOCKED_MISSING_CACHE');continue
            h=sha(p);protected[str(p)]=h;meta=dict(path=str(p),sha256=h,bytes=p.stat().st_size,status='AVAILABLE')
            if p.suffix=='.npz':
                a=arrays(p);payloads[name]=a
                meta['array_schema']={key:dict(shape=list(value.shape),dtype=str(value.dtype)) for key,value in a.items()}
            task['assets'][name]=meta
        # These are indispensable for target construction and conflict provenance.
        for name in ('RAW','SOURCE_C','GEO_ONLY_BASE','SHAPE_BASE','RELDEF_BASE','SHAPE_PAIRS','RELDEF_PAIRS','SHAPE_PRE_GEO','RELDEF_PRE_GEO'):
            if task['assets'][name]['status']!='AVAILABLE':raise FileNotFoundError(name+': '+task['assets'][name]['path'])
        raw=payloads['RAW'];k,n=task['K'],task['N']
        assert raw['raw'].shape==(n,3,3) and int(raw['known_faces'])==k
        assert np.array_equal(raw['raw_slot_ids'],np.arange(n))
        for name in ('OFFICIAL_PRE_GEO','SHAPE_PRE_GEO','RELDEF_PRE_GEO'):
            assert np.array_equal(payloads[name]['output'][:k],raw['raw'][:k]),(task['request_id'],name,'C differs')
        maps={}
        for name in ('RAW','OFFICIAL_BASE','OFFICIAL_FINAL','GEO_ONLY_BASE','SHAPE_BASE','SHAPE_FINAL','RELDEF_BASE','RELDEF_FINAL'):
            if name not in payloads:continue
            vals=stage_maps(payloads[name],n)
            for key,value in vals.items():maps[name+'__'+key]=value
            mapping_records.append(dict(request_id=task['request_id'],stage=name,output_faces=len(vals['output_to_raw_slot']),retained_raw_slots=int(vals['raw_slot_kept'].sum()),synthetic_or_unknown_faces=int((vals['output_to_raw_slot']<0).sum()),
                input_corner_map_available='raw_corner_to_merged_vertex' in vals))
        mp=dest/'maps'/f"{task['request_id']}.npz";C.npz(mp,maps);derived[str(mp)]=sha(mp)
        task['slot_corner_maps']=dict(path=str(mp),sha256=sha(mp),schema='stage__field; RAW slot and per-face corner IDs; -1 only synthetic/unknown')
        tasks.append(task)
        for kind in ('CF','FF'):pairs.extend(pair_transition_rows(task,kind))
    pairpath=dest/'conflict_attribution.csv';C.csvwrite(pairpath,pairs);derived[str(pairpath)]=sha(pairpath)
    mapcsv=dest/'mapping_summary.csv';C.csvwrite(mapcsv,mapping_records);derived[str(mapcsv)]=sha(mapcsv)
    summary=[]
    for task in tasks:
        for kind in ('CF','FF'):
            r=[x for x in pairs if x['request_id']==task['request_id'] and x['kind']==kind]
            counts=Counter((x['retention_attribution'],x['conflict_transition']) for x in r)
            for (retention,transition),count in sorted(counts.items()):summary.append(dict(request_id=task['request_id'],parent_id=task['parent_id'],kind=kind,retention_attribution=retention,conflict_transition=transition,pairs=count))
    sp=dest/'conflict_attribution_summary.csv';C.csvwrite(sp,summary);derived[str(sp)]=sha(sp)
    m=dict(schema='bench_raw_local_repair_assets_v1',status='FROZEN',parent_order=list(PARENT_PREFIXES),requests=12,parents=6,tasks=tasks,
        source_manifest=dict(path=str(BASE_MANIFEST),sha256=sha(BASE_MANIFEST)),protected_assets=protected,derived_files=derived,
        pair_accounting=dict(stages=['SHAPE_BASE','RELDEF_BASE'],scope='Union of illegal or UNKNOWN cached CF/FF pairs; paired by two original RAW slots',
            absent_pair_policy='Full scan with retained faces: absent contact means disjoint. Missing face means NA, never resolved.',
            ownership='No source-parent reconstruction labels used; post face IDs are translated through source_face_ids to RAW slots.',
            source_corner_rule='source_corner_ids follows exported winding and maps to original RAW per-face corner order',
            retention_categories=['COMMON_RETAINED','NEWLY_RETAINED','NO_LONGER_RETAINED','UNKNOWN']),
        budget=dict(model_calls=0,scoring_calls=0,geometry_calls=0,new_noise=0),source_hashes={str(Path(__file__)):sha(__file__)})
    C.verify(protected);C.write(dest/'manifest.json',m)
    result=dict(status='COMPLETE',requests=12,parents=6,conflict_rows=len(pairs),protected_files=len(protected),seconds=time.perf_counter()-tick,
        source_hashes=m['source_hashes'],model_calls=0,scoring_calls=0,geometry_calls=0,new_noise=0,
        conflict_counts={'|'.join(k):v for k,v in sorted(Counter((x['kind'],x['retention_attribution'],x['conflict_transition']) for x in pairs).items())},
        files={**derived,str(dest/'manifest.json'):sha(dest/'manifest.json'),str(attempt):sha(attempt)})
    C.write(done,result);return result

def inspect():
    t=selected_tasks()[0]
    print(json.dumps({k:str(v) for k,v in asset_paths(t).items()},indent=2))
    for name,path in asset_paths(t).items():
        if path.suffix=='.npz' and path.exists():
            print(name,{k:(list(v.shape),str(v.dtype)) for k,v in arrays(path).items()})
    pair=read(asset_paths(t)['RELDEF_PAIRS'])
    print('PAIR_SCHEMA',{k:{a:(len(v) if isinstance(v,list) else v) for a,v in p.items() if a!='contacts'} for k,p in pair.items()})
    print('CONTACT', {k:p.get('contacts',[None])[:1] for k,p in pair.items()})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['inspect','build','verify']);a=parser.parse_args()
    if a.action=='inspect':inspect()
    elif a.action=='verify':
        m=load_manifest(True);print(json.dumps(dict(status='VERIFIED',requests=len(m['tasks']))))
    else:print(json.dumps(build_manifest(),indent=2))
