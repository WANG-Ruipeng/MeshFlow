"""CPU-only conservative exact-duplicate target cleanup; original assets read-only."""
from pathlib import Path
import argparse, collections, csv, hashlib, io, json, os, re, time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_target_cleanup_v1'
SOURCE = ROOT / 'experiments/r2_full_seat_learning_curve_v1/inputs/training_data.json'

def resolve(p):
    s = str(p).replace('\\', '/')
    if os.name == 'nt' and re.match(r'^/mnt/[a-z]/', s): s=s[5].upper()+':/'+s[7:]
    elif os.name != 'nt' and re.match(r'^[A-Za-z]:/', s): s='/mnt/'+s[0].lower()+s[2:]
    p=Path(s); return p if p.is_absolute() else ROOT/p

def read(p): return json.loads(resolve(p).read_text(encoding='utf-8-sig'))
def arrays(p):
    with np.load(resolve(p), allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
def sha(p): return hashlib.sha256(resolve(p).read_bytes()).hexdigest()
def same(a,b): return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,set)): return [clean(v) for v in x]
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,Path): return str(x)
    return x
def write(p,x):
    p=resolve(p); assert p.resolve().is_relative_to(OUT.resolve())
    p.parent.mkdir(parents=True, exist_ok=True)
    payload=(json.dumps(clean(x),ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    if p.exists(): assert p.read_bytes()==payload, 'Frozen output differs: '+str(p)
    else: p.write_bytes(payload)
def csvwrite(p,rows):
    rows=clean(rows); fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    s=io.StringIO(newline=''); w=csv.DictWriter(s,fieldnames=fields); w.writeheader()
    for r in rows:w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    p=resolve(p); assert p.resolve().is_relative_to(OUT.resolve());p.parent.mkdir(parents=True,exist_ok=True)
    payload=s.getvalue().encode('utf-8-sig')
    if p.exists(): assert p.read_bytes()==payload
    else:p.write_bytes(payload)
def savez(p,a):
    p=resolve(p); assert p.resolve().is_relative_to(OUT.resolve());p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        old=arrays(p);assert set(old)==set(a) and all(same(old[k],np.asarray(a[k])) for k in a)
    else:np.savez_compressed(p,**a)

def coordinate_key(tri):
    # Exact numerical FP32 geometry; signed zero is not a distinct geometric point.
    return tuple(sorted(tuple(float(x) for x in vertex) for vertex in tri))

def analyze(t):
    a=arrays(t['parent_npz']); c=arrays(t['condition_source']); q=read(t['source_quality_path']); proof=read(t['edge_proof_path'])
    n,k=int(t['N']),int(t['K']); full=a['full_target'].reshape(n,3,3)
    known=np.asarray(t['source_task']['source_face_ids'],np.int64); free=np.asarray(t['source_task']['free_source_ids'],np.int64)
    assert np.isfinite(full).all() and same(full[known],c['C'])
    assert same(a['model_vertices'][a['source_face_vertex_ids']],full)
    assert same(a['model_vertices'][a['source_faces'][known]],c['C_original'])
    assert np.array_equal(np.sort(np.r_[known,free]),np.arange(n)) and k>=2
    assert same(a['pre_ot_full']*np.float32(2),full)
    groups=collections.defaultdict(list)
    for fi,tri in enumerate(full):groups[coordinate_key(tri)].append(fi)
    owners=set(int(x['free_source_face_rows'][0]) for x in proof['legal_edges'])
    protected=set(known.tolist())|owners; removed=set();dups=[]
    for gi,ids in enumerate(sorted((ids for ids in groups.values() if len(ids)>1),key=lambda ids:ids[0])):
        locked=sorted(set(ids)&protected)
        retained=locked if locked else [min(ids)]
        drop=sorted(set(ids)-set(retained)); removed.update(drop)
        # C duplicates cannot be removed; free duplicates of C may be, but all C/Y owner faces are protected.
        for fi in drop:assert fi not in protected and fi in free
        dups.append(dict(group_index=gi,source_face_rows=ids,protected_source_rows=locked,retained_source_rows=retained,removed_source_rows=drop,
                         source_vertex_id_rows=a['source_faces'][ids].tolist(),canonical_vertex_id_rows=a['source_face_vertex_ids'][ids].tolist(),
                         source_winding_rows=a['source_faces'][ids].tolist(),reason='KEEP_ALL_INTERFACE_OWNERS_AND_C_ELSE_LOWEST_SOURCE_ROW'))
    kept=np.asarray([fi for fi in range(n) if fi not in removed],np.int64)
    old_to_new=np.full(n,-1,np.int64);old_to_new[kept]=np.arange(len(kept))
    keepfree=np.asarray([fi for fi in free if fi not in removed],np.int64)
    assert not owners&removed and not set(known)&removed
    assert {coordinate_key(tri) for tri in full}=={coordinate_key(tri) for tri in full[kept]}
    assert same(full[known],full[kept][old_to_new[known]])
    # No pair detector: subtract deleted rows from the old complete pair ledger.
    oldff=[];newff=[];oldunknown=[];newunknown=[]
    for row in q['pair_contacts']:
        i,j=map(int,row['pair'])
        if i in known or j in known:continue
        cat=row['category']; cls=row['classification']
        illegal=cat in ('proper','coplanar_positive_overlap','illegal') or cls in ('proper','coplanar_positive_overlap')
        unknown=cat in ('UNKNOWN','unknown') or cls=='UNKNOWN'
        if illegal:
            oldff.append(row)
            if i not in removed and j not in removed:newff.append(row)
        if unknown:
            oldunknown.append(row)
            if i not in removed and j not in removed:newunknown.append(row)
    eligible=128<=len(kept)<=256 and len(keepfree)>0
    summary=dict(task_id=t['task_id'],parent_id=t['parent_id'],N_original=n,N_clean=len(kept),K=k,removed_free_faces=len(removed),
        original_free_faces=n-k,clean_free_faces=len(keepfree),exact_duplicate_groups=len(dups),has_actual_change=bool(removed),
        retained_multiple_protected_duplicate_groups=sum(len(x['retained_source_rows'])>1 for x in dups),
        legal_interface_edges=len(proof['legal_edges']),legal_interface_owner_faces=len(owners),legal_interface_owners_lost=0,
        FF_illegal_source_pairs_original=len(oldff),FF_illegal_source_pairs_clean=len(newff),FF_unknown_source_pairs_original=len(oldunknown),FF_unknown_source_pairs_clean=len(newunknown),
        zero_confirmed_illegal_original=not oldff,zero_confirmed_illegal_clean=not newff,zero_illegal_and_unknown_clean=not newff and not newunknown,
        eligible_common_pool=eligible,exclusion_reason='' if eligible else 'CLEAN_N_OUTSIDE_128_256_OR_EMPTY_FREE',
        C_bitwise_preserved=True,positive_area_geometric_surface_union_preserved=True,source_vertices_unchanged=True,
        operation='EXACT_GEOMETRIC_DUPLICATE_FREE_FACE_REMOVAL_ONLY',pair_risk_method='FILTER_SAVED_COMPLETE_PAIR_TABLE_NO_NEW_GEOMETRY_SCAN')
    return dict(t=t,a=a,c=c,q=q,proof=proof,known=known,free=free,kept=kept,keepfree=keepfree,old_to_new=old_to_new,removed=sorted(removed),duplicates=dups,summary=summary,oldff=oldff,newff=newff,oldunknown=oldunknown,newunknown=newunknown)

def inspect():
    m=read(SOURCE);t=next(iter(m['tasks'].values()));a=arrays(t['parent_npz']);c=arrays(t['condition_source']);q=read(t['source_quality_path'])
    print(json.dumps(dict(parent_fields={k:[str(v.shape),str(v.dtype)] for k,v in a.items()},condition_fields={k:[str(v.shape),str(v.dtype)] for k,v in c.items()},quality_keys=list(q),quality_first=q['pair_contacts'][0]),indent=2))

def scan():
    m=read(SOURCE); results=[analyze(m['tasks'][tid]) for tid in sorted(m['train_task_ids'])]
    rows=[r['summary'] for r in results]
    print(json.dumps(dict(parents=len(rows),eligible=sum(r['eligible_common_pool'] for r in rows),changed=sum(r['eligible_common_pool'] and r['has_actual_change'] for r in rows),removed=sum(r['removed_free_faces'] for r in rows),rows=rows),indent=2))
    return results

def prepare():
    final=OUT/'data_manifest.json'
    if final.exists():
        m=read(final)
        for p,h in m['protected_source_hashes'].items():assert sha(p)==h,p
        for p,h in m['output_hashes'].items():assert sha(p)==h,p
        print(json.dumps(dict(status='VERIFIED_FROZEN',summary=m['summary'],manifest=str(final)),indent=2));return m
    m=read(SOURCE); source_manifest=read(m['source_training_data']);results=[analyze(m['tasks'][tid]) for tid in sorted(m['train_task_ids'])]
    assert len(results)==47
    assert sum(r['summary']['FF_illegal_source_pairs_original'] for r in results)==1138
    selected=[r for r in results if r['summary']['eligible_common_pool'] and r['summary']['has_actual_change']]
    assert len(selected)==16
    protected={str(SOURCE):sha(SOURCE),str(resolve(m['source_training_data'])):sha(m['source_training_data'])}; outputs={}; mappings=[]; duplicates=[]; interface_rows=[]; pair_rows=[]
    selected_ids={r['t']['task_id'] for r in selected}
    manifests={arm:dict(schema='r2_target_cleanup_training_data_v1',status='FROZEN',arm=arm,parents={},tasks={},pools={'COMMON_AFFECTED_16':sorted(r['t']['parent_id'] for r in selected)},
        selection='Same 16 TRAIN47 parents with at least one safely removable exact geometric free duplicate; no generated-outcome selection',
        condition_sampling='one frozen original coherent SEAT_FULL per parent',source_training_manifest=str(SOURCE),source_training_manifest_sha256=sha(SOURCE)) for arm in ('ORIGINAL','DEDUP')}
    for r in results:
        t,a,c=r['t'],r['a'],r['c'];tid=t['task_id'];uid=t['parent_id'];n=t['N'];orig_ids=a.get('retained_to_original_face',np.arange(n))
        for key,hkey in [('parent_npz','parent_npz_sha256'),('condition_source','condition_source_sha256'),('source_quality_path','source_quality_sha256'),('edge_proof_path','edge_proof_sha256'),('mask_path','mask_sha256')]:
            h=sha(t[key]);assert h==t[hkey],t[key];protected[str(resolve(t[key]))]=h
        assert r['q']['status']=='COMPLETE' and r['q']['independent_verification']
        assert r['q']['receipt']['new_unique_pair_tests']==n*(n-1)//2
        assert r['q']['input_npz_sha256']==sha(t['parent_npz'])
        assert r['q']['full_target_sha256']==hashlib.sha256(a['full_target'].reshape(n,3,3).tobytes()).hexdigest()
        area=np.linalg.norm(np.cross(a['full_target'].reshape(n,3,3)[:,1]-a['full_target'].reshape(n,3,3)[:,0],a['full_target'].reshape(n,3,3)[:,2]-a['full_target'].reshape(n,3,3)[:,0]),axis=1)*.5
        assert np.all(area>0)
        for x in r['duplicates']:
            ids=x['source_face_rows'];base=a['source_faces'][ids[0]]
            vertexsets=[tuple(sorted(map(int,a['source_faces'][fi]))) for fi in ids]
            x.update(task_id=tid,parent_id=uid,same_original_vertex_id_set=len(set(vertexsets))==1,
                     original_face_ids=[int(orig_ids[fi]) for fi in ids])
            duplicates.append(x)
        for fi in range(n):
            mappings.append(dict(task_id=tid,parent_id=uid,source_face_row=fi,original_face_id=int(orig_ids[fi]),clean_face_row=int(r['old_to_new'][fi]),
                role='C' if fi in r['known'] else 'FREE',removed=fi in r['removed'],protected_legal_CY_owner=any(fi in e['free_source_face_rows'] for e in r['proof']['legal_edges']),
                original_vertex_ids=a['source_faces'][fi].tolist(),canonical_vertex_ids=a['source_face_vertex_ids'][fi].tolist(),original_surface_area_model2=float(area[fi])))
        for e in r['proof']['legal_edges']:
            fi=int(e['free_source_face_rows'][0]);ci=int(e['C_source_face_rows'][0]);assert fi not in r['removed'] and ci not in r['removed']
            # Actual source-ID edge incidence remains precisely one C and one free.
            edge={int(e['source_vertex_u']),int(e['source_vertex_v'])}
            incidence=[int(j) for j in r['kept'] if edge.issubset(set(a['source_faces'][j].tolist()))]
            assert set(incidence)=={ci,fi}
            interface_rows.append(dict(task_id=tid,parent_id=uid,C_source_row=ci,free_source_row=fi,clean_C_row=int(r['old_to_new'][ci]),clean_free_row=int(r['old_to_new'][fi]),
                source_vertex_u=e['source_vertex_u'],source_vertex_v=e['source_vertex_v'],length_model=e['length_model'],source_winding_opposite=e['original_source_winding_opposite'],retained=True))
        for kind,old,new in [('ILLEGAL',r['oldff'],r['newff']),('UNKNOWN',r['oldunknown'],r['newunknown'])]:
            kept_pairs={tuple(x['pair']) for x in new}
            for x in old:pair_rows.append(dict(task_id=tid,parent_id=uid,risk=kind,source_pair=x['pair'],classification=x['classification'],category=x['category'],survives_cleanup=tuple(x['pair']) in kept_pairs))
        # Save all47 paired targets for reproducibility; only changed16 enter main manifests.
        for arm in ('ORIGINAL','DEDUP'):
            keep=np.arange(n) if arm=='ORIGINAL' else r['kept']; newn=len(keep)
            remap=np.full(n,-1,np.int64);remap[keep]=np.arange(newn)
            known=remap[r['known']];free=remap[np.asarray([i for i in r['free'] if remap[i]>=0],np.int64)]
            assert len(known)==t['K'] and np.array_equal(np.sort(np.r_[known,free]),np.arange(newn))
            parent={k:v.copy() for k,v in a.items() if k not in ('p0_constraints','p1_constraints','p2_constraints','p3_constraints')}
            for key in ('full_target','pre_ot_full','source_face_vertex_ids','source_faces','source_corner_permutations','retained_to_original_face','canonical_to_source_face'):
                if key in parent:parent[key]=parent[key][keep]
            if 'original_to_retained_face' in parent:
                old=parent['original_to_retained_face'];new=np.full_like(old,-1);valid=old>=0;new[valid]=remap[old[valid]];parent['original_to_retained_face']=new
            parent.update(cleanup_input_source_rows=keep,cleanup_input_N=np.asarray(n,np.int64),cleanup_removed_source_rows=np.asarray([] if arm=='ORIGINAL' else r['removed'],np.int64),
                          cleanup_policy=np.asarray('NO_CHANGE' if arm=='ORIGINAL' else 'EXACT_GEOMETRIC_FREE_DUPLICATES_ONLY_PROTECT_C_AND_CY_OWNERS'))
            cond={k:v.copy() for k,v in c.items()}
            cond.update(N=np.asarray(newn,np.int64),source_face_ids=known,free_source_ids=free,region_source_face_ids=known,
                cleanup_input_C_source_rows=r['known'],cleanup_input_free_source_rows=np.asarray([i for i in r['free'] if remap[i]>=0],np.int64))
            assert same(parent['full_target'].reshape(newn,3,3)[known],c['C'])
            assert same(cond['C_original'],c['C_original']) and same(parent['model_vertices'],a['model_vertices'])
            assert same(parent['pre_ot_full']*np.float32(2),parent['full_target'].reshape(newn,3,3))
            p=OUT/'datasets'/arm/'parents'/(uid+'.npz');cp=OUT/'datasets'/arm/'conditions'/(tid+'.npz')
            savez(p,parent);savez(cp,cond);outputs[str(p)]=sha(p);outputs[str(cp)]=sha(cp)
            loaded=arrays(p);loadedc=arrays(cp)
            assert all(same(loaded[key],np.asarray(v)) for key,v in parent.items())
            assert all(same(loadedc[key],np.asarray(v)) for key,v in cond.items())
            if tid not in selected_ids:continue
            tr=dict(t['source_task']);tr.update(N=newn,K=t['K'],npz=str(cp),sha256=sha(cp),source_face_ids=known.tolist(),free_source_ids=free.tolist(),region_source_face_ids=known.tolist(),
                area_C_over_region=1.0,cleanup_input_source_task_id=tid,cleanup_input_N=n,source_quality_scope='PRE_CLEANUP_INHERITED_ONLY; see paired_parent_summary.csv',cleanup_applied=arm=='DEDUP')
            manifests[arm]['tasks'][tid]=tr
            parent_row=dict(source_manifest['parents'][uid]);parent_row.update(uid=uid,N=newn,npz=str(p),sha256=sha(p),original_parent_npz=t['parent_npz'],original_parent_sha256=t['parent_npz_sha256'],
                clean_removed_face_rows=[] if arm=='ORIGINAL' else r['removed'],face_id_policy='compact training rows; original source vertex IDs unchanged; cleanup_input_source_rows gives reversible surviving-face mapping')
            manifests[arm]['parents'][uid]=parent_row
    for arm,mf in manifests.items():
        path=OUT/'datasets'/arm/'train_manifest.json';write(path,mf);outputs[str(path)]=sha(path)
    rows=[r['summary'] for r in results]
    csvwrite(OUT/'paired_parent_summary.csv',rows);csvwrite(OUT/'exact_duplicate_groups.csv',duplicates);csvwrite(OUT/'source_face_mapping.csv',mappings)
    csvwrite(OUT/'legal_interface_preservation.csv',interface_rows);csvwrite(OUT/'inherited_FF_pair_survival.csv',pair_rows)
    csvwrite(OUT/'zero_FF_feasibility.csv',[dict(x,main_training_selected=x['has_actual_change'] and x['eligible_common_pool'],certification='NO_DETECTED_ILLEGAL_AND_UNKNOWN_PAIR_ONLY_NOT_GLOBAL_MANIFOLD_QUALITY') for x in rows])
    for name in ('paired_parent_summary.csv','exact_duplicate_groups.csv','source_face_mapping.csv','legal_interface_preservation.csv','inherited_FF_pair_survival.csv','zero_FF_feasibility.csv'):outputs[str(OUT/name)]=sha(OUT/name)
    selected_rows=[r['summary'] for r in selected]
    summary=dict(examined_train_parents=47,main_common_parents=len(selected),unchanged_control_parents=sum(not r['has_actual_change'] for r in rows),excluded_N_parents=sum(not r['eligible_common_pool'] for r in rows),
        exact_duplicate_groups=len(duplicates),removed_free_faces=sum(r['removed_free_faces'] for r in rows),protected_duplicate_groups_left=sum(r['retained_multiple_protected_duplicate_groups'] for r in rows),
        N_original_range=[min(r['N_original'] for r in selected_rows),max(r['N_original'] for r in selected_rows)],N_clean_range=[min(r['N_clean'] for r in selected_rows),max(r['N_clean'] for r in selected_rows)],
        original_total_faces=sum(r['N_original'] for r in selected_rows),clean_total_faces=sum(r['N_clean'] for r in selected_rows),
        original_total_free_faces=sum(r['original_free_faces'] for r in selected_rows),clean_total_free_faces=sum(r['clean_free_faces'] for r in selected_rows),
        main_original_FF_illegal_pairs=sum(r['FF_illegal_source_pairs_original'] for r in selected_rows),main_clean_FF_illegal_pairs=sum(r['FF_illegal_source_pairs_clean'] for r in selected_rows),
        all47_original_FF_illegal_pairs=sum(r['FF_illegal_source_pairs_original'] for r in rows),all47_clean_FF_illegal_pairs=sum(r['FF_illegal_source_pairs_clean'] for r in rows),
        all47_source_legal_interfaces=len(interface_rows),lost_source_legal_interfaces=0,main_source_legal_interfaces=sum(r['legal_interface_edges'] for r in selected_rows),
        all47_zero_illegal_and_unknown_after=sum(r['zero_illegal_and_unknown_clean'] for r in rows),main_zero_illegal_and_unknown_after=sum(r['zero_illegal_and_unknown_clean'] for r in selected_rows),
        all47_zero_illegal_and_unknown_before=sum(r['zero_confirmed_illegal_original'] and r['FF_unknown_source_pairs_original']==0 for r in rows),
        protected_inputs=len(protected),output_files=len(outputs),source_geometric_pair_scans=0,model_calls=0,OT_calls=0)
    for p,h in protected.items():assert sha(p)==h,p
    write(final,dict(schema='r2_target_cleanup_data_manifest_v1',status='FROZEN',summary=summary,
        selection='Only16 actual-change parents, both arms share exact same parents and original coherent C; all47 audit retained',
        arm_manifests={arm:str(OUT/'datasets'/arm/'train_manifest.json') for arm in manifests},main_parent_ids=sorted(r['t']['parent_id'] for r in selected),
        main_task_ids=sorted(r['t']['task_id'] for r in selected),scientific_scope='Conservative exact geometric free duplicate removal; NOT whole-target cleanup, no D0, no intersection deletion, no coordinate changes, no retriangulation. N/free denominator changes are an inseparable documented part of this test.',
        topology_limit='Preserves source C and all original legal C/Y owners, source vertex IDs, and geometry union. Removing redundant source triangles may change free incidence/topology; no assertion of global topology preservation or watertightness.',
        protected_source_hashes=protected,output_hashes=outputs,preparation_code_sha256=sha(__file__),roundtrip_checks='all saved NPZ arrays byte-identical on reload; all C canonical/original arrays unchanged; preOTx2 and indexed source identity preserved; all366 original legal CF edge incidences retained'))
    print(json.dumps(dict(status='FROZEN',summary=summary,manifest=str(final)),indent=2));return read(final)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['inspect','scan','prepare'],default='prepare');args=p.parse_args()
    {'inspect':inspect,'scan':scan,'prepare':prepare}[args.stage]()
