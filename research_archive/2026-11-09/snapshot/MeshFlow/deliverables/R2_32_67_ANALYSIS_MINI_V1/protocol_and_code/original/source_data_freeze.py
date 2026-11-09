"""Join actual input reviews and frozen CPU source scans; never waive the scale gate."""
from __future__ import annotations
import csv,io,itertools,sys
from collections import defaultdict,Counter
from data_prepare import OUT,ROOT,old,read,save,sha,rank,proposal
from quality_scan import partition
import numpy as np
sys.path.insert(0,str(ROOT/'r2/src'))
from meshflow_control.source_winding import capture_source,validate_source

def csv_save(path,rows):
    rows=list(rows);keys=list(dict.fromkeys(k for r in rows for k in r));s=io.StringIO(newline='');w=csv.DictWriter(s,keys);w.writeheader()
    for r in rows:w.writerow({k:json_value(v) for k,v in r.items()})
    old.write_bytes(path,('\ufeff'+s.getvalue()).encode('utf-8'))
def json_value(v):
    import json
    return json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v
def accepted(s):return s in ('ACCEPT','ACCEPT_AI_PROXY','ACCEPTED','ACCEPTED_AI_PROXY','ACCEPTED_WITH_RISK_AI_PROXY')

def reviews():
    files=[OUT/'reviews/reviewer_existing.json']+sorted((OUT/'data/reviews').glob('reviewer_v2_*.json'))
    # First-generation candidates can supplement only if they were actually reviewed and accepted.
    first=OUT/'data/reviews/reviewer_060_119.json'
    if first.exists():files.append(first)
    result=[];coverage=set()
    for fp in files:
        d=read(fp)
        for r in d.get('rows',[]):
            viewed=r.get('actually_viewed',r.get('reviewed',r.get('actual_image_viewed',False)))
            if not viewed:continue
            uid=r.get('uid',r.get('parent_uid'));variant=r['variant_uid']
            if 'v2_' in fp.name:coverage.add(uid)
            image=r.get('image',r.get('image_path',r.get('viewed_image')))
            image_hash=r.get('image_sha256',r.get('viewed_image_sha256'))
            if image and sha(image)!=image_hash:raise ValueError('Reviewed image hash changed')
            priority=0 if fp.name=='reviewer_existing.json' else 1 if 'v2_' in fp.name else 2
            if 'conditions' in r and 'regions' not in r:
                for typ,c in r['conditions'].items():
                    kind,tier=typ.split('_')
                    result.append(dict(uid=uid,variant_uid=variant,kind=kind,tier=tier,ids=c['source_face_ids'],region_ids=r['proposed_ids'][kind],
                        review_status=c['status'],reason=c['reason'],priority=priority,review_path=str(fp),review_sha256=sha(fp),image=image,image_sha256=image_hash,
                        structure_tags=r.get('tags',[]),region_declared='upper sitting surface' if kind=='SEAT' else 'seat-facing supporting surface / front frame or rails'))
                continue
            region_rows=r['regions']
            if isinstance(region_rows,list):
                for reg in region_rows:
                    for c in reg['conditions']:
                        kind,tier=c['condition_type'].split('_')
                        result.append(dict(uid=uid,variant_uid=variant,kind=kind,tier=tier,ids=c['source_face_ids'],region_ids=reg['source_face_ids'],
                            review_status=c['status'],reason=c['reason'],priority=priority,review_path=str(fp),review_sha256=sha(fp),
                            image=image,image_sha256=image_hash,structure_tags=r['structure_tags'],region_declared=reg.get('declared_surface',kind)))
            else:
                for kind,reg in region_rows.items():
                    if kind not in ('SEAT','BACK'):continue
                    for tier in ('FULL','HALF'):
                        if 'full' in reg:
                            z=reg[tier.lower()];ids=z['source_face_ids'];status=z['status'];reason=z['reason'];region_ids=reg['full']['source_face_ids']
                        else:
                            ids=reg['proposed_ids'] if tier=='FULL' else reg['half_ids'];status=reg[tier.lower()+'_status'];reason=reg[tier.lower()+'_reason'];region_ids=reg['proposed_ids']
                        result.append(dict(uid=uid,variant_uid=variant,kind=kind,tier=tier,ids=ids or [],region_ids=region_ids or [],review_status=status,reason=reason,
                            priority=priority,review_path=str(fp),review_sha256=sha(fp),image=image,image_sha256=image_hash,structure_tags=r.get('structure_tags',[]),
                            region_declared='upper sitting surface' if kind=='SEAT' else 'seat-facing supporting surface / front frame or rails'))
    return result,coverage,files

def components_of(a,ids):
    f=a['source_faces'];edge=defaultdict(list);adj={int(i):set() for i in ids}
    for i in ids:
        for j in range(3):edge[tuple(sorted([int(f[i,j]),int(f[i,(j+1)%3])]))].append(int(i))
    for members in edge.values():
        for i in members:adj[i].update(set(members)-{i})
    return len(proposal.components(ids,adj))

def emit(parent,record):
    uid=parent['uid'];ids=np.asarray(sorted(record['ids']),np.int64);n=parent['N'];free=np.setdiff1d(np.arange(n),ids)
    with np.load(parent['npz'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    tri=a['full_target'].reshape(-1,3,3);raw_faces=a['source_faces'][ids];corners=a['source_corner_permutations'][ids]
    original=a['model_vertices'][raw_faces];original_ids=a['retained_to_original_face'][ids]
    arrays=capture_source(original,raw_faces,corners,original_ids)
    assert arrays['C'].tobytes()==tri[ids].tobytes();validate_source(arrays,tri[ids])
    arrays.update(parent_uid=np.asarray(uid),condition_type=np.asarray(record['kind']+'_'+record['tier']),
        source_face_ids=ids,free_source_ids=free,N=np.asarray(n),K=np.asarray(len(ids)),
        region_source_face_ids=np.asarray(sorted(record['region_ids']),np.int64),
        C_original_raw_coordinates=a['source_vertices'][raw_faces].astype(np.float32))
    cid=uid+'__'+record['kind']+'_'+record['tier'];dest=OUT/'data/conditions'/(cid+'.npz');old.save_npz(dest,arrays)
    ar=np.linalg.norm(np.cross(tri[:,1].astype(float)-tri[:,0],tri[:,2].astype(float)-tri[:,0]),axis=1)/2
    region_ids=record['region_ids'];pa=record['quality'];summary={k:dict(illegal=v['confirmed_illegal_pair_count'],unknown=v['unknown_pair_count']) for k,v in pa['partitions'].items()}
    row=dict(task_id=cid,condition_id=cid,parent_id=uid,uid=uid,group_uid=parent['group_uid'],official_split=parent['official_split'],variant_uid=parent['variant_uid'],
        N=n,K=len(ids),source_face_ids=ids.tolist(),free_source_ids=free.tolist(),region_source_face_ids=region_ids,
        original_source_face_ids=original_ids.tolist(),original_vertex_ids=np.unique(raw_faces).tolist(),
        npz=str(dest),sha256=sha(dest),condition_source=str(dest),condition_source_sha256=sha(dest),
        condition_type=record['kind']+'_'+record['tier'],region_type=record['kind'],tier=record['tier'],
        K_over_N=len(ids)/n,area_C_over_M=float(ar[ids].sum()/ar.sum()),area_C_over_region=float(ar[ids].sum()/ar[region_ids].sum()),
        C_components=components_of(a,ids),Y_components=components_of(a,free),
        CC_CF_FF=summary,source_quality=pa['whole_Y_risk'],structure_tags=record['structure_tags'],
        annotation_source=record['review_path'],annotation_sha256=record['review_sha256'],review_kind='AI_PROXY',review_reason=record['reason'],
        image=record['image'],image_sha256=record['image_sha256'],region_declared=record['region_declared'],
        conditional_pretrain_seen=parent['conditional_pretrain_seen'],official_pretrain_seen='UNKNOWN',
        quality_path=record['quality_path'],quality_sha256=sha(record['quality_path']))
    return row

def run():
    parents=read(OUT/'data/parent_candidates.json')['rows'];lookup={(r['uid'],r['variant_uid']):r for r in parents}
    records,coverage,review_files=reviews()
    expected={r['uid'] for r in parents}
    if coverage!=expected:raise RuntimeError(f'Input review incomplete: actual {len(coverage)} / {len(expected)}; no freeze allowed')
    assessments=[];byparent=defaultdict(list)
    for r in records:
        p=lookup[r['uid'],r['variant_uid']];qpath=OUT/'data/quality'/(p['uid']+'__'+p['variant_uid']+'.json')
        r=dict(r,quality_path=str(qpath));reasons=[]
        if not accepted(r['review_status']):reasons.append('SEMANTIC_'+r['review_status'])
        if not r['ids']:reasons.append('NO_FACES')
        if r['ids'] and len(r['ids'])<p['N']:
            r['quality']=partition(qpath,r['ids'])
            if not r['quality']['local_CC_CF_qualified']:reasons.append('CC_CF_ILLEGAL_OR_UNKNOWN')
        else:reasons.append('INVALID_K')
        r['eligibility_reasons']=reasons;r['eligible']=not reasons;assessments.append(r)
        if not reasons:byparent[p['uid']].append(r)
    selected=[];selected_parent=[]
    for uid,recs in sorted(byparent.items()):
        variants=sorted({r['variant_uid'] for r in recs},key=lambda v:(abs(lookup[uid,v]['N']-192),v));v=variants[0];p=lookup[uid,v]
        recs=[r for r in recs if r['variant_uid']==v];chosen=[];seen=set()
        # One declared region per part: prior trusted mapping > visibility proposal > initial patch proposal.
        for kind in ('SEAT','BACK'):
            kr=[r for r in recs if r['kind']==kind]
            if not kr:continue
            priorities=sorted(set(r['priority'] for r in kr));pool=[r for r in kr if r['priority']==priorities[0]]
            for tier in ('FULL','HALF'):
                tr=[r for r in pool if r['tier']==tier]
                if not tr:continue
                rr=sorted(tr,key=lambda r:(r['review_path'],tuple(r['ids'])))[0];key=tuple(sorted(rr['ids']))
                if key in seen:continue
                seen.add(key);chosen.append(rr)
        if chosen:selected_parent.append(p);selected.extend((p,r) for r in chosen)
    tasks=[emit(p,r) for p,r in selected];by_uid=defaultdict(list)
    for t in tasks:by_uid[t['uid']].append(t)
    dev_candidates=[p for p in selected_parent if p['new_dev_eligible'] and len(by_uid[p['uid']])>=2]
    dev_candidates.sort(key=lambda p:rank('new_dev_order',p['group_uid']));dev=dev_candidates[:32];dev_ids={p['uid'] for p in dev}
    train=[p for p in selected_parent if p['uid'] not in dev_ids];train.sort(key=lambda p:p['candidate_rank']);train=train[:256]
    small=sorted(train,key=lambda p:rank('small_nested_order',p['group_uid']))[:32]
    m=len(train);status='FROZEN' if m>=128 and len(dev)==32 and len(small)==32 else 'DATA_SCALE_BLOCKED'
    large_label='USER_LARGE_'+str(m);panel=[]
    for i,p in enumerate(dev):
        rows=by_uid[p['uid']];types={r['condition_type']:r for r in rows};wanted=['SEAT_FULL','BACK_HALF'] if i<16 else ['SEAT_HALF','BACK_FULL']
        pick=[types[k] for k in wanted if k in types]
        for kind in ('SEAT','BACK'):
            if not any(r['region_type']==kind for r in pick):
                opt=[r for r in rows if r['region_type']==kind and r not in pick]
                if opt:pick.append(sorted(opt,key=lambda r:r['condition_type'])[0])
        for r in sorted(rows,key=lambda r:r['condition_type']):
            if len(pick)>=2:break
            if r not in pick:pick.append(r)
        assert len(pick)>=2;panel+=pick[:2]
    train_ids={p['uid'] for p in train};train_tasks={r['task_id']:r for r in tasks if r['uid'] in train_ids}
    pm={p['uid']:p for p in train}
    manifest=dict(schema='r2_user_scaling_data_v1',status=status,parents=pm,tasks=train_tasks,
        pools={'USER_SMALL':[p['uid'] for p in small],large_label:[p['uid'] for p in train]},
        no_scientific_run=status!='FROZEN',parent_sampling='balanced without replacement within each effective batch',condition_sampling='uniform deduplicated valid conditions per parent')
    save(OUT/'data/train_manifest.json',manifest);save(OUT/'data/panel_manifest.json',dict(status=status,conditions=panel,dev_parents=dev,gaussian_status='NOT_MATERIALIZED_BEFORE_DATA_GATE'))
    save(OUT/'data/condition_eligibility.json',dict(schema='review_plus_exact_CC_CF_v1',rows=assessments))
    save(OUT/'data/qualified_condition_manifest.json',dict(status='CPU_DATA_COMPLETE',parents=selected_parent,conditions=tasks))
    csv_save(OUT/'data/qualified_conditions.csv',tasks)
    csv_save(OUT/'data/condition_decisions.csv',[dict(uid=r['uid'],variant_uid=r['variant_uid'],condition_type=r['kind']+'_'+r['tier'],source_face_ids=r['ids'],region_source_face_ids=r['region_ids'],review_status=r['review_status'],review_reason=r['reason'],eligible=r['eligible'],reasons=r['eligibility_reasons'],quality=r.get('quality'),review_source=r['review_path']) for r in assessments])
    csv_save(OUT/'data/parent_decisions.csv',[dict(uid=p['uid'],group_uid=p['group_uid'],variant_uid=p['variant_uid'],N=p['N'],new_dev_eligible=p['new_dev_eligible'],
        reviewed=p['uid'] in coverage,condition_count=len([t for t in tasks if t['uid']==p['uid'] and t['variant_uid']==p['variant_uid']]),
        selected_split=('DEV' if p['uid'] in dev_ids else 'TRAIN' if p['uid'] in train_ids else 'EXCLUDED') if any(z['uid']==p['uid'] and z['variant_uid']==p['variant_uid'] for z in selected_parent) else 'EXCLUDED_OR_UNSELECTED_VARIANT') for p in parents])
    summary=dict(schema='r2_user_scaling_data_freeze_v1',status=status,train_manifest=dict(path=str(OUT/'data/train_manifest.json'),sha256=sha(OUT/'data/train_manifest.json')),
        panel_manifest=dict(path=str(OUT/'data/panel_manifest.json'),sha256=sha(OUT/'data/panel_manifest.json')),large_label=large_label,
        train_parent_count=m,small_parent_count=len(small),dev_parent_count=len(dev),qualified_total_parents=len(selected_parent),
        qualified_total_conditions=len(tasks),train_conditions=len(train_tasks),dev_conditions=len(panel),new_dev_eligible_parents_with_two_conditions=len(dev_candidates),
        inspected_parent_count=len(coverage),source_variants_scanned=len(parents),
        train_parent_shortfall_minimum=max(0,128-m),train_parent_shortfall_target=max(0,256-m),dev_parent_shortfall=max(0,32-len(dev)),
        semantics_accepted_source_options=sum(accepted(r['review_status']) for r in records),eligible_source_options=sum(r['eligible'] for r in assessments),
        review_sources={str(f):sha(f) for f in review_files},condition_counts=dict(Counter(t['condition_type'] for t in tasks)),
        no_training_no_new_generation_yet=True,scientific_endpoint=None,
        decision='Proceed to preregistered preflight only if FROZEN; otherwise deliver actual data gap and stop. No upsampling parents, no definition relaxation.')
    save(OUT/'data/data_freeze.json',summary);print(json_value(summary),flush=True)

if __name__=='__main__':run()
