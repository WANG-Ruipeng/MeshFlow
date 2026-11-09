"""Paired identity/repair accounting and local-label survival; cached data only."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import collections, csv, json
import numpy as np
from . import common as C, assets as A
from r2.pilots.bench_clean_multipart_seam_v1 import post_cpu as CPU


def rows(p):
    with C.resolve(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def original_contacts(mesh,pairs,k):
    sf=np.asarray(mesh.get('parent_source_raw_slot',mesh.get('raw_slot_ids',mesh.get('source_face_ids'))),int)
    out={kind:dict(illegal=set(),unknown=set(),bridge_illegal=0) for kind in ('CF','FF')}
    for kind,scan in pairs.items():
        for r in scan['contacts']:
            i,j=map(int,r['pair']);i=i if kind=='CF' else i+k;j+=k
            pair=(int(sf[i]),int(sf[j]));pair=pair if kind=='CF' else tuple(sorted(pair))
            illegal=bool(r.get('detector',{}).get('illegal',False))
            if min(pair)<0:
                out[kind]['bridge_illegal']+=int(illegal);continue
            if illegal:out[kind]['illegal'].add(pair)
            if A._unknown(r):out[kind]['unknown'].add(pair)
    return out


def slot_data(mesh):
    tri=mesh.get('output',mesh.get('raw'))
    sf=np.asarray(mesh.get('parent_source_raw_slot',mesh.get('raw_slot_ids',mesh.get('source_face_ids'))),int)
    sc=mesh['source_corner_ids'];out={}
    for i,s in enumerate(sf):
        if s<0:continue
        if int(s) in out:raise ValueError('Duplicate original slot in processing result')
        t=np.empty_like(tri[i]);t[sc[i]]=tri[i]
        out[int(s)]=t
    return out


def area(t):return float(np.linalg.norm(np.cross(t[1]-t[0],t[2]-t[0]))*.5)


def legal_original_edges(mesh,pairs,k):
    """Original face/corner identities only; no synthetic bridge identity pairing."""
    ledger=CPU.ledger(mesh,pairs,k)
    return {tuple(tuple(map(int,x)) for x in r['source_edge_key']):r
            for r in ledger['edges'] if r['valid_edge'] and
            all(x['source_face_id']>=0 for x in r['source_incidents'])}


def run():
    dest=C.OUT/'analysis';done=dest/'COMPLETE.json'
    if done.exists():r=C.read(done);C.verify(r['files']);return r
    m=A.load_manifest(True);sr=C.read(C.OUT/'scores/COMPLETE.json');C.verify(sr['files'])
    assert sr['new_scores']==48
    C.write(dest/'attempt.json',dict(operation='Cached local-label and paired-stage audit',new_scoring_calls=0,model_calls=0))
    per_request=[];patchrows=[];slotrows=[];pairrows=[];cornerrows=[];edgerows=[]
    rejection=collections.Counter();funnel=collections.Counter();retention=[]
    for t in m['tasks']:
        rid=t['request_id'];k=t['K'];td=C.OUT/'targets'/rid
        rr=C.read(td/'receipt.json');p=C.arrays(td/'target.npz');raw=p['raw'];state=p['corner_state']
        det=C.read(td/'candidate_detection.json');funnel.update(det['funnel'])
        pp=C.read(td/'patches.json')['rows'];accepted=[x for x in pp if x['decision']=='ACCEPTED']
        for x in pp:rejection.update(x['reasons'])
        stages={}
        for name,meshpath,pairpath in [
            ('RAW',t['assets']['RAW']['path'],t['assets']['RAW_PAIRS']['path']),
            ('TARGET',td/'target.npz',td/'final_target_pairs.json'),
            ('IDENTITY_BASE',t['assets']['GEO_ONLY_BASE']['path'],t['assets']['GEO_ONLY_PAIRS']['path']),
            ('TARGET_BASE',C.OUT/'geometry'/rid/'post.npz',C.OUT/'geometry'/rid/'pairs.json'),
            ('IDENTITY_FINAL',C.OUT/'cpu/RAW_IDENTITY'/rid/'CLEAN_FINAL_POST/post.npz',C.OUT/'cpu/RAW_IDENTITY'/rid/'CLEAN_FINAL_POST/scores/pairs.json'),
            ('TARGET_FINAL',C.OUT/'cpu/LOCAL_REPAIR'/rid/'CLEAN_FINAL_POST/post.npz',C.OUT/'cpu/LOCAL_REPAIR'/rid/'CLEAN_FINAL_POST/scores/pairs.json')]:
            a=C.arrays(meshpath)
            if name=='TARGET':a=dict(a,output=a['target'])
            pairs=C.read(pairpath)
            stages[name]=dict(mesh=a,slots=slot_data(a),pairs=pairs,contacts=original_contacts(a,pairs,k))
        for name,s in stages.items():
            row=dict(request_id=rid,parent_id=t['parent_id'],noise_index=t['noise_index'],stage=name)
            for kind,v in s['contacts'].items():
                row[kind+'_illegal_original_pairs']=len(v['illegal']);row[kind+'_unknown_original_pairs']=len(v['unknown']);row[kind+'_bridge_illegal_pairs']=v['bridge_illegal']
            row['retained_free_original_slots']=sum(x>=k for x in s['slots'])
            retention.append(row)
        edge_losses={}
        for suffix in ('BASE','FINAL'):
            old,new=stages['IDENTITY_'+suffix],stages['TARGET_'+suffix]
            oe=legal_original_edges(old['mesh'],old['pairs'],k)
            ne=legal_original_edges(new['mesh'],new['pairs'],k)
            edge_losses[suffix]=len(oe.keys()-ne.keys())
            for edge in sorted(oe.keys()|ne.keys()):
                before,after=oe.get(edge),ne.get(edge)
                edgerows.append(dict(request_id=rid,parent_id=t['parent_id'],stage=suffix,
                    source_edge_key=json.dumps(edge),kind=(before or after)['kind'],
                    change='RETAINED_LEGAL' if before and after else ('LOST_LEGAL' if before else 'GAINED_LEGAL'),
                    baseline_length_model=before['length_model'] if before else None,
                    target_length_model=after['length_model'] if after else None))
            for kind in ('CF','FF'):
                oi,ni=old['contacts'][kind]['illegal'],new['contacts'][kind]['illegal']
                for pair in sorted(oi|ni):
                    both=all(x in new['slots'] for x in pair)
                    change='ILLEGAL_BOTH' if pair in oi and pair in ni else ('NEW_ILLEGAL' if pair in ni else ('RESOLVED_BOTH_FACES_RETAINED' if both else 'NO_LONGER_OBSERVABLE_FACE_REMOVED'))
                    pairrows.append(dict(request_id=rid,parent_id=t['parent_id'],stage=suffix,kind=kind,raw_slot_i=pair[0],raw_slot_j=pair[1],change=change))
        meaningful=0;surviving=0
        for patch in accepted:
            designated={tuple(x) for x in patch['designated_pairs']}
            pr=dict(request_id=rid,parent_id=t['parent_id'],patch_id=patch['patch_id'],designated_CF_pairs=len(designated))
            for name in ('TARGET','IDENTITY_BASE','TARGET_BASE','IDENTITY_FINAL','TARGET_FINAL'):
                s=stages[name];contact=s['contacts']['CF']
                missing={x for pair in designated for x in pair if x not in s['slots']}
                remaining=designated&(contact['illegal']|contact['unknown'])
                pr[name+'_missing_designated_slots']=len(missing);pr[name+'_illegal_or_unknown']=len(remaining)
            pr['survives_BASE_FINAL_without_designated_deletion']=all(pr[name+'_missing_designated_slots']==0 and pr[name+'_illegal_or_unknown']==0 for name in ('TARGET_BASE','TARGET_FINAL'))
            pr['additional_FINAL_repair_vs_identity']=bool(pr['survives_BASE_FINAL_without_designated_deletion'] and pr['IDENTITY_FINAL_illegal_or_unknown']>0)
            surviving+=int(pr['survives_BASE_FINAL_without_designated_deletion']);meaningful+=int(pr['additional_FINAL_repair_vs_identity']);patchrows.append(pr)
        keep_slots=np.flatnonzero((state==1).any(1));edit_slots=np.flatnonzero((state==2).any(1))
        keep_added_loss=0;keep_added_move=0
        for label,ids in [('KEEP',keep_slots),('EDIT',edit_slots)]:
            for sid in ids:
                row=dict(request_id=rid,parent_id=t['parent_id'],raw_slot=int(sid),state=label,RAW_area=area(raw[sid]))
                for name in ('IDENTITY_BASE','TARGET_BASE','IDENTITY_FINAL','TARGET_FINAL'):
                    x=stages[name]['slots'].get(int(sid));row[name+'_retained']=x is not None;row[name+'_area']=area(x) if x is not None else None
                    row[name+'_max_corner_displacement_from_RAW']=float(np.linalg.norm(x.astype(float)-raw[sid].astype(float),axis=1).max()) if x is not None else None
                before=stages['IDENTITY_FINAL']['slots'].get(int(sid));after=stages['TARGET_FINAL']['slots'].get(int(sid))
                row['target_induced_FINAL_loss']=before is not None and after is None
                row['target_induced_FINAL_coordinate_change']=None if before is None or after is None else float(np.linalg.norm(after.astype(float)-before.astype(float),axis=1).max())
                if label=='KEEP':
                    keep_added_loss+=int(row['target_induced_FINAL_loss']);keep_added_move+=int(row['target_induced_FINAL_coordinate_change'] not in (None,0.))
                slotrows.append(row)
        final_added={kind:len(stages['TARGET_FINAL']['contacts'][kind]['illegal']-stages['IDENTITY_FINAL']['contacts'][kind]['illegal']) for kind in ('CF','FF')}
        changed=np.linalg.norm(p['target'].astype(float)-raw.astype(float),axis=-1)
        target_bridge_illegal=sum(stages['TARGET_FINAL']['contacts'][kind]['bridge_illegal'] for kind in ('CF','FF'))
        target_unknown=sum(len(stages['TARGET_FINAL']['contacts'][kind]['unknown']-stages['IDENTITY_FINAL']['contacts'][kind]['unknown']) for kind in ('CF','FF'))
        allfinalnew=sum(final_added.values())
        construction=dict(request_id=rid,parent_id=t['parent_id'],noise_index=t['noise_index'],eval_group=t['eval_group'],
            seeds=rr['seeds'],patches=rr['patches'],accepted_patches=rr['accepted_patches'],rejected_patches=rr['rejected_patches'],
            KEEP_corners=int((state==1).sum()),EDIT_corners=int((state==2).sum()),UNKNOWN_corners=int((state==0).sum()),
            max_edit_displacement_L=float(changed.max()/C.L),post_surviving_patches=surviving,
            FINAL_additional_repair_patches=meaningful,FINAL_new_CF=final_added['CF'],FINAL_new_FF=final_added['FF'],
            FINAL_target_induced_KEEP_slot_loss=keep_added_loss,FINAL_target_induced_KEEP_slot_coordinate_change=keep_added_move,
            FINAL_target_bridge_illegal_pairs=target_bridge_illegal,FINAL_new_original_unknown_pairs=target_unknown,
            BASE_lost_original_legal_edges=edge_losses['BASE'],FINAL_lost_original_legal_edges=edge_losses['FINAL'],
            numerical_local_pair_ready=bool(surviving>0 and allfinalnew==0 and target_unknown==0 and target_bridge_illegal==0 and keep_added_loss==0 and keep_added_move==0 and not any(edge_losses.values())),
            actual_visual_review='PENDING_SEPARATE_REVIEW',N=t['N'],K=k)
        per_request.append(construction)
        for i,j in np.argwhere(state==2):
            cornerrows.append(dict(request_id=rid,raw_slot=int(i),raw_corner=int(j),displacement_model=float(changed[i,j]),displacement_L=float(changed[i,j]/C.L)))
    for name,rs,fields in [('per_request',per_request,None),('patch_post_survival',patchrows,['request_id','parent_id','patch_id','designated_CF_pairs']),
        ('KEEP_EDIT_slot_survival',slotrows,['request_id','parent_id','raw_slot','state']),('paired_illegal_original_faces',pairrows,None),
        ('paired_original_legal_edges',edgerows,None),
        ('per_stage_illegal_pairs',retention,None),('edited_corners',cornerrows,['request_id','raw_slot','raw_corner','displacement_model','displacement_L'])]:
        C.csvwrite(dest/(name+'.csv'),rs,fields=fields if not rs else None)
    C.csvwrite(dest/'candidate_detection_funnel.csv',[dict(reason=k,count=v) for k,v in sorted(funnel.items())])
    C.csvwrite(dest/'patch_rejection_reasons.csv',[dict(reason=k,count=v) for k,v in sorted(rejection.items())],fields=['reason','count'])
    # Preserve complete original scorers' values on exactly these selected requests.
    selected={t['request_id'] for t in m['tasks']}
    previous=rows(C.PREV/'scores/per_output.csv');refs=[r for r in previous if r['request_id'] in selected]
    C.csvwrite(dest/'read_only_reference_all_metrics.csv',refs)
    result=dict(status='COMPLETE',requests=12,parents=6,requests_with_EDIT=sum(r['EDIT_corners']>0 for r in per_request),
        requests_with_numerically_surviving_EDIT=sum(r['numerical_local_pair_ready'] for r in per_request),
        requests_with_additional_FINAL_repair=sum(r['FINAL_additional_repair_patches']>0 for r in per_request),
        accepted_patches=sum(r['accepted_patches'] for r in per_request),rejected_patches=sum(r['rejected_patches'] for r in per_request),
        actual_EDIT_corners=sum(r['EDIT_corners'] for r in per_request),actual_KEEP_corners=sum(r['KEEP_corners'] for r in per_request),
        candidate_detection_funnel=dict(funnel),rejection_reasons=dict(rejection),
        readiness_gate_required_requests=7,visual_review='PENDING_SEPARATE_REVIEW',new_scoring_calls=0,model_calls=0,
        reference_rows=len(refs),reference_sha256=C.sha(C.PREV/'scores/per_output.csv'),
        files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file()})
    C.write(done,result);print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False));return result

if __name__=='__main__':run()
