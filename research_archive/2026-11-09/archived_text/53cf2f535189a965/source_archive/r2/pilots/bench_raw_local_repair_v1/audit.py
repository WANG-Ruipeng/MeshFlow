"""Independent read-only target/result audit; no detector, geometry, or model.

Reconstructs accepted edits and cached contact transitions from original RAW
identities. Only this experiment's audit JSONs are written.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, collections, csv, json, math, sys, time, traceback
from pathlib import Path
import numpy as np
from . import common as C

COUNTS=collections.Counter()
HASHES={}


def check(ok,name):
    if not bool(ok):raise AssertionError(name)
    COUNTS[name]+=1


def verify(files):
    for path,digest in files.items():
        path=str(C.resolve(path))
        actual=HASHES.get(path)
        if actual is None:actual=C.sha(path);HASHES[path]=actual
        check(actual==digest,'protected_hash')


def load_receipt(path):
    r=C.read(path);verify(r.get('files',{}));verify(r.get('source_hashes',{}));return r


def cached_pairs(path,k,slots):
    scans=C.read(path);out={kind:dict(illegal=set(),unknown=set()) for kind in ('CF','FF')}
    for kind,scan in scans.items():
        expected=k*(len(slots)-k) if kind=='CF' else (len(slots)-k)*(len(slots)-k-1)//2
        check(scan['tested_pairs']==expected and scan['expected_full_pairs']==expected,'full_cached_pair_domain')
        check(scan.get('affected_subset') is None,'full_cached_pair_scan')
        for r in scan['contacts']:
            i,j=map(int,r['pair']);i=i if kind=='CF' else i+k;j+=k
            p=(int(slots[i]),int(slots[j]));p=p if kind=='CF' else tuple(sorted(p))
            if r.get('detector',{}).get('illegal',False):out[kind]['illegal'].add(p)
            cls=r.get('classification','').lower()
            if any(s in cls for s in ('unknown','unresolved','degenerate')) or r.get('detector',{}).get('resolved') is False:
                out[kind]['unknown'].add(p)
    return out


def targets():
    m=C.read(C.OUT/'data/manifest.json');verify(m['protected_assets']);verify(m['derived_files'])
    end=load_receipt(C.OUT/'targets/COMPLETE.json');strategy=C.read(C.OUT/'targets/strategy.json')
    check(end['requests']==12 and end['status']=='COMPLETE','targets_complete12')
    amendment=C.read(C.OUT/'targets/engineering_amendment.json')
    check(amendment['pair_scan_calls_before_failure']==0 and amendment['connectivity_ledger_calls_before_failure']==0
        and amendment['replayed_scientific_calls']==0 and not amendment['scientific_rule_changed'],'zero_work_boundary_recovery')
    check(C.sha(C.OUT/'targets/code_before_index_fix.py')==amendment['old_code_sha256'],'failed_source_preserved')
    check(C.sha(C.OUT/'targets/failure.json')==amendment['old_failure_sha256'],'failure_record_preserved')
    results=[];unmappable=[];total_patches=total_accepted=total_rejected=scan_calls=0
    for t in m['tasks']:
        rid=t['request_id'];folder=C.OUT/'targets'/rid;receipt=load_receipt(folder/'receipt.json')
        a=C.arrays(folder/'target.npz');original=C.arrays(t['assets']['RAW']['path']);r=original['raw']
        x=a['target'];s=a['corner_state'];k,n=int(t['K']),int(t['N']);ident=np.tile(np.arange(3),(n,1))
        check(x.dtype==np.float32 and x.shape==(n,3,3) and np.isfinite(x).all(),'target_N33_float32')
        check(a['raw'].tobytes()==r.tobytes(),'target_RAW_identity')
        check(np.array_equal(a['raw_slot_ids'],np.arange(n)),'RAW_slot_identity')
        check(np.array_equal(a['source_corner_ids'],ident),'canonical_target_corner_identity')
        check(np.array_equal(a['input_export_source_corner_ids'],original['source_corner_ids']),'original_export_corner_map_preserved')
        inv=np.argsort(original['source_corner_ids'],axis=1)
        faces=np.take_along_axis(original['faces'],inv,axis=1)
        check(original['vertices'][faces].tobytes()==r.tobytes(),'canonical_index_inverse_roundtrip')
        check(original['vertices'][original['faces']].tobytes()==original['output'].tobytes(),'original_export_index_roundtrip')
        check(s.shape==(n,3) and set(np.unique(s))<={0,1,2,3},'state_domain')
        check(np.all(s[:k]==3) and not np.any(s[k:]==3),'C_state_rows')
        check(np.array_equal(a['loss_mask'],(s==1)|(s==2)) and not a['loss_mask'][:k].any(),'KEEP_EDIT_only_loss_mask')
        check(x[:k].tobytes()==r[:k].tobytes(),'C_exact')
        check(x[(s==0)|(s==1)].tobytes()==r[(s==0)|(s==1)].tobytes(),'UNKNOWN_KEEP_exact_RAW')
        delta=np.linalg.norm(x.astype(float)-r.astype(float),axis=-1)
        check(np.all(delta[s==2]>0) and delta.max()<=strategy['maximum_member_displacement_L']*C.L+1e-7,'EDIT_nonzero_bounded')
        check(np.array_equal(delta>0,s==2),'all_changed_corners_labeled_EDIT')
        if np.any(s==1):check(t['parent_id'].startswith(('a8','27')),'KEEP_inspected_support_parent')
        maps=C.arrays(t['slot_corner_maps']['path']);cv=maps['GEO_ONLY_BASE__raw_corner_to_merged_vertex']
        geo=C.arrays(t['assets']['GEO_ONLY_BASE']['path']);gids=geo['source_face_ids']
        patches=C.read(folder/'patches.json')['rows'];current=r.copy()
        rawpairs=cached_pairs(t['assets']['RAW_PAIRS']['path'],k,np.arange(n))
        gpairs=cached_pairs(t['assets']['GEO_ONLY_PAIRS']['path'],k,gids)
        accepted=0;rejected=0;max_member=0.
        for patch in patches:
            prop=C.arrays(patch['proposal_path']);trial=current.copy();affected=set()
            negative_clusters=[]
            for cluster in patch['clusters']:
                cid=int(cluster['cluster_id']);members=np.flatnonzero(cv.reshape(-1)==cid)
                if cid<0:negative_clusters.append(cid)
                free=members[members>=3*k];fixed=members[members<3*k]
                check(np.array_equal(free,np.asarray(cluster['free_flat_corners'])),'real_GEO_free_cluster_members')
                check(np.array_equal(fixed,np.asarray(cluster['fixed_C_members'])),'real_GEO_C_cluster_members')
                q=np.asarray(cluster['target'],np.float32);trial.reshape(-1,3)[free]=q
                affected.update(int(i//3) for i in free)
                d=np.linalg.norm(r.reshape(-1,3)[free].astype(float)-q,axis=1)
                if patch['decision']=='ACCEPTED':
                    check(cid>=0,'accepted_cluster_has_real_vertex')
                    check(len(free)>0 and d.max()<=.005*C.L+1e-7,'accepted_all_cluster_members_bounded')
                    max_member=max(max_member,float(d.max()))
                    if len(fixed):check(np.all(r.reshape(-1,3)[fixed]==q),'accepted_C_cluster_point_exact')
            check(prop['raw'].tobytes()==r.tobytes() and prop['output'].tobytes()==trial.tobytes(),'proposal_reconstructed_from_original_clusters')
            moved=np.flatnonzero(np.any(trial.reshape(-1,3)!=current.reshape(-1,3),axis=1))
            check(np.array_equal(moved,np.asarray(patch['affected_flat_corners'])),'proposal_changed_corner_ledger')
            check(affected==set(patch['affected_slots']),'all_cluster_affected_faces_recorded')
            if patch['decision']=='REJECTED':
                check(len(patch['reasons'])>0,'rejection_reason_retained')
                if negative_clusters:
                    check(not(folder/'proposals'/(patch['patch_id']+'_RAW_pairs.json')).exists()
                        and not(folder/'proposals'/(patch['patch_id']+'_fixed_GEO_pairs.json')).exists(),
                        'unmappable_cluster_rejected_before_detector')
                    unmappable.append(dict(request_id=rid,patch_id=patch['patch_id'],cluster_ids=negative_clusters,
                        rejected_reasons=patch['reasons'],classification='UNMAPPABLE_NOT_A_REAL_GEO_CLUSTER',
                        actual_impact='Failed proposal retained; no detector/frozen decode/target edit for this patch'))
                rejected+=1;continue
            check(patch['decision']=='ACCEPTED' and not patch['reasons'],'acceptance_has_no_rejection')
            check(not np.any(s.reshape(-1)[moved]==1),'accepted_does_not_move_KEEP')
            p=folder/'proposals'/patch['patch_id']
            newer=cached_pairs(str(p)+'_RAW_pairs.json',k,np.arange(n))
            newgeo=cached_pairs(str(p)+'_fixed_GEO_pairs.json',k,gids)
            for old,new in ((rawpairs,newer),(gpairs,newgeo)):
                for kind in ('CF','FF'):
                    for state in ('illegal','unknown'):
                        check(not(new[kind][state]-old[kind][state]),'accepted_no_new_cached_'+state+'_'+kind)
            designated={tuple(q) for q in patch['designated_pairs']}
            check(bool(designated) and not(designated&(newer['CF']['illegal']|newer['CF']['unknown'])),'designated_CF_resolved')
            check(patch['checks']['RAW_old_legal_edges_lost']==0 and patch['checks']['fixed_GEO_old_legal_edges_lost']==0,'cached_original_legal_edge_guard')
            tx=trial.astype(float);rx=r.astype(float)
            aa=np.linalg.norm(np.cross(tx[:,1]-tx[:,0],tx[:,2]-tx[:,0]),axis=1)
            base=np.linalg.norm(np.cross(rx[:,1]-rx[:,0],rx[:,2]-rx[:,0]),axis=1)
            idx=np.asarray(sorted(affected));ratios=aa[idx]/base[idx]
            check(np.all(ratios>=.8-1e-6) and np.all(ratios<=1.25+1e-6),'accepted_area_ratio_independent')
            current=trial;rawpairs=newer;gpairs=newgeo;accepted+=1
        check(current.tobytes()==x.tobytes(),'final_target_equals_accepted_sequence_only')
        check(rawpairs==cached_pairs(folder/'final_target_pairs.json',k,np.arange(n)),'final_RAW_contact_cache_consistent')
        check(gpairs==cached_pairs(folder/'final_fixed_GEO_pairs.json',k,gids),'final_fixed_GEO_contact_cache_consistent')
        check(accepted==receipt['accepted_patches'] and rejected==receipt['rejected_patches'],'patch_counts_exact')
        for name,value in (('EDIT_corners',int((s==2).sum())),('KEEP_corners',int((s==1).sum())),('UNKNOWN_corners',int((s==0).sum()))):
            check(receipt[name]==value,'receipt_'+name)
        results.append(dict(request_id=rid,parent_id=t['parent_id'],accepted_patches=accepted,rejected_patches=rejected,
            EDIT_corners=int((s==2).sum()),KEEP_corners=int((s==1).sum()),maximum_displacement_L=float(delta.max()/C.L),
            maximum_accepted_cluster_member_displacement_L=max_member/C.L))
        total_patches+=len(patches);total_accepted+=accepted;total_rejected+=rejected
        scan_calls+=receipt['detector_calls']['pair_scan_calls']
    check(total_accepted==end['accepted_patches'] and total_rejected==end['rejected_patches'],'aggregate_patch_counts')
    check(sum(r['EDIT_corners']>0 for r in results)==end['requests_with_actual_EDIT'],'aggregate_EDIT_requests')
    check(scan_calls==end['detector_pair_scan_calls'],'recorded_detector_call_count')
    return dict(requests=12,parents=6,rows=results,accepted_patches=total_accepted,rejected_patches=total_rejected,
        requests_with_EDIT=sum(r['EDIT_corners']>0 for r in results),recorded_construction_pair_scan_calls=scan_calls,
        recorded_engineering_failures=1,scientific_replay_calls=0,
        unmappable_cluster_failed_proposals=unmappable,
        caveat='This audit reads cached full detectors and source ledgers; it does not rescore geometry or certify a complete clean mesh.')


def csvrows(path):
    with C.resolve(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def numeric(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)


def csv_equal(text,value):
    if value is None:return text==''
    if numeric(value):return math.isclose(float(text),float(value),rel_tol=2e-12,abs_tol=1e-12)
    return text==str(value)


def independent_aggregate(rows,keys,fields):
    groups=collections.defaultdict(list)
    for row in rows:groups[tuple(row[k] for k in keys)].append(row)
    out=[]
    for key,group in sorted(groups.items()):
        row=dict(zip(keys,key));row['rows']=len(group)
        for f in fields:
            values=[float(r[f]) for r in group if numeric(r.get(f))]
            row[f]=math.fsum(values)/len(values) if values else None;row[f+'__valid_n']=len(values)
        out.append(row)
    return out


def results():
    prior=load_receipt(C.OUT/'audits/independent_target_audit.json')
    check(prior['status']=='PASS','target_audit_passed')
    m=C.read(C.OUT/'data/manifest.json');r=load_receipt(C.OUT/'scores/COMPLETE.json');verify(r['protected'])
    check(r['new_scores']==48 and r['requests']==12 and r['GEO_calls']==12 and r['cascade_calls']==24,'actual_operation_budget')
    for k in ('model_calls','D0_calls','GPU_calls','new_noise','new_queries'):check(r[k]==0,'zero_'+k)
    fields=r['metric_fields'];check(len(fields)==177,'original177fields')
    rows=C.read(C.OUT/'scores/metric_rows.json');check(len(rows)==48,'score_rows48')
    by={(x['label'],x['stage'],x['request_id']):x for x in rows};check(len(by)==48,'unique_stage_identity')
    for t in m['tasks']:
        rid=t['request_id'];k,n=t['K'],t['N']
        g=load_receipt(C.OUT/'geometry'/rid/'complete.json');check(g['GEO_calls']==1,'one_target_GEO')
        post=C.arrays(C.OUT/'geometry'/rid/'post.npz');sf=post['source_face_ids'];sc=post['source_corner_ids']
        check(np.array_equal(post['faces'],post['corner_vertex_ids'][sf[:,None],sc]),'POST_corner_map_roundtrip')
        check(post['vertices'][post['faces']].tobytes()==post['output'].tobytes(),'POST_index_roundtrip')
        check(len(np.unique(sf))==len(sf) and sf.min()>=0 and sf.max()<n,'POST_RAW_slot_range')
        check(np.array_equal(sf[:k],np.arange(k)),'POST_C_face_identity')
        source=C.arrays(t['assets']['SOURCE_C']['path'])
        check(post['C'].tobytes()==source['C_original'].tobytes(),'POST_source_C_exact')
        for label in ('RAW_IDENTITY','LOCAL_REPAIR'):
            folder=C.OUT/'cpu'/label/rid;load_receipt(folder/'complete.json')
            stored=C.read(folder/'metric_rows.json')
            for original in stored:
                row=by[(label,original['stage'],rid)]
                for f in fields:check(row[f]==original[f],'original_score_value_preserved')
        old=C.read(C.resolve(t['assets']['GEO_ONLY_SCORE']['path']).parent.parent/'complete.json')['row']
        for f in fields:check(by[('RAW_IDENTITY','BASE_POST',rid)][f]==old[f],'identity_BASE_old_GEO_exact')
    for row,c in zip(rows,csvrows(C.OUT/'scores/per_output.csv')):
        check(row['request_id']==c['request_id'] and row['label']==c['label'] and row['stage']==c['stage'],'CSV_row_identity')
        for f in fields:check(csv_equal(c[f],row[f]),'CSV_full_field_preserved')
    for name,keys in [('per_condition',('label','stage','eval_group','parent_id','condition_id')),
            ('per_parent',('label','stage','eval_group','parent_id')),('parent_macro_summary',('label','stage','eval_group'))]:
        rows=independent_aggregate(rows,keys,fields);actual=csvrows(C.OUT/'scores'/(name+'.csv'))
        check(len(rows)==len(actual),'aggregate_row_count')
        for expected,got in zip(rows,actual):
            for key,value in expected.items():check(csv_equal(got[key],value),'independent_aggregate_and_denominator')
    pairs=csvrows(C.OUT/'scores/paired_per_output.csv');check(len(pairs)==24,'paired_rows24')
    for p in pairs:
        a=by[('RAW_IDENTITY',p['stage'],p['request_id'])];b=by[('LOCAL_REPAIR',p['stage'],p['request_id'])]
        for f in fields:
            delta=b[f]-a[f] if numeric(a[f]) and numeric(b[f]) else None
            check(csv_equal(p[f+'__reference'],a[f]) and csv_equal(p[f+'__candidate'],b[f]) and csv_equal(p[f+'__delta'],delta),'same_chain_paired_field')
    analysis=load_receipt(C.OUT/'analysis/COMPLETE.json')
    ar=csvrows(C.OUT/'analysis/per_request.csv')
    check(len(ar)==12,'analysis_rows12')
    check(sum(int(x['EDIT_corners'])>0 for x in ar)==prior['requests_with_EDIT'],'analysis_EDIT_request_count')
    check(sum(int(x['accepted_patches']) for x in ar)==prior['accepted_patches'],'analysis_accepted_count')
    check(sum(int(x['rejected_patches']) for x in ar)==prior['rejected_patches'],'analysis_rejected_count')
    check(sum(x['numerical_local_pair_ready']=='True' for x in ar)==analysis['requests_with_numerically_surviving_EDIT'],'analysis_readiness_count')
    return dict(requests=12,parents=6,score_rows=48,paired_rows=24,fields=177,
        original_unchanged_BASE_values=12*177,actual_GEO_calls=12,actual_cascade_calls=24,
        recorded_engineering_failures=1,scientific_replay_calls=0,
        numerical_ready_requests=analysis['requests_with_numerically_surviving_EDIT'],
        additional_FINAL_repair_requests=analysis['requests_with_additional_FINAL_repair'],
        gate_required_requests=7,actual_visual_review='Separate actual reviewer records; not inferred from scores')


def run(stage):
    path=C.OUT/'audits'/('independent_target_audit.json' if stage=='targets' else 'independent_result_audit.json')
    if path.exists():r=load_receipt(path);check(r['status']=='PASS','completed_audit_pass');return r
    tick=time.perf_counter()
    try:
        details=targets() if stage=='targets' else results()
        check('torch' not in sys.modules,'no_neural_import')
        r=dict(status='PASS',stage=stage,**details,checks=dict(COUNTS),verified_unique_files=len(HASHES),
            seconds=time.perf_counter()-tick,model_calls=0,detector_calls=0,scoring_calls=0,geometry_calls=0,
            source_hashes={str(Path(__file__)):C.sha(__file__)},files=HASHES)
        C.write(path,r);return r
    except BaseException as exc:
        C.write(C.OUT/'audits'/('independent_'+stage+'_failure_'+str(time.time_ns())+'.json'),
            dict(status='FAIL',error=repr(exc),traceback=traceback.format_exc(),checks=dict(COUNTS),
                model_calls=0,detector_calls=0,scoring_calls=0,geometry_calls=0));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('targets','results'),required=True);a=p.parse_args()
    r=run(a.stage);print(json.dumps({k:v for k,v in r.items() if k not in ('files','source_hashes','rows')},ensure_ascii=False))
