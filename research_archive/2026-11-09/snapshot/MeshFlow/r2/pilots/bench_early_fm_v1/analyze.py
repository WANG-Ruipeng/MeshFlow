"""Original all-metric tables, paired deltas, saved-state checks and actual review."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from .common import *
import numpy as np,collections,csv,json
STAGES=('RAW','BASE_POST','CLEAN_FINAL_POST')
LABELS=('COMMON','USER','EARLY')

def scorepath(label,q,stage):
    base=OUT if label=='EARLY' else OLD;rid=q['request_id']
    return base/'evaluation/scores'/label/rid/'RAW' if stage=='RAW' else base/'cpu'/label/rid/stage/'scores'

def readcsv(p):
    with resolve(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    reg=registration();qs=reg['requests'];rows=[];checks=[];survival=[];trajectory=[]
    for q in qs:
        rid=q['request_id'];layout=arrays(q['layout_npz']);fixed=layout['anchor_ids']>=0;states={}
        for label in LABELS:
            base=OUT if label=='EARLY' else OLD;rawdir=base/'evaluation'/label/rid/'raw'
            tr=arrays(rawdir/'trajectory.npz');states[label]=tr['states']
            raw=arrays(rawdir/'raw.npz');post=arrays(base/'evaluation'/label/rid/'post/post.npz')
            assert tr['states'].shape==(51,q['N'],3,3)
            assert np.array_equal(tr['states'][0],arrays(q['baseline']['USER']['trajectory'])['states'][0])
            assert np.array_equal(tr['states'][-1],raw['raw'])
            assert np.all(tr['states'][:,fixed]==tr['states'][0,fixed])
            for i in range(50):
                nxt=tr['states'][i]+np.float32(.02)*tr['velocities'][i];nxt[fixed]=tr['states'][0,fixed]
                assert np.array_equal(nxt,tr['states'][i+1])
            for stage in STAGES:
                folder=scorepath(label,q,stage);r=read(folder/'row.json');s=read(folder/'score.json')
                assert s['C_preserved_exact'] and s['C_original_order_preserved']
                r.update(label=label,part=q['part_type'],split=q['split'],stage=stage,
                    result_origin='NEW' if label=='EARLY' else 'CACHED_UNCHANGED',
                    score_sha256=sha(folder/'row.json'),score_path=str(folder),
                    C_preserved_exact=s['C_preserved_exact'],C_original_order_preserved=s['C_original_order_preserved'],
                    query_sha256=sha(q['query_descriptor']['path']),C_query_count=s['C_query_count'],
                    C_query_weight_denominator=s['C_query_weight_denominator'],CF_tested_pair_denominator=s['CF_tested_pair_denominator'],
                    FF_tested_pair_denominator=s['FF_tested_pair_denominator'],S_status=s['status_by_metric']['S'],
                    E_status=s['status_by_metric']['E'],GT_status=s['status_by_metric']['GT'])
                rows.append(r)
            checks.append(dict(label=label,request_id=rid,initial_state_exact=True,all_anchors_exact=True,
                all50_euler_steps_exact=True,raw_endpoint_exact=True,N=q['N'],K=q['K']))
            proposal=arrays(base/'evaluation'/label/rid/'post/proposal.npz')['official_unconstrained_proposal']
            pre=arrays(base/'evaluation'/label/rid/'post/pre_geo.npz')['output']
            records=read(base/'evaluation'/label/rid/'post/geometry.json')['face_records']
            def area(x):return np.linalg.norm(np.cross(x[:,1]-x[:,0],x[:,2]-x[:,0]),axis=1)/2
            areas={name:area(v) for name,v in [('RAW',raw['raw']),('D0',proposal),('PRE_GEO',pre)]}
            for record in records:
                if record['known']:continue
                i=record['source_face_id'];survival.append(dict(label=label,request_id=rid,parent_id=q['parent_id'],split=q['split'],part=q['part_type'],
                    **record,role='SEAM' if raw['raw_seam_slot_mask'][i] else 'BULK',**{k+'_area':float(v[i]) for k,v in areas.items()}))
        # Descriptive same-noise slot distance; not geometry quality or GT.
        for step in range(51):
            d=states['EARLY'][step].astype(float)-states['USER'][step]
            trajectory.append(dict(request_id=rid,split=q['split'],part=q['part_type'],t=step/50,
                moving_corner_RMS_vector_model=float(np.sqrt(np.mean(np.sum(d[~fixed]**2,axis=-1)))),
                status='DESCRIPTIVE_SAME_SLOT_NOT_QUALITY'))
    assert len(rows)==144
    csvwrite(OUT/'analysis/per_output_all_metrics.csv',rows);csvwrite(OUT/'analysis/trajectory_integrity.csv',checks)
    csvwrite(OUT/'analysis/GEO_face_survival.csv',survival);csvwrite(OUT/'analysis/paired_trajectory_distance.csv',trajectory)
    numeric=[k for k in dict.fromkeys(k for r in rows for k in r) if k not in ('N','K','noise_index') and any(isinstance(r.get(k),(int,float)) and not isinstance(r.get(k),bool) for r in rows)]
    def nums(rr,k):return [float(r[k]) for r in rr if isinstance(r.get(k),(int,float)) and not isinstance(r.get(k),bool) and np.isfinite(r[k])]
    agg=[];parents=[];paired=[];compact=[]
    for split in ('CURRENT_TRAIN','DEV'):
        for stage in STAGES:
            for label in LABELS:
                rr=[r for r in rows if (r['split'],r['stage'],r['label'])==(split,stage,label)]
                for k in numeric:
                    v=nums(rr,k);agg.append(dict(split=split,stage=stage,label=label,metric=k,requests=len(rr),numeric_count=len(v),NA=len(rr)-len(v),
                        mean=float(np.mean(v)) if v else None,median=float(np.median(v)) if v else None,minimum=min(v) if v else None,maximum=max(v) if v else None))
                ci=[r for r in rr if r['ConflictCov_C_status']=='PASS']
                compact.append(dict(split=split,stage=stage,label=label,n=len(rr),
                    CF_pct=100*np.mean(nums(rr,'CF_illegal_affected_free_fraction')),FF_pct=100*np.mean(nums(rr,'FF_illegal_affected_free_fraction')),
                    I_all_pct=100*np.mean(nums(rr,'ConflictCov_C')),I_PASS_pct=100*np.mean(nums(ci,'ConflictCov_C')) if ci else None,I_PASS_n=len(ci),
                    small_angle_pct=100*np.mean(nums(rr,'free_minimum_angle_lt5_fraction')),
                    illegal_FF_pairs=float(np.mean(nums(rr,'FF_illegal_pair_count'))),
                    free_faces=float(np.mean(nums(rr,'actual_free_faces'))),deleted_free_faces=float(np.mean(nums(rr,'deleted_free_faces'))),
                    free_area=float(np.mean(nums(rr,'free_total_surface_area'))),
                    legal_CF_edges=float(np.mean(nums(rr,'C_F_valid_edge_count'))),
                    bulk_reach_pct=100*np.mean(nums(rr,'free_only_reached_bulk_area_fraction')),
                    bulk_ge30=sum(r.get('bulk_reach_ge_0_30_descriptive')==1 for r in rr)))
    for q in qs[::2]:
        for label in LABELS:
            for stage in STAGES:
                rr=[r for r in rows if (r['parent_id'],r['label'],r['stage'])==(q['parent_id'],label,stage)]
                rec=dict(parent_id=q['parent_id'],part=q['part_type'],split=q['split'],label=label,stage=stage,noises=len(rr))
                for k in numeric:
                    v=nums(rr,k);rec[k]=float(np.mean(v)) if v else None
                parents.append(rec)
    for q in qs:
        for stage in STAGES:
            rr={r['label']:r for r in rows if r['request_id']==q['request_id'] and r['stage']==stage}
            for k in numeric:
                v={label:rr[label].get(k) for label in LABELS};ok=all(isinstance(x,(int,float)) and np.isfinite(x) for x in v.values())
                paired.append(dict(request_id=q['request_id'],parent_id=q['parent_id'],noise_index=q['noise_index'],split=q['split'],part=q['part_type'],
                    stage=stage,metric=k,status='NUMERIC_PAIRED' if ok else 'NA',**v,
                    EARLY_minus_USER=v['EARLY']-v['USER'] if ok else None,USER_minus_COMMON=v['USER']-v['COMMON'] if ok else None))
    csvwrite(OUT/'analysis/all_metric_summary.csv',agg);csvwrite(OUT/'analysis/per_parent_metrics.csv',parents)
    csvwrite(OUT/'analysis/per_request_paired.csv',paired);csvwrite(OUT/'analysis/key_metrics.csv',compact)
    # Notes are entered only after actual image inspection; no automatic proxy labeling.
    note_path=OUT/'visual_review_notes.json'
    if note_path.exists():
        notes=read(note_path)['rows'];assert len(notes)==16
        old={(r['request_id'],r['stage']):r for r in readcsv(OLD/'visual_review/main.csv') if r['label']=='USER'}
        review=[];adoption=[]
        for q in qs:
            n=next(r for r in notes if q['parent_id'].startswith(r['parent_prefix']) and q['noise_index']==r['noise'])
            page=OUT/'gallery'/(q['request_id']+'.png')
            for label in ('USER','EARLY'):
                for stage in ('RAW','CLEAN_FINAL_POST'):
                    key=label+('_RAW' if stage=='RAW' else '_FINAL');labelvalue=n[key]
                    assert labelvalue in ('YES','NO','PARTIAL','UNCERTAIN')
                    review.append(dict(request_id=q['request_id'],parent_id=q['parent_id'],part=q['part_type'],split=q['split'],noise_index=q['noise_index'],
                        label=label,stage=stage,reviewer='AI_PROXY',blind=False,C_adoption=labelvalue,
                        body_completeness=n.get(key+'_body',n.get('body','BROADLY_PRESENT_WITH_LOCAL_DEFECTS')),observation=n[key+'_note'],
                        confidence=n.get(key+'_confidence','MEDIUM'),image='gallery/'+page.name,image_sha256=sha(page),
                        original_USER_label=old.get((q['request_id'],stage),{}).get('C_adoption') if label=='USER' else '',
                        exact_connectivity='NOT_CERTIFIED_BY_IMAGE'))
            adoption.append(dict(request_id=q['request_id'],parent_id=q['parent_id'],split=q['split'],part=q['part_type'],noise_index=q['noise_index'],
                **{k:n[k] for k in ('USER_RAW','EARLY_RAW','USER_FINAL','EARLY_FINAL')}))
        csvwrite(OUT/'analysis/visual_review.csv',review);csvwrite(OUT/'analysis/adoption_paired.csv',adoption)
        for r in compact:
            vv=[v for v in review if v['label']==r['label'] and v['stage']==r['stage'] and v['split']==r['split']]
            r['reviewed']=len(vv);r['adoption_YES']=sum(v['C_adoption']=='YES' for v in vv) if vv else None
            r['strict_joint_descriptive']=sum(v['C_adoption']=='YES' and next(x for x in rows if x['label']==v['label'] and x['request_id']==v['request_id'] and x['stage']==v['stage'])['bulk_reach_ge_0_30_descriptive']==1 for v in vv) if vv else None
    save(OUT/'analysis'/('summary.json' if note_path.exists() else 'numeric_summary.json'),dict(status='ANALYZED' if note_path.exists() else 'AWAITING_REAL_VISUAL_REVIEW',
        scored_rows=len(rows),new_score_rows=48,original_numeric_fields=len(numeric),key_metrics=compact))
    print(json.dumps(compact,ensure_ascii=False))

if __name__=='__main__':main()
