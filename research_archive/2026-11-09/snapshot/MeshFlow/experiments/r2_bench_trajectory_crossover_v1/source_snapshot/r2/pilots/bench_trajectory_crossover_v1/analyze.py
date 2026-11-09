"""CPU aggregation of all four combinations; preserve every original field."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from .common import *
import numpy as np, itertools, collections, statistics
from r2.pilots.seam_abc_v1.eval_common import L

def readcsv(p):
    with resolve(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    reg=registration();qs=reg['requests'];allrows=[];review=[];integrity=[];distances=[];paths=[]
    notes=read(OUT/'visual_review_notes.json')
    oldreview={(r['request_id'],r['label']):r for r in readcsv(OLD/'visual_review/main.csv') if r['stage']=='RAW'}
    callrows=[]
    for combo in reg['new_combinations']:
        complete=read(OUT/'execution'/combo/'complete.json');verify(complete['files'])
        events=[json.loads(s) for s in (OUT/'execution'/combo/'calls.jsonl').read_text().splitlines()]
        assert sum(e['stage']=='STARTED' for e in events)==400
        assert sum(e['stage']=='RETURNED' for e in events)==400
        for q in qs:
            ee=[e for e in events if e['request_id']==q['request_id']]
            assert [e['absolute_step'] for e in ee if e['stage']=='RETURNED']==list(range(25,50))
        callrows.append(dict(combo=combo,started=400,returned=400,first_start=min(e['time'] for e in events),last_return=max(e['time'] for e in events),**{k:complete[k] for k in ('seconds','peak_allocated_bytes','peak_reserved_bytes')}))
    for q in qs:
        rid=q['request_id'];lay=arrays(q['layout_npz']);fixed=lay['anchor_ids']>=0
        nn=next(r for r in notes['rows'] if q['parent_id'].startswith(r['parent_prefix']) and q['noise_index']==r['noise'])
        page=OUT/'gallery'/(rid+'.png');pagehash=sha(page);ends={};trajectory={}
        for combo in COMBOS:
            mesh=arrays(raw_path(combo,rid));ends[combo]=mesh['raw']
            row=read(score_path(combo,rid)/'row.json');score=read(score_path(combo,rid)/'score.json')
            verify(read(score_path(combo,rid)/'complete.json')['files'])
            assert score['C_preserved_exact'] and score['C_original_order_preserved']
            assert row['actual_free_faces']==q['N']-q['K'] and row['deleted_free_faces']==0
            row.update(combo=combo,part=q['part_type'],data_origin='CACHED_UNCHANGED' if combo in ('CC','UU') else 'NEW_SUFFIX25',
                C_preserved_exact=score['C_preserved_exact'],C_original_order_preserved=score['C_original_order_preserved'],
                raw_sha256=sha(raw_path(combo,rid)),score_sha256=sha(score_path(combo,rid)/'row.json'),
                query_descriptor_sha256=sha(q['query_descriptor']['path']),
                C_query_count=score['C_query_count'],C_query_weight_denominator=score['C_query_weight_denominator'],
                CF_tested_pair_denominator=score['CF_tested_pair_denominator'],FF_tested_pair_denominator=score['FF_tested_pair_denominator'],
                S_status=score['status_by_metric']['S'],E_status=score['status_by_metric']['E'],GT_status=score['status_by_metric']['GT'])
            allrows.append(row)
            if combo in ('CC','UU'):
                base=oldreview[(rid,MODELS[combo[0]])]
                adoption=base['C_adoption'];obs=base['observation'];body=base['body_completeness'];confidence='INHERITED_RECHECKED'
                tr=arrays(q['baseline'][MODELS[combo[0]]]['trajectory']);trajectory[combo]=tr['states'][25:]
            else:
                adoption=nn[combo];obs=nn[combo+'_note'];body=notes['all_new_body_completeness'];confidence=nn.get(combo+'_confidence','MEDIUM_HIGH')
                tr=arrays(raw_path(combo,rid).parent/'trajectory.npz');trajectory[combo]=tr['states']
                receipt=read(raw_path(combo,rid).parent/'receipt.json');verify(receipt['files'])
                assert tr['states'].shape==(26,q['N'],3,3) and tr['velocities'].shape==(25,q['N'],3,3)
                assert np.array_equal(tr['states'][0],arrays(q['starts'][MODELS[combo[0]]])['state'])
                assert np.array_equal(tr['states'][-1],mesh['raw'])
                assert np.all(tr['states'][:,fixed]==tr['states'][0,fixed][None])
                for i in range(25):
                    expected=tr['states'][i]+np.float32(.02)*tr['velocities'][i];expected[fixed]=tr['states'][0,fixed]
                    assert np.array_equal(expected,tr['states'][i+1])
                integrity.append(dict(combo=combo,request_id=rid,cached_start_exact=True,states=26,returned_calls=25,
                    all_anchors_exact=True,export_endpoint_exact=True,euler_recurrence_exact=True,RNG_unchanged=receipt['RNG_unchanged']))
            review.append(dict(request_id=rid,parent_id=q['parent_id'],part=q['part_type'],split=q['split'],noise_index=q['noise_index'],
                combo=combo,stage='RAW',reviewer='AI_PROXY',blind=False,review_date='2026-10-22',C_adoption=adoption,
                body_completeness=body,confidence=confidence,observation=obs,image='gallery/'+page.name,image_sha256=pagehash,
                original_label_preserved=combo in ('CC','UU'),exact_connectivity='NOT_CERTIFIED_BY_IMAGE'))
        # Supplementary, explicitly not a shape-quality metric: paired noise-slot RMS.
        for a,b in itertools.combinations(COMBOS,2):
            for index in range(26):
                d=trajectory[a][index].astype(float)-trajectory[b][index]
                value=float(np.sqrt(np.mean(np.sum(d[~fixed]**2,axis=-1))))
                paths.append(dict(request_id=rid,split=q['split'],part=q['part_type'],pair=a+'_'+b,t=.5+index*.02,
                    moving_corner_count=int((~fixed).sum()),RMS_vector_distance_model=value,RMS_vector_distance_fraction_L=value/L,
                    status='SUPPLEMENTARY_SLOT_PAIRED_DISTANCE_NOT_QUALITY_OR_GT'))
                if index==25:distances.append(paths[-1])
    assert len(allrows)==len(review)==64 and len(integrity)==32
    csvwrite(OUT/'analysis/per_output_all_metrics.csv',allrows)
    csvwrite(OUT/'analysis/visual_review.csv',review)
    csvwrite(OUT/'analysis/integrity.csv',integrity)
    csvwrite(OUT/'analysis/call_ledger.csv',callrows)
    csvwrite(OUT/'analysis/paired_trajectory_distance.csv',paths)
    csvwrite(OUT/'analysis/paired_endpoint_distance.csv',distances)
    numeric=[k for k in allrows[0] if k not in ('noise_index','N','K') and any(isinstance(r.get(k),(int,float)) and not isinstance(r.get(k),bool) for r in allrows)]
    grouped=[];parents=[];paired=[]
    for split in ('CURRENT_TRAIN','DEV'):
        for combo in COMBOS:
            rr=[r for r in allrows if r['split']==split and r['combo']==combo]
            for metric in numeric:
                vals=[float(r[metric]) for r in rr if isinstance(r.get(metric),(int,float)) and not isinstance(r.get(metric),bool) and np.isfinite(r[metric])]
                grouped.append(dict(split=split,combo=combo,metric=metric,requests=len(rr),valid_numeric=len(vals),NA=len(rr)-len(vals),
                    mean=float(np.mean(vals)) if vals else None,median=float(np.median(vals)) if vals else None,
                    minimum=min(vals) if vals else None,maximum=max(vals) if vals else None,
                    I_LOW_SUPPORT=sum(r['ConflictCov_C_status']=='LOW_SUPPORT' for r in rr)))
    for q in qs[::2]:
        pid=q['parent_id']
        for combo in COMBOS:
            rr=[r for r in allrows if r['parent_id']==pid and r['combo']==combo]
            vv=[r for r in review if r['parent_id']==pid and r['combo']==combo]
            row=dict(parent_id=pid,condition_id=q['condition_id'],part=q['part_type'],split=q['split'],combo=combo,noises=2,
                adoption_YES=sum(v['C_adoption']=='YES' for v in vv),adoption_labels='|'.join(v['C_adoption'] for v in vv))
            for metric in numeric:
                vals=[float(r[metric]) for r in rr if isinstance(r.get(metric),(int,float)) and np.isfinite(r[metric])]
                row[metric]=float(np.mean(vals)) if vals else None
            parents.append(row)
    for q in qs:
        rr={r['combo']:r for r in allrows if r['request_id']==q['request_id']}
        vv={r['combo']:r for r in review if r['request_id']==q['request_id']}
        for metric in numeric:
            vals={c:rr[c].get(metric) for c in COMBOS}
            valid=all(isinstance(v,(int,float)) and np.isfinite(v) for v in vals.values())
            row=dict(request_id=q['request_id'],parent_id=q['parent_id'],noise_index=q['noise_index'],split=q['split'],part=q['part_type'],metric=metric,
                status='NUMERIC_PAIRED' if valid else 'NA',**vals,**{c+'_adoption':vv[c]['C_adoption'] for c in COMBOS})
            if valid:
                row.update(USER_suffix_given_COMMON_prefix=vals['CU']-vals['CC'],USER_suffix_given_USER_prefix=vals['UU']-vals['UC'],
                    USER_prefix_given_COMMON_suffix=vals['UC']-vals['CC'],USER_prefix_given_USER_suffix=vals['UU']-vals['CU'],
                    interaction=vals['UU']-vals['UC']-vals['CU']+vals['CC'])
            paired.append(row)
    csvwrite(OUT/'analysis/all_metric_summary.csv',grouped)
    csvwrite(OUT/'analysis/per_parent_paired.csv',parents)
    csvwrite(OUT/'analysis/per_request_factorial.csv',paired)
    qualitative=[]
    for q in qs:
        rr={r['combo']:r for r in review if r['request_id']==q['request_id']}
        qualitative.append(dict(request_id=q['request_id'],parent_id=q['parent_id'],noise=q['noise_index'],split=q['split'],part=q['part_type'],
            **{c:rr[c]['C_adoption'] for c in COMBOS},CU_keeps_CC_label=rr['CU']['C_adoption']==rr['CC']['C_adoption'],
            UC_keeps_UU_label=rr['UC']['C_adoption']==rr['UU']['C_adoption']))
    csvwrite(OUT/'analysis/adoption_paired.csv',qualitative)
    compact=[]
    for split in ('CURRENT_TRAIN','DEV'):
        for combo in COMBOS:
            rr=[r for r in allrows if r['split']==split and r['combo']==combo];vv=[r for r in review if r['split']==split and r['combo']==combo]
            compact.append(dict(split=split,combo=combo,n=len(rr),YES=sum(r['C_adoption']=='YES' for r in vv),
                CF_pct=100*np.mean([r['CF_illegal_affected_free_fraction'] for r in rr]),FF_pct=100*np.mean([r['FF_illegal_affected_free_fraction'] for r in rr]),
                I_all_pct=100*np.mean([r['ConflictCov_C'] for r in rr]),I_PASS_pct=100*np.mean([r['ConflictCov_C'] for r in rr if r['ConflictCov_C_status']=='PASS']),
                I_low_support=sum(r['ConflictCov_C_status']=='LOW_SUPPORT' for r in rr),
                small_angle_pct=100*np.mean([r['free_minimum_angle_lt5_fraction'] for r in rr]),
                free_area=np.mean([r['free_total_surface_area'] for r in rr]),illegal_FF_pairs=np.mean([r['FF_illegal_pair_count'] for r in rr])))
    dsummary=[]
    for split in ('CURRENT_TRAIN','DEV'):
        for pair in sorted({r['pair'] for r in distances}):
            vals=[r['RMS_vector_distance_fraction_L']*100 for r in distances if r['split']==split and r['pair']==pair]
            dsummary.append(dict(split=split,pair=pair,mean_pct_L=float(np.mean(vals)),median_pct_L=float(np.median(vals))))
    summary=dict(metrics=compact,endpoint_distance=dsummary,all_numeric_fields=len(numeric),per_output_fields=len(allrows[0]),
        cases=qualitative,new_generator_calls=800,training=0,D0=0,head=0,new_noise=0,new_OT=0,
        sensitivity='UC TRAIN 5/8 strict; 6/8 if borderline 943dde/n0 is treated YES. CU 1/8, UU 6/8, CC 1/8 unchanged.')
    save(OUT/'analysis/summary.json',summary)
    save(OUT/'analysis/complete.json',dict(status='COMPLETE',outputs=64,new_RAW=32,reused_RAW=32,visual_review_cells=64,visual_review_pages=16,
        files={str(p):sha(p) for p in (OUT/'analysis').iterdir() if p.is_file()}))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
