"""CPU-only complete numeric collection; no reranking or new model calls."""
from .common import *
import numpy as np,collections,math
from .geometry import roles,projection,shape

def main():
    out=OUT/'analysis_full';done=out/'numeric_complete.json'
    if done.exists():verify(read(done)['files']);return
    qs=read(OUT/'evaluation/score_registrations/COMMON.json')['requests'];qby={q['request_id']:q for q in qs}
    def readcsv(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
    fixed=[];paired=[];endpoints=[];checks=[];response=[]
    for q in [q for q in qs if q['split']=='CURRENT_TRAIN']:
        rid=q['request_id'];s=arrays(q['fixed_sample_npz']);masks=roles(s)
        for t in manifest()['times']:
            pred={m:arrays(OUT/'execution'/m/'fixed'/rid/f'prediction_t{t}.npz') for m in LABELS}
            assert np.array_equal(pred['COMMON']['x'],pred['USER']['x'])
            for m in LABELS:
                x,v=pred[m]['x'],pred[m]['v'];row=dict(label=m,request_id=rid,parent_id=q['parent_id'],part=q['part_type'],t=t,**projection(x,v,s))
                for r,mask in masks.items():
                    row[r+'_velocity_mse']=float(np.mean((v[mask]-s['u'][mask])**2)) if mask.any() else None
                row.update({'truth_'+k:a for k,a in projection(x,s['u'],s).items() if k.endswith('derivative')});fixed.append(row)
                err=(x+(1-t)*v-s['x1'])[masks['all']];identity=(1-t)*(v-s['u'])[masks['all']]
                checks.append(dict(request_id=rid,label=m,t=t,endpoint_identity_max_error=float(np.abs(err-identity).max())))
            for r,mask in masks.items():
                if not mask.any():continue
                a=float(np.mean((pred['COMMON']['v'][mask]-s['u'][mask])**2));b=float(np.mean((pred['USER']['v'][mask]-s['u'][mask])**2))
                paired.append(dict(request_id=rid,parent_id=q['parent_id'],part=q['part_type'],t=t,region=r,scalar_denominator=int(mask.sum()*3),COMMON_MSE=a,USER_MSE=b,delta_MSE=b-a,ratio=b/a if a else None))
        for start in manifest()['continuations']:
            tr=arrays(OUT/'execution/USER/fixed'/rid/f'continuation_t{start}.npz');p=tr['states'];v=tr['velocities']
            for stage,i in [('START',0),('END',len(p)-1)]:
                row=dict(request_id=rid,parent_id=q['parent_id'],part=q['part_type'],start_t=start,stage=stage,**shape(p[i],s),**projection(p[i],np.zeros_like(p[i]),s))
                endpoints.append(row)
    csvwrite(out/'fixed_state_full.csv',fixed);csvwrite(out/'fixed_state_paired.csv',paired);csvwrite(out/'continuation_start_end.csv',endpoints);csvwrite(out/'endpoint_identity_check.csv',checks)
    for q in qs[::2]:
        rid=q['request_id'];lay=arrays(OUT/'layouts'/q['condition_id']/'layout.npz')
        for t in manifest()['response_times']:
            banks={m:arrays(OUT/'execution'/m/'response'/rid/f'response_t{t}.npz') for m in LABELS}
            assert np.array_equal(banks['COMMON']['Cminus'],banks['USER']['Cminus']) and np.array_equal(banks['COMMON']['Cplus'],banks['USER']['Cplus'])
            den=float(np.sqrt(np.mean((banks['COMMON']['Cplus']-banks['COMMON']['Cminus'])**2)))
            for r,mask in roles(lay).items():
                if not mask.any():continue
                d={m:(banks[m]['vplus']-banks[m]['vminus'])[mask].astype(float) for m in LABELS}
                norm={m:float(np.linalg.norm(d[m])) for m in LABELS}
                response.append(dict(request_id=rid,condition_id=q['condition_id'],split=q['split'],part=q['part_type'],t=t,region=r,C_delta_RMS=den,
                    COMMON_response=float(np.sqrt(np.mean(d['COMMON']**2))/den),USER_response=float(np.sqrt(np.mean(d['USER']**2))/den),
                    response_direction_cosine=float(np.sum(d['COMMON']*d['USER'])/(norm['COMMON']*norm['USER'])) if norm['COMMON']*norm['USER'] else None,
                    scalar_denominator=int(mask.sum()*3)))
    csvwrite(out/'response_paired.csv',response)
    metrics=[]
    for label in LABELS:
        for q in qs:
            for stage in ('RAW','BASE_POST','CLEAN_FINAL_POST'):
                p=OUT/'evaluation/scores'/label/q['request_id']/'RAW/row.json' if stage=='RAW' else OUT/'cpu'/label/q['request_id']/stage/'scores/row.json'
                r=read(p);r.update(label=label,stage=stage,part=q['part_type']);metrics.append(r)
    csvwrite(out/'per_output_all_metrics.csv',metrics)
    keys=['CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','free_minimum_angle_lt5_fraction','nonmanifold_edges','hinge_diag_isolated_1face_edge_components','C_F_valid_edge_count','free_only_reached_bulk_area_fraction','actual_free_faces','deleted_free_faces']
    grouped=[]
    for label in LABELS:
        for split in ('CURRENT_TRAIN','DEV'):
            for stage in ('RAW','BASE_POST','CLEAN_FINAL_POST'):
                rr=[r for r in metrics if r['label']==label and r['split']==split and r['stage']==stage];row=dict(label=label,split=split,stage=stage,outputs=len(rr))
                for k in keys:
                    values=[r[k] for r in rr if isinstance(r.get(k),(int,float))];row[k]=float(np.mean(values)) if values else None;row[k+'__n']=len(values)
                row['bulk_reach_ge30_count']=sum(r.get('free_only_reached_bulk_area_fraction',0)>=.3 for r in rr if isinstance(r.get('free_only_reached_bulk_area_fraction'),(int,float)))
                grouped.append(row)
    csvwrite(out/'summary_by_split_stage.csv',grouped)
    # Frozen request pairing preserves every metric; no favorable-metric selection.
    idx={(r['label'],r['request_id'],r['stage']):r for r in metrics};pr=[]
    for q in qs:
        for stage in ('RAW','BASE_POST','CLEAN_FINAL_POST'):
            a,b=idx['COMMON',q['request_id'],stage],idx['USER',q['request_id'],stage];r=dict(request_id=q['request_id'],split=q['split'],part=q['part_type'],stage=stage)
            for k in a:
                if isinstance(a[k],(int,float)) and not isinstance(a[k],bool) and isinstance(b.get(k),(int,float)):
                    r[k+'__COMMON']=a[k];r[k+'__USER']=b[k];r[k+'__delta']=b[k]-a[k]
            pr.append(r)
    csvwrite(out/'per_request_metric_deltas.csv',pr)
    ledger=[]
    for label in LABELS:
        ee=[json.loads(x) for x in (OUT/'execution'/label/'calls.jsonl').read_text().splitlines()]
        for kind in sorted({e['kind'] for e in ee}):
            a=[e for e in ee if e['kind']==kind];ledger.append(dict(label=label,kind=kind,started=sum(e['stage']=='STARTED' for e in a),returned=sum(e['stage']=='RETURNED' for e in a)))
    csvwrite(out/'forward_budget.csv',ledger)
    assert sum(r['returned'] for r in ledger)==2382 and all(r['started']==r['returned'] for r in ledger)
    save(done,dict(status='COMPLETE',generator_returned=2382,head_calls=4,D0_calls=sum(read(OUT/'evaluation'/l/'post_complete_full.json')['new_D0_forwards'] for l in LABELS),
        model_updates=0,full_metrics=len(metrics),fixed_paired=len(paired),response_paired=len(response),max_endpoint_identity_error=max(x['endpoint_identity_max_error'] for x in checks),
        files={str(p):sha(p) for p in out.glob('*.csv')}))
    print('NUMERIC_COMPLETE',len(metrics),2382)
if __name__=='__main__':main()
