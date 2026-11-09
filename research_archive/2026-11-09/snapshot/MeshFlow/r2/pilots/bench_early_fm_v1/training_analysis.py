"""Paired original FM errors, not a comparison of differently weighted losses."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from .common import *
import numpy as np, collections, json, argparse

def loadlines(path):return [json.loads(s) for s in path.read_text().splitlines()]

def main():
    status=read(OUT/'training'/ARM/'status.json');assert status['status']=='COMPLETE'
    original=OLD_TRAIN/'training/USER_V2_SEAM';new=OUT/'training'/ARM
    logs={label:loadlines(base/'training_log.jsonl') for label,base in [('USER',original),('EARLY',new)]}
    exposure={label:loadlines(base/'exposure.jsonl') for label,base in [('USER',original),('EARLY',new)]}
    assert all(len(v)==1500 for v in logs.values()) and all(len(v)==12000 for v in exposure.values())
    rows=[];audit=[];binrows=[];parents=[];clips=[]
    for a,b in zip(exposure['USER'],exposure['EARLY']):
        keys=['local_step','microbatch','parent_id','condition_id','part_type','scale_xyz','noise_seed','epsilon_seed','time_seed','permutation_seed','t','N','K','all_free_scalar_count','bulk_scalar_count','seam_tip_scalar_count']
        assert all(a[k]==b[k] for k in keys),'Unpaired training stream'
        assert b['time_weight']==time_weight(b['t'])
    first_equal=all(all(a[k]==b[k] for k in ['all_free_velocity_sse','bulk_velocity_sse','seam_tip_velocity_sse','endpoint_sse']) for a,b in zip(exposure['USER'][:8],exposure['EARLY'][:8]))
    assert first_equal
    for label in ('USER','EARLY'):
        for r in logs[label]:
            rows.append(dict(label=label,local_step=r['local_step'],weighted_objective=r['objective'],
                unweighted_FM=r['all_free_velocity_sse']/r['all_free_scalar_count'],
                seam_tip_MSE=r['seam_tip_velocity_sse']/r['seam_tip_scalar_count'] if r['seam_tip_scalar_count'] else None,
                bulk_MSE=r['bulk_velocity_sse']/r['bulk_scalar_count'],gradient_norm=r['generator_gradient_norm'],
                gradient_clipped=r['generator_gradient_norm']>1,update_seconds=r['update_seconds'],input_seconds=r['input_seconds'],
                peak_allocated_bytes=r['peak_allocated_bytes'],peak_reserved_bytes=r['peak_reserved_bytes']))
        for lo,hi in ((1,250),(251,500),(501,750),(751,1000),(1001,1250),(1251,1500)):
            for tl,th in ((0.,.25),(.25,.5),(.5,.75),(.75,1.)):
                rr=[r for r in exposure[label] if lo<=r['local_step']<=hi and tl<=r['t']<th]
                rec=dict(label=label,step_start=lo,step_end=hi,t_start=tl,t_end=th,instances=len(rr))
                for name,sk,nk,ek in [('all','all_free_velocity_sse','all_free_scalar_count','endpoint_sse'),('tip','seam_tip_velocity_sse','seam_tip_scalar_count','seam_tip_endpoint_sse'),('bulk','bulk_velocity_sse','bulk_scalar_count','bulk_endpoint_sse')]:
                    ss=sum(r[sk] for r in rr);count=sum(r[nk] for r in rr);es=sum(r[ek] for r in rr)
                    rec.update({name+'_sse':ss,name+'_scalar_count':count,name+'_velocity_MSE':ss/count if count else None,name+'_endpoint_MSE':es/count if count else None})
                binrows.append(rec)
            ll=[r for r in logs[label] if lo<=r['local_step']<=hi]
            clips.append(dict(label=label,start=lo,end=hi,updates=len(ll),clipped=sum(r['generator_gradient_norm']>1 for r in ll),
                mean_norm=float(np.mean([r['generator_gradient_norm'] for r in ll])),mean_unweighted=float(np.mean([r['all_free_velocity_sse']/r['all_free_scalar_count'] for r in ll]))))
        group=collections.defaultdict(list)
        for r in exposure[label]:group[(r['parent_id'],r['part_type'])].append(r)
        for (pid,part),rr in group.items():parents.append(dict(label=label,parent_id=pid,part_type=part,instances=len(rr),
            early_instances=sum(r['t']<.5 for r in rr),mean_time_weight=float(np.mean([r.get('time_weight',1.) for r in rr])),
            unweighted_MSE=sum(r['all_free_velocity_sse'] for r in rr)/sum(r['all_free_scalar_count'] for r in rr)))
    csvwrite(OUT/'analysis/training_per_update.csv',rows);csvwrite(OUT/'analysis/training_time_bins.csv',binrows)
    csvwrite(OUT/'analysis/training_parent_exposure.csv',parents);csvwrite(OUT/'analysis/gradient_clipping.csv',clips)
    events=loadlines(new/'calls.jsonl')
    counts=collections.Counter((e.get('kind'),e.get('event')) for e in events)
    expected={'train_forward':12000,'train_backward':12000,'optimizer_update':1500}
    assert status['calls']['attempted']==status['calls']['successful']==expected
    assert not status['calls']['failed'] and not status['calls']['pending_call_ids']
    for kind,count in expected.items():
        assert counts[(kind,'CALL_BEGIN')]==counts[(kind,'CALL_DONE')]==count
    save(OUT/'analysis/training_integrity.json',dict(status='PASS',paired_exposures=12000,first_eight_forward_statistics_bit_equal=first_equal,
        exact_initial_generator_and_optimizer=True,source_registration_sha256=sha(OLD_TRAIN/'training_registration.json'),
        final_status_calls=status['calls'],recorded_event_fields=list(events[0]),event_type_counts={str(k):v for k,v in counts.items()},
        parameter_count=status.get('generator_parameters',130401155),updates=1500,intermediate_checkpoint_selection=False))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(12,7),dpi=140)
    for ax,key,title in zip(axs.flat,['unweighted_FM','tip_velocity_MSE','bulk_velocity_MSE','gradient_norm'],['Original FM (50-update mean)','Seam-tip MSE by time','Bulk MSE by time','Gradient norm (50-update mean)']):
        if key in ('unweighted_FM','gradient_norm'):
            for label in ('USER','EARLY'):
                rr=[r for r in rows if r['label']==label];vals=[r[key] for r in rr]
                ax.plot(np.arange(25,1500,50),[np.mean(vals[j:j+50]) for j in range(0,1500,50)],label=label)
        else:
            for label in ('USER','EARLY'):
                for early in (True,False):
                    vv=[]
                    for end in (250,500,750,1000,1250,1500):
                        rr=[r for r in binrows if r['label']==label and r['step_end']==end and (r['t_start']<.5)==early]
                        prefix=key.split('_')[0];den=sum(r[prefix+'_scalar_count'] for r in rr)
                        vv.append(sum(r[prefix+'_sse'] for r in rr)/den if den else np.nan)
                    ax.plot([250,500,750,1000,1250,1500],vv,label=label+(' t<.5' if early else ' t>=.5'),linestyle='-' if early else '--')
        ax.set_title(title);ax.set_xlabel('Local update');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Same initialization and12000 examples; original unweighted errors, no extra model probes')
    fig.tight_layout();fig.savefig(OUT/'analysis/training_curves.png');plt.close(fig)
    print('TRAINING_INTEGRITY_PASS12000',clips[-1])

if __name__=='__main__':main()
