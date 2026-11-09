"""Render the completed, already-recorded learning curve; no model imports."""
from .common import OUT, read, sha, write_new
import csv, json
import numpy as np

def run():
    status=read(OUT/'training/NATIVE/status.json')
    if status['status']!='COMPLETE':return {'status':'PENDING_TRAINING'}
    out=OUT/'results';out.mkdir(exist_ok=True)
    receipt=out/'loss_plot_receipt.json'
    if receipt.exists():
        r=read(receipt)
        assert all(sha(p)==h for p,h in r['files'].items())
        return r
    log=OUT/'training/NATIVE/training_log.jsonl';exposure=OUT/'training/NATIVE/exposure.jsonl'
    rows=[json.loads(s) for s in log.read_text().splitlines()]
    samples=[json.loads(s) for s in exposure.read_text().splitlines()]
    assert len(rows)==1500 and len(samples)==12000
    bins=[]
    for start in range(1,1501,50):
        rr=[r for r in rows if start<=r['stage_step']<start+50]
        block=dict(start_step=start,end_step=start+49,FM_mean=float(np.mean([r['FM'] for r in rr])),
                   gradient_norm_mean=float(np.mean([r['generator_gradient_norm'] for r in rr])))
        for group in ('OLD','NEW'):
            ss=[s for s in samples if s['replay_group']==group and start<=s['stage_step']<start+50]
            denominator=sum(s['all_free_scalar_count'] for s in ss)
            block[group+'_velocity_MSE']=sum(s['all_free_velocity_sse'] for s in ss)/denominator
            block[group+'_endpoint_MSE']=sum(s['all_free_endpoint_sse'] for s in ss)/denominator
            block[group+'_exposures']=len(ss)
        bins.append(block)
    table=out/'learning_curve_50step_bins.csv'
    with table.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(bins[0]));w.writeheader();w.writerows(bins)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,constrained_layout=True)
    x=[(b['start_step']+b['end_step'])/2 for b in bins]
    axes[0].plot(x,[b['FM_mean'] for b in bins],color='#2a5c91',marker='.',label='Original FM, 50-update mean')
    axes[0].set_ylabel('Batch FM');axes[0].legend()
    for group,color in (('OLD','#4863a0'),('NEW','#cf6d32')):
        axes[1].plot(x,[b[group+'_velocity_MSE'] for b in bins],color=color,marker='.',label=group+' free-coordinate MSE')
    axes[1].set_ylabel('Velocity MSE');axes[1].legend()
    axes[2].plot(x,[b['gradient_norm_mean'] for b in bins],color='#647651')
    axes[2].set_ylabel('Pre-clip gradient norm');axes[2].set_xlabel('New Native optimizer updates (global18000 + local step)')
    for a in axes:
        a.grid(alpha=.2)
        a.set_ylim(bottom=0)
    fig.suptitle('Bench seat coverage: 1500 fixed updates, OLD22 + NEW2\nRecorded training states; these curves do not certify rollout quality')
    path=out/'learning_curve.png';fig.savefig(path,dpi=160);plt.close(fig)
    result=dict(status='COMPLETE',updates=len(rows),exposures=len(samples),model_calls=0,
        sources={str(p):sha(p) for p in (log,exposure)},files={str(p):sha(p) for p in (table,path)},
        interpretation='50-update bins. OLD/NEW MSE weighted by actual free-coordinate count; not independent eval nor identical-request source comparison.')
    write_new(receipt,result);return result

if __name__=='__main__':print(run())
