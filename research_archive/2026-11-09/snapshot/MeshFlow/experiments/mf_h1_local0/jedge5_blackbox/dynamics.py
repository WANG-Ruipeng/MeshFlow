"""Offline trajectory diagnostics. No model evaluation or parameter mutation."""
from pathlib import Path
import csv,time
import numpy as np
from experiments.mf_h1_local0.jedge5_diagnosis.common import load_cases,read_json,write_json
from .recorder import OUT,OLD,key

def rms3(x):
    return float(np.sqrt(np.mean(np.sum(np.asarray(x,dtype=np.float64).reshape(-1,3)**2,axis=-1))))

def l2(x):return float(np.linalg.norm(np.asarray(x,dtype=np.float64).ravel()))

def write_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    start=time.perf_counter();folder=OUT/'dynamics';folder.mkdir(exist_ok=True);cases=load_cases();steps=[];aligned=[];summaries=[];allpaths={};recorded=[]
    for label in ('B','J'):
        result=read_json(OUT/'trajectories'/label/'results.json')
        if result['status']!='PASS':raise RuntimeError('Recorder not complete')
        for row in result['rows']:
            with np.load(OUT/row['path']) as z:s=z['states'].copy();v=z['velocities'].copy()
            kcase=(label,row['patch'],row['seed'],row['nfe']);allpaths[kcase]=(s,v)
            L=cases[0,row['patch']]['bbox'];n=row['nfe']
            for k in range(n):
                vf=v[k,12:].astype(np.float64);prev=v[k-1,12:].astype(np.float64) if k else None;dv=vf-prev if k else None
                r=dict(model=label,parent=0,patch=row['patch'],seed=row['seed'],nfe=n,k=k,t=k/n,
                    free_velocity_L2=l2(vf),free_velocity_point_RMS=rms3(vf),free_velocity_point_RMS_bbox_pct=100*rms3(vf)/L,
                    known_raw_velocity_L2=l2(v[k,:12]),known_applied_velocity_L2=0.,
                    free_delta_v_L2=None if k==0 else l2(dv),free_acceleration_difference_L2=None if k==0 else l2(dv)*n,
                    free_delta_v_point_RMS_bbox_pct=None if k==0 else 100*rms3(dv)/L,
                    free_acceleration_point_RMS_bbox_pct=None if k==0 else 100*rms3(dv)*n/L,
                    velocity_cosine_previous=None if k==0 else float(np.sum(vf*prev)/(l2(vf)*l2(prev)+1e-30)),
                    free_actual_update_point_RMS_bbox_pct=100*rms3(s[k+1,12:]-s[k,12:])/L,
                    free_max_vertex_radius_from_origin_bbox=float(np.linalg.norm(s[k,12:].reshape(-1,3),axis=-1).max()/L))
                steps.append(r)
            recorded.append(dict(model=label,patch=row['patch'],seed=row['seed'],nfe=n,
                peak_velocity_k=int(np.argmax([l2(x[12:]) for x in v])),
                peak_delta_v_k=int(1+np.argmax([l2(x) for x in np.diff(v[:,12:].astype(np.float64),axis=0)]))))
    for label in ('B','J'):
        for p in (0,4):
            for seed in (10401,10402):
                L=cases[0,p]['bbox'];s50,v50=allpaths[label,p,seed,50];s100,v100=allpaths[label,p,seed,100]
                oldfp=OLD/'A'/label/f'{label}_P0_patch{p}_seed{seed}_fp32_Euler100.npz'
                with np.load(oldfp) as z:sf=z['states'].copy();ids=z['state_indices'].copy()
                assert np.array_equal(ids,np.arange(101))
                d=np.array([100*rms3(a-b)/L for a,b in zip(s50[:,12:],s100[::2,12:])])
                for k in range(51):
                    r=dict(model=label,parent=0,patch=p,seed=seed,k50=k,k100=2*k,t=k/50,
                        aligned_free_L2=l2(s50[k,12:]-s100[2*k,12:]),aligned_free_point_RMS_bbox_pct=float(d[k]),
                        delta_aligned_RMS_bbox_pct=None if not k else float(d[k]-d[k-1]),
                        aligned_velocity_L2=None if k==50 else l2(v50[k,12:]-v100[2*k,12:]),
                        BF16_100_vs_FP32_100_point_RMS_bbox_pct=100*rms3(s100[2*k,12:]-sf[2*k,12:])/L,
                        BF16_50_vs_FP32_100_point_RMS_bbox_pct=100*rms3(s50[k,12:]-sf[2*k,12:])/L)
                    aligned.append(r)
                top=np.argsort(-np.diff(d),kind='stable')[:5]
                summaries.append(dict(model=label,parent=0,patch=p,seed=seed,final_aligned_point_RMS_bbox_pct=d[-1],
                    first_bitwise_departure_t=next((k/50 for k in range(51) if not np.array_equal(s50[k].view(np.uint32),s100[2*k].view(np.uint32))),None),
                    first_threshold_crossings={str(th):next((k/50 for k,x in enumerate(d) if x>=th),None) for th in (.1,.5,1.,2.)},
                    at_t={str(k/50):float(d[k]) for k in (0,10,25,40,50)},
                    top_five_increases=[dict(k_start=int(k),t_start=float(k/50),t_end=float((k+1)/50),increase_bbox_pct=float(d[k+1]-d[k])) for k in top],
                    final_BF16_100_vs_FP32_100_point_RMS_bbox_pct=100*rms3(s100[-1,12:]-sf[-1,12:])/L))
    write_csv(folder/'per_step.csv',steps);write_csv(folder/'aligned_50_100.csv',aligned)
    write_json(folder/'results.json',dict(status='PASS',summaries=summaries,peaks=recorded,model_forwards=0,optimizer_updates=0,seconds=time.perf_counter()-start,
        definitions=dict(velocity='Actual model output at start of each Euler step; free slots only. Known raw output is separately observed but never applied.',
            acceleration='(v_k-v_{k-1})/dt along each trajectory; includes state and explicit time variation, not a spatial derivative or Jacobian norm.',
            trajectory_distance='Euclidean point RMS across the SAME 300 free slots, divided by registered bbox diagonal. This is an integration discrepancy, not surface error against GT.',
            thresholds='Descriptive first crossings at 0.1,0.5,1,2 percent bbox; not a statistical change point test.',
            fp32='Reuse archived complete FP32/100 states; no new FP32 complete generation.' )))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric,name,ylabel in [('aligned_free_point_RMS_bbox_pct','trajectory_divergence','50 vs 100 state point RMS (% bbox)'),('free_velocity_point_RMS_bbox_pct','velocity','Actual free velocity RMS (% bbox / time)'),('free_acceleration_point_RMS_bbox_pct','velocity_change','Along-path delta v / dt (% bbox / time^2)')]:
        fig,axes=plt.subplots(2,2,figsize=(12,7.5),sharex=True)
        for ax,(p,seed) in zip(axes.flat,[(0,10401),(0,10402),(4,10401),(4,10402)]):
            for label,color in [('B','#3269ad'),('J','#d65d37')]:
                if metric.startswith('aligned'):
                    rr=[r for r in aligned if r['model']==label and r['patch']==p and r['seed']==seed]
                    ax.plot([r['t'] for r in rr],[r[metric] for r in rr],color=color,label=label)
                else:
                    for n,style in [(50,'-'),(100,'--')]:
                        rr=[r for r in steps if r['model']==label and r['patch']==p and r['seed']==seed and r['nfe']==n and r[metric] is not None]
                        ax.plot([r['t'] for r in rr],[r[metric] for r in rr],color=color,ls=style,label=f'{label} Euler{n}')
            ax.set_title(f'P0 patch{p}, seed{seed}');ax.grid(alpha=.25);ax.legend(fontsize=8);ax.set_xlabel('Absolute t');ax.set_ylabel(ylabel)
        fig.tight_layout();fig.savefig(folder/(name+'.png'),dpi=170);plt.close(fig)
    print('DYNAMICS PASS',flush=True)

if __name__=='__main__':main()
