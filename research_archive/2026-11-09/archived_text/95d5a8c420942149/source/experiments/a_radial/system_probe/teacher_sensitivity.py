"""Training-only teacher-neighborhood angular sensitivity; not a new mesh method."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..exporter import RadialSupport
from ..io_utils import new_dir,write_json,file_hash
from ..layout_student.deployment import named_quality
from ..legacy.chart import dets


def run(source,out):
    source=Path(source).resolve();out=new_dir(out);p=json.loads((source/'RUN_PROTOCOL.json').read_text())
    label_path=source/'train_labels.npz';sha=file_hash(label_path)
    assert sha==p['hashes']['training_labels_sha256']
    with np.load(label_path,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    uids=data['train_uids'].tolist();assert uids==p['train_uids'] and not set(uids)&set(p['internal_check_uids'])
    fractions=[0.,.0001,.001,.01,.1,.5,1.];trials=16
    protocol={'study':'TEACHER-NEIGHBORHOOD-ANGULAR-SENSITIVITY','source_protocol_sha256':file_hash(source/'RUN_PROTOCOL.json'),
        'training_labels_sha256':sha,'source_sha256':file_hash(__file__),'uids':uids,'fraction_RMS_of_teacher_displacement':fractions,
        'trials_per_parent':trials,'seed_base':1393501,'seed_rule':'1393501+16*training_parent_index+trial',
        'perturbation':'isotropic Gaussian tangent to teacher directions, normalized to unit component RMS; same noise across fractions; normalize onto sphere',
        'angular_tolerance_deg':1e-6,'zero_teacher':'retained stationary control',
        'scope':'Only original12 training parents; no RMS optimization, no model training or selection, no exact intersection audit; angular pass is not complete validity',
        'aggregation':'per-parent16 fixed random perturbations; not independent real objects or generalization samples'}
    write_json(out/'RUN_PROTOCOL.json',protocol);support=RadialSupport();rows=[]
    for i,uid in enumerate(uids):
        u0,u1,y=data['directions0'][i],data['directions1'][i],data['logr'][i];faces=data['faces']
        energy=float(np.square(u1-u0).mean());base_r=support.query(y,u0)
        floor=named_quality(u0*base_r[:,None],faces)
        for trial in range(trials):
            noise=np.random.default_rng(1393501+16*i+trial).normal(size=u1.shape)
            noise-=np.sum(noise*u1,axis=-1,keepdims=True)*u1
            noise/=np.sqrt(np.square(noise).mean())
            for fraction in fractions:
                row={'uid':uid,'trial':trial,'fraction':fraction,'teacher_energy':energy,'status':'ok'}
                try:
                    u=u1.copy() if fraction==0 or energy==0 else u1+fraction*np.sqrt(energy)*noise
                    if fraction and energy:u/=np.linalg.norm(u,axis=-1,keepdims=True)
                    r=support.query(y,u);q=named_quality(u*r[:,None],faces)
                    margin={k:q[k]-floor[k] for k in ('min_triangle_angle_deg','triangle_min_angle_p05_deg')}
                    row.update(angular_pass=all(v>=-1e-6 for v in margin.values()),margins_deg=margin,
                        direction_determinants_positive=bool(dets(u,faces).min()>0),
                        direction_MSE_to_teacher=float(np.square(u-u1).mean()),min_angle=q['min_triangle_angle_deg'],p05_angle=q['triangle_min_angle_p05_deg'])
                except Exception as e:row.update(status='error',error=repr(e),angular_pass=False,direction_determinants_positive=False)
                rows.append(row)
    summary=[];per_parent=[]
    moving=[u for i,u in enumerate(uids) if np.square(data['directions1'][i]-data['directions0'][i]).mean()>0]
    for uid in uids:
        for fraction in fractions:
            group=[r for r in rows if r['uid']==uid and r['fraction']==fraction]
            per_parent.append({'uid':uid,'fraction':fraction,'moving_teacher':uid in moving,'planned':trials,
                'angular_pass':sum(r['angular_pass'] for r in group),'errors':sum(r['status']!='ok' for r in group)})
    for fraction in fractions:
        g=[r for r in rows if r['fraction']==fraction and r['uid'] in moving]
        summary.append({'fraction':fraction,'planned_moving_perturbations':len(moving)*trials,'angular_pass':sum(r['angular_pass'] for r in g),
            'direction_positive':sum(r['direction_determinants_positive'] for r in g),'errors':sum(r['status']!='ok' for r in g),
            'angular_pass_fraction':sum(r['angular_pass'] for r in g)/(len(moving)*trials)})
    assert len(rows)==12*len(fractions)*trials
    write_json(out/'RESULTS.json',{'rows':rows,'planned':len(rows),'errors':sum(r['status']!='ok' for r in rows)})
    write_json(out/'SUMMARY.json',{'moving_parents':moving,'stationary_parents':[u for u in uids if u not in moving],
        'per_parent':per_parent,'moving_aggregate':summary,'not_full_validity_or_generalization':True})
    fig,ax=plt.subplots(figsize=(8,4.5))
    for uid in moving:
        g=[r for r in per_parent if r['uid']==uid and r['fraction']>0]
        ax.plot([r['fraction'] for r in g],[r['angular_pass']/16 for r in g],marker='.',alpha=.55,label=uid[:8])
    ax.plot([r['fraction'] for r in summary if r['fraction']>0],[r['angular_pass_fraction'] for r in summary if r['fraction']>0],color='black',linewidth=2.5,label='equal-parent average')
    ax.set_xscale('log');ax.set(xlabel='Tangent perturbation RMS / teacher displacement RMS',ylabel='Fraction passing both angular conditions',ylim=(-.03,1.03),title='Teacher-neighborhood sensitivity: 11 moving training parents, 16 fixed trials')
    ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2,bbox_to_anchor=(1,1),loc='upper left');fig.tight_layout()
    fig.savefig(out/'sensitivity.png',dpi=160);fig.savefig(out/'sensitivity.pdf');plt.close(fig)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-run',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.source_run,a.out)