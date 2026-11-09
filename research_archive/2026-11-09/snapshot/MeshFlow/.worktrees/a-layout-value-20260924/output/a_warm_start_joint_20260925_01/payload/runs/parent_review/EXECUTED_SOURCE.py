"""Read saved warm-start results; report all fixed parents and budgets without resampling."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..io_utils import file_hash,write_json,new_dir,array_hash

MODELS=['fixed','longfit_10','longfit_11','balanced_10','balanced_11']
COLORS=['#555555','#4477aa','#66ccee','#cc6677','#aa3377']

def main(run,out):
    run=Path(run).resolve();out=new_dir(out)
    result=json.loads((run/'RESULTS.json').read_text());summary=json.loads((run/'SUMMARY.json').read_text())
    p=json.loads((run/'RUN_PROTOCOL.json').read_text());rows=[r for a in result['attempts'] for r in a['rows']]
    by={(r['uid'],r['model'],r['kind']):r for r in rows}
    assert len(rows)==len(by)==120 and len(result['attempts'])==20
    uids=p['uids'];assert len(uids)==4 and set(p['models'])==set(MODELS)
    errors=[];validations=[];table=[]
    for a in result['attempts']:
        uid=a['uid'];name=a['model'];raw=by[uid,name,'raw'];final=by[uid,name,'optimizer_final'];fixed=by[uid,'fixed','raw'];fixedopt=by[uid,'fixed','optimizer_final']
        for r in a['rows']:
            if r.get('status')!='success':errors.append({'uid':uid,'model':name,'kind':r['kind'],'status':r.get('status')});continue
            with np.load(run/r['mesh_file'],allow_pickle=False) as z:
                assert z['vertices'].shape==(258,3) and z['faces'].shape==(512,3)
                assert array_hash(z['vertices'],z['faces'])==r['stored_array_hash']
                assert array_hash(z['vertices'],z['faces'].astype(np.int32))==r['mesh_hash']
                assert file_hash(run/r['mesh_file'])==r['mesh_npz_sha256']
            assert r['mainfield_hash']==fixed['mainfield_hash'] and r['GT_reference_hash']==fixed['GT_reference_hash']
            assert [m['seed'] for m in r['metrics']]==[m['seed'] for m in fixed['metrics']]
            assert all(m['points_per_direction']==8192 for m in r['metrics'])
            validations.append([uid,name,r['kind']])
        def rms(row):return np.array([m['rms_bbox_percent'] for m in row['metrics']]) if row.get('status')=='success' else np.array([np.nan,np.nan])
        finalgain=100*(1-rms(final)/rms(fixed));vsopt=100*(1-rms(final)/rms(fixedopt))
        stats=a.get('optimizer_stats',{})
        table.append({'uid':uid,'model':name,'raw_qualified':raw.get('qualified',False),'final_qualified':final.get('qualified',False),
          'final_gain_vs_Fixed_raw_two_MC':finalgain.tolist(),'final_gain_vs_Fixed_opt16_two_MC':vsopt.tolist(),
          'joint_success_vs_Fixed_raw':bool(final.get('qualified') and (finalgain>=2).all()),
          'joint_success_vs_Fixed_opt16':bool(final.get('qualified') and (vsopt>=2).all()),
          'actual_iterations':stats.get('iterations'),'first_feasible_iteration':stats.get('first_feasible_iteration'),
          'best_feasible_iteration':stats.get('best_feasible_iteration'),
          'initialization_plus_optimizer_seconds':a.get('initialization_plus_optimizer_seconds')})
    fig,axes=plt.subplots(2,2,figsize=(12,8));xs=np.array([0,1,4,16],float)
    for ax,uid in zip(axes.flat,uids):
        fixed=np.array([m['rms_bbox_percent'] for m in by[uid,'fixed','raw']['metrics']])
        for name,color in zip(MODELS,COLORS):
            data=[];qualified=[]
            for step in [0,1,4,16]:
                r=by[uid,name,'snapshot_'+str(step)]
                vals=np.array([m['rms_bbox_percent'] for m in r['metrics']])/fixed if r.get('status')=='success' else np.array([np.nan,np.nan])
                data.append(vals);qualified.append(r.get('qualified',False))
            data=np.array(data);means=data.mean(1)
            ax.plot(xs,means,'-',color=color,label=name,alpha=.8)
            ax.fill_between(xs,data.min(1),data.max(1),color=color,alpha=.12)
            for x,y,q in zip(xs,means,qualified):ax.scatter(x,y,color=color,marker='o' if q else 'x',s=28)
            r=by[uid,name,'optimizer_final'];value=np.mean(np.array([m['rms_bbox_percent'] for m in r['metrics']])/fixed) if r.get('status')=='success' else np.nan
            ax.scatter(18,value,color=color,marker='*' if r.get('qualified') else 'x',s=65)
        ax.axhline(1,color='#999999',lw=.8,ls='--');ax.set_title(uid[:8]+' (seen train parent)')
        ax.set_xticks([0,1,4,16,18],['0','1','4','16','returned']);ax.set_xlabel('Optimizer iteration / original returned endpoint')
        ax.set_ylabel('Same-field RMS / raw Fixed RMS');ax.grid(alpha=.15)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5,bbox_to_anchor=(.5,.028))
    fig.suptitle('Learned initialization + fixed 16-step optimizer: all four registered parents',fontsize=13)
    fig.text(.5,.007,'Line: mean of two MC ratios; band: their range (not a confidence interval).  o / star: qualified; x: failed quality.  No checkpoint selection.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.085,1,.95]);fig.savefig(out/'warm_start_all_parents.png',dpi=160);fig.savefig(out/'warm_start_all_parents.pdf');plt.close(fig)
    grouped=[]
    for name in MODELS:
        group=[r for r in table if r['model']==name]
        grouped.append({'model':name,'raw_qualified':sum(r['raw_qualified'] for r in group),'final_qualified':sum(r['final_qualified'] for r in group),
           'joint_vs_Fixed_raw':sum(r['joint_success_vs_Fixed_raw'] for r in group),'joint_vs_Fixed_opt16':sum(r['joint_success_vs_Fixed_opt16'] for r in group)})
    write_json(out/'REVIEW.json',{'status':'PASS' if not errors else 'COMPLETE_WITH_FAILED_SLOTS','validated_meshes':len(validations),'errors':errors,'rows':table,'groups':grouped,
      'reference':'same radial field complete2050node4096face surface; historical GT label corrected outside frozen protocol',
      'all_requested_rows_accounted':True,'results_sha256':file_hash(run/'RESULTS.json'),'protocol_sha256':file_hash(run/'RUN_PROTOCOL.json'),
      'source_sha256':file_hash(__file__),'no_training_optimization_or_MC_recomputation':True})
    (out/'EXECUTED_SOURCE.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'groups':grouped,'errors':errors,'validated_meshes':len(validations)},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();main(a.run,a.out)