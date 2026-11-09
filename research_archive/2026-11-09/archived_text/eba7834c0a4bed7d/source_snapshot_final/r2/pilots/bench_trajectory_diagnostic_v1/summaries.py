"""Deterministic summary tables and scientific plots from saved numeric evidence."""
from .common import *
import numpy as np,collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    d=OUT/'analysis_full';dest=d/'summaries';dest.mkdir(parents=True,exist_ok=True)
    def load_csv(path):
        rr=list(csv.DictReader(path.open(encoding='utf-8-sig')))
        for r in rr:
            for k,v in r.items():
                try:r[k]=float(v)
                except (ValueError,TypeError):pass
        return rr
    def group(rr,keys,vals):
        b=collections.defaultdict(list)
        for r in rr:b[tuple(r[k] for k in keys)].append(r)
        return [{**dict(zip(keys,k)),**{v:float(np.mean([r[v] for r in rows])) for v in vals}} for k,rows in sorted(b.items())]
    f=load_csv(d/'fixed_state_paired.csv');fixed=group(f,['t','region'],['COMMON_MSE','USER_MSE'])
    for r in fixed:r.update(ratio=r['USER_MSE']/r['COMMON_MSE'],improved_requests=sum(x['t']==r['t'] and x['region']==r['region'] and x['delta_MSE']<0 for x in f))
    csvwrite(dest/'fixed_by_time.csv',fixed)
    pf=group(f,['parent_id','part','region'],['COMMON_MSE','USER_MSE'])
    for r in pf:r['ratio']=r['USER_MSE']/r['COMMON_MSE']
    csvwrite(dest/'fixed_by_parent.csv',pf)
    a=load_csv(d/'fixed_state_full.csv');p=group(a,['label','t'],[c for c in a[0] if c.endswith('derivative') or c.endswith('split_mse')]);csvwrite(dest/'projection_by_time.csv',p)
    z=load_csv(d/'continuation_start_end.csv')
    for r in z:r['free_area_ratio']=r['free_area']/r['source_free_area']
    vals=['endpoint_reference_mse','all_split_mse','tip_split_mse','bulk_split_mse','edge_ratio_median','edge_below_half_fraction','area_ratio_median','area_below_quarter_fraction','source_normal_flip_fraction','free_area_ratio']
    cont=group(z,['start_t','stage'],vals);csvwrite(dest/'continuation_by_start.csv',cont)
    r=load_csv(d/'response_paired.csv');resp=group(r,['split','t','region'],['COMMON_response','USER_response','response_direction_cosine']);csvwrite(dest/'response_by_time.csv',resp)
    w=load_csv(OUT/'weights/functional/delta_W_h.csv');ww=group(w,['label','module'],['base_output_RMS','delta_output_RMS','delta_over_base']);csvwrite(dest/'functional_by_module.csv',ww)
    fig,ax=plt.subplots(2,3,figsize=(15,8));colors={'COMMON':'#777777','USER':'#166da2'}
    for i,region in enumerate(('all','bulk','tip')):
        part=[r for r in fixed if r['region']==region]
        for label in LABELS:ax[0,i].plot([r['t'] for r in part],[r[label+'_MSE'] for r in part],'o-',label=label,color=colors[label]);ax[0,i].set_yscale('log')
        ax[0,i].set_title('Fixed train state: '+region+' velocity MSE');ax[0,i].set_xlabel('t');ax[0,i].legend()
        for label in LABELS:
            part=[r for r in p if r['label']==label];ax[1,i].plot([r['t'] for r in part],[r[region+'_split_derivative'] for r in part],'o-',label=label,color=colors[label])
        ax[1,i].plot([r['t'] for r in part],[r['truth_'+region+'_split_derivative'] for r in part],'k--',label='paired target velocity');ax[1,i].axhline(0,c='gray',lw=.5);ax[1,i].set_title('Source-copy split derivative: '+region);ax[1,i].set_xlabel('t');ax[1,i].legend()
    fig.suptitle('8 current TRAIN requests; exact same states and transported source identities');fig.tight_layout();fig.savefig(dest/'fixed_and_projection.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(14,4))
    for i,(key,title) in enumerate([('all_split_mse','Same-source corner split MSE'),('edge_ratio_median','Median edge / source edge'),('free_area_ratio','Total free area / source area')]):
        for t in manifest()['continuations']:
            rr={r['stage']:r for r in cont if r['start_t']==t};ax[i].plot([0,1],[rr['START'][key],rr['END'][key]],'o-',label=f'start {t}')
        ax[i].set_xticks([0,1],['START','END']);ax[i].set_title(title);ax[i].legend()
    ax[0].set_yscale('log');fig.suptitle('USER continuation: source-informed states; not generation success');fig.tight_layout();fig.savefig(dest/'continuation.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for i,split in enumerate(('CURRENT_TRAIN','DEV')):
        for label in LABELS:
            for region,ls in [('bulk','-'),('tip','--')]:
                rr=[r for r in resp if r['split']==split and r['region']==region];ax[i].plot([r['t'] for r in rr],[r[label+'_response'] for r in rr],'o'+ls,label=label+' '+region,color=colors[label])
        ax[i].set_title(split);ax[i].set_xlabel('Common rollout state t');ax[i].set_ylabel('RMS velocity difference / RMS C difference');ax[i].legend()
    fig.suptitle('C-only x scaling +/-2%; fixed free states and layout; larger is not better');fig.tight_layout();fig.savefig(dest/'condition_response.png',dpi=160);plt.close(fig)
    for title,rr in [('FIXED',fixed),('CONTINUATION',cont),('RESPONSE',resp),('FUNCTIONAL',ww)]:print(title,json.dumps(rr))
if __name__=='__main__':main()
