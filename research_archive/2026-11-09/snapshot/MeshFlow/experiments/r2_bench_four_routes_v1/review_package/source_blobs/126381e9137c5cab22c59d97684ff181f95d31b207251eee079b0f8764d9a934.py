"""Plot completed SEAM fitting curves; never compare their scale to Native."""
from __future__ import annotations
import os
os.environ['MPLBACKEND']='Agg'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import csv,io,json
from . import common as C


def run():
    folder=C.OUT/'closeout';done=folder/'loss_plot_complete.json'
    if done.exists():
        result=C.read(done)
        for path,h in result['files'].items():
            if C.sha(path)!=h:raise ValueError('Changed loss plot artifact')
        return result
    final=C.read(folder/'training.json')
    if final['status']!='PASS' or final['local_updates']!=1500:raise ValueError('Completed1500 training audit required')
    source=folder/'loss_bins.json';rows=C.read(source)['rows']
    if len(rows)!=30 or rows[0]['first_step']!=1 or rows[-1]['last_step']!=1500:raise ValueError('Fixed50-update bins differ')
    csv_path=folder/'loss_curve.csv';png=folder/'loss_curve.png'
    if csv_path.exists() or png.exists():raise RuntimeError('Unsealed plot outputs; explicit engineering review required')
    buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with csv_path.open('x',encoding='utf-8',newline='') as handle:handle.write(buffer.getvalue())
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    steps=[(r['first_step']+r['last_step'])/2 for r in rows]
    for ax,kind,title in zip(axes,('velocity','endpoint'),('Velocity error','One-step endpoint error')):
        for group,name,color in (('all_free','Whole free','#333333'),('bulk','Bulk','#2878a0'),('seam_tip','SEAM third points','#ca7526')):
            values=[r[group+'_coordinate_weighted_'+kind+'_MSE'] for r in rows]
            ax.plot(steps,values,label=name,color=color,linewidth=1.6)
        ax.set(xlabel='Additional SEAM optimizer updates',ylabel='SSE / applicable coordinates in50-update bin',title=title)
        ax.grid(alpha=.22);ax.legend(fontsize=9)
    fig.suptitle('Original SEAM1500: whole / bulk / third-point fitting',fontsize=13)
    fig.text(.5,-.025,'Endpoint errors reuse the training interpolation, not free rollout. Different masks prevent absolute comparison to Native loss.',ha='center',fontsize=9)
    try:
        image=io.BytesIO();fig.savefig(image,format='png',dpi=160,bbox_inches='tight')
        with png.open('xb') as handle:handle.write(image.getvalue())
    finally:plt.close(fig)
    result=dict(status='COMPLETE',updates=1500,bins=30,source=str(source),source_sha256=C.sha(source),
        code_sha256=C.sha(__file__),model_calls=0,scoring_calls=0,new_training_forwards=0,
        files={str(p):C.sha(p) for p in (csv_path,png)},
        meaning='Each series is sum recorded SSE / sum applicable scalar coordinates within50 updates; not a Native loss comparison or generation-quality score.')
    C.write_new(done,result);return result


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
