"""Plot completed Exposure training summaries without model/scorer calls."""
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
    if final['status']!='PASS' or final['updates']!=3000:raise ValueError('Completed3000 training audit required')
    source=folder/'loss_bins.json';rows=C.read(source)['rows']
    if len(rows)!=60 or rows[0]['first']!=1 or rows[-1]['last']!=3000:raise ValueError('Fixed50-update bins differ')
    csv_path=folder/'loss_curve.csv';png=folder/'loss_curve.png'
    if csv_path.exists() or png.exists():raise RuntimeError('Unsealed plot outputs; explicit engineering review required')
    buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with csv_path.open('x',encoding='utf-8',newline='') as handle:handle.write(buffer.getvalue())
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9.5,5),layout='constrained')
    steps=[(r['first']+r['last'])/2 for r in rows]
    ax.plot(steps,[r['mean_batch_objective'] for r in rows],label='Actual weighted objective',color='#a64335',linewidth=1.8)
    ax.plot(steps,[r['mean_batch_FM'] for r in rows],label='Unweighted FM (SSE / actual batch coordinates)',color='#2878a0',linewidth=1.6)
    ax.set(xlabel='Additional Native optimizer updates',ylabel='50-update arithmetic mean',
           title='Exposure +3000: actual objective and unweighted FM')
    ax.grid(alpha=.22);ax.legend(fontsize=9)
    fig.text(.5,-.025,'Same recorded training forwards. Fitting curves do not rank final generation quality.',ha='center',fontsize=9)
    try:
        image=io.BytesIO();fig.savefig(image,format='png',dpi=160,bbox_inches='tight')
        with png.open('xb') as handle:handle.write(image.getvalue())
    finally:plt.close(fig)
    result=dict(status='COMPLETE',updates=3000,bins=60,source=str(source),source_sha256=C.sha(source),
        code_sha256=C.sha(__file__),model_calls=0,scoring_calls=0,new_training_forwards=0,
        files={str(p):C.sha(p) for p in (csv_path,png)},
        meaning='Arithmetic mean of50 effective-batch losses. Weighted objective and unweighted FM are separate recorded quantities.')
    C.write_new(done,result);return result


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
