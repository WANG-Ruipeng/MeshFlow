"""Fixed seed endpoint panels, no geometry modification or selection."""
from pathlib import Path
import argparse
import numpy as np
from .common import read_json,write_json,load_cases,registered_seeds,file_sha256

def run(out,phase='A'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    out=Path(out);seed=registered_seeds(out)[0];cases=load_cases();figdir=out/'figures';figdir.mkdir(exist_ok=True)
    if phase=='A':
        rows=sum([read_json(out/'scores'/('A_'+m)/'results.json')['rows'] for m in ('B','J')],[])
        configs=[(m,pr,n) for pr,n in (('bf16',50),('bf16',100),('fp32',50),('fp32',100)) for m in ('B','J')]
        selected={(r['parent'],r['patch'],r['model'],r['precision'],r['nfe']):r for r in rows if r['seed']==seed}
    else:
        raise ValueError('Only registered A panels implemented')
    records=[]
    for parent in (0,1):
        for patch in range(5):
            entries=[]
            for m,pr,n in configs:
                row=selected[parent,patch,m,pr,n]
                with np.load(out/row['path']) as z:raw=z['raw'].reshape(112,3,3)
                entries.append((row,raw))
            allpoints=np.concatenate([x.reshape(-1,3) for _,x in entries]+[cases[parent,patch]['full'].reshape(-1,3)])[:,[0,2,1]]
            center=(allpoints.min(0)+allpoints.max(0))*.5;radius=float(np.ptp(allpoints,axis=0).max())*.54
            fig=plt.figure(figsize=(24,6),layout='constrained')
            for col,(row,raw) in enumerate(entries):
                tris=raw[:,:,[0,2,1]]
                for part in range(2):
                    ax=fig.add_subplot(2,len(entries),part*len(entries)+col+1,projection='3d')
                    ax.add_collection3d(Poly3DCollection(tris[12:],facecolors='#91a9c2',edgecolors='#30445a',linewidths=.35,alpha=.86))
                    if part==0:ax.add_collection3d(Poly3DCollection(tris[:12],facecolors='#f6aa3d',edgecolors='#a45309',linewidths=.7,alpha=.95))
                    ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)
                    ax.set_box_aspect((1,1,1));ax.view_init(elev=24,azim=-54);ax.set_axis_off()
                    title=f"{row['model']} {row['precision']} NFE{row['nfe']}" if part==0 else 'free only'
                    ax.set_title(title,fontsize=10)
            fig.suptitle(f'P{parent} patch{patch}, fixed seed {seed}; orange C, blue free. Shared bounds/view; no postprocessing.',fontsize=13)
            path=figdir/f'A_P{parent}_patch{patch}_seed{seed}.png'
            if path.exists():raise FileExistsError(path)
            fig.savefig(path,dpi=140);plt.close(fig)
            records.append(dict(parent=parent,patch=patch,seed=seed,path=str(path.relative_to(out)),sha256=file_sha256(path),models_and_configs=configs,bounds_center=center.tolist(),radius=radius))
    write_json(out/'figures/A_manifest.json',dict(status='PASS',records=records,all_ten_contexts_included=True,no_output_selected_for_success=True,model_calls=0,postprocessing=0))
    return records
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--phase',default='A');a=p.parse_args();run(a.out,a.phase)
