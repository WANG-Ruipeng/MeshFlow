"""Deterministic same-field qualitative panels, plus every failed QEM case."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ..exporter import RadialSupport
from ..io_utils import file_hash,new_dir,write_json,array_hash


def render(run,out,seed,ids,tag,dpi=160):
    support=RadialSupport();base=run/f'seed{seed}'
    samples=json.loads((base/'samples/SAMPLES.json').read_text())
    records={r['sample_id']:r for r in samples['records']}
    manifests={arm:json.loads((base/'exports'/f'{arm}_MANIFEST.json').read_text()) for arm in ('fixed','a','qem')}
    rows={arm:{r['sample_id']:r for r in m['records']} for arm,m in manifests.items()}
    fig=plt.figure(figsize=(12,2.9*len(ids)+.8));evidence=[]
    for i,sid in enumerate(ids):
        rec=records[sid]
        with np.load(base/'samples'/rec['path'],allow_pickle=False) as z:nodes=z['nodes'].copy()
        assert array_hash(nodes)==rec['nodes_hash']
        ref_v,ref_f=support.mesh(nodes);center=(ref_v.min(0)+ref_v.max(0))/2;scale=np.ptp(ref_v,axis=0).max()
        item={'seed':seed,'sample_id':sid,'nodes_hash':rec['nodes_hash'],'outlets':{}}
        for j,arm in enumerate(('mainfield','fixed','a','qem')):
            ax=fig.add_subplot(len(ids),4,i*4+j+1,projection='3d')
            if arm=='mainfield':v,f=ref_v,ref_f;status='source, 4096 faces';edge='none';alpha=1.
            else:
                row=rows[arm][sid];path=base/'exports'/row['mesh_path'];assert file_hash(path)==row['mesh_sha256']
                with np.load(path,allow_pickle=False) as z:v=z['vertices'];f=z['faces']
                valid=row['exact']['embedded_endpoint_pass'];status=f'{len(f)} faces; exact '+('pass' if valid else 'FAIL')
                if len(f)!=512:status+='; budget FAIL'
                edge=(0.08,.11,.13,.6);alpha=.95
                item['outlets'][arm]={'mesh_sha256':row['mesh_sha256'],'faces':len(f),'exact_pass':valid}
            triangles=((v-center)/scale)[f]
            normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);length=np.linalg.norm(normals,axis=1)
            normals/=np.maximum(length[:,None],1e-20);light=np.array([.4,-.6,.7]);light/=np.linalg.norm(light)
            shade=.5+.5*np.abs(normals@light)
            color=np.array({'mainfield':[.7,.74,.78],'fixed':[.90,.65,.27],'a':[.3,.64,.87],'qem':[.35,.74,.55]}[arm])
            colors=np.clip(shade[:,None]*color[None,:],0,1)
            poly=Poly3DCollection(triangles,facecolors=colors,edgecolors=edge,linewidths=.20,alpha=alpha)
            ax.add_collection3d(poly);ax.set(xlim=(-.6,.6),ylim=(-.6,.6),zlim=(-.6,.6));ax.set_box_aspect((1,1,1));ax.view_init(elev=18,azim=35);ax.set_axis_off()
            title={'mainfield':'Shared mainfield','fixed':'Fixed','a':'Original A','qem':'VTK QEM'}[arm]
            ax.set_title(f'{title} | seed {seed}, ID {sid}\n{status}',fontsize=8,color='darkred' if 'FAIL' in status else 'black',pad=0)
        evidence.append(item)
    fig.suptitle('Same saved field, common camera and scale; closed-set16 development',fontsize=11,y=.99)
    fig.subplots_adjust(wspace=0,hspace=.08,left=.01,right=.99,top=1-.8/(2.9*len(ids)+.8),bottom=.025)
    filename=f'{tag}_seed{seed}.png';fig.savefig(out/filename,dpi=dpi,facecolor='white');plt.close(fig)
    return {'file':filename,'sha256':file_hash(out/filename),'cases':evidence}


def main(run,preview=False,version=""):
    run=Path(run).resolve();out=new_dir(run/(('qualitative_preview' if preview else 'qualitative')+version));figures=[]
    for seed in (10,11):
        figures.append(render(run,out,seed,['00000','00016','00032','00048'],'fixed_index_gallery',80 if preview else 160))
        m=json.loads((run/f'seed{seed}/exports/qem_MANIFEST.json').read_text())
        failed=[r['sample_id'] for r in m['records'] if r['status']=='ok' and (not r['exact']['embedded_endpoint_pass'] or r['actual_faces']!=512)]
        if failed:figures.append(render(run,out,seed,failed,'all_qem_gate_failures',80 if preview else 160))
    write_json(out/'MANIFEST.json',{'selection':'fixed indices0,16,32,48 for each seed; supplementary panel includes every QEM exact/budget failure',
        'source_sha256':file_hash(__file__),'not_evidence_of_generalization':True,'source_field_common_center_scale_per_row':True,'figures':figures})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--preview',action='store_true');p.add_argument('--version',default='');a=p.parse_args();main(a.run,a.preview,a.version)