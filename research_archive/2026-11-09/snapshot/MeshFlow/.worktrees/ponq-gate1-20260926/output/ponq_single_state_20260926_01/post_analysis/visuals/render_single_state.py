"""Read-only static visualization of a COMPLETE single-state FM diagnostic.

Writes only a new post_analysis/visuals directory with a supplemental manifest.
Preselected displays are sample 0 and, when available, free samples IDs 0..3.
"""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

COMBINATIONS=['PPP','PPO','POP','POO','OPP','OPO','OOP','OOO']
DOMAINS=['fixed','enclosing']
ELEVATION=58
AZIMUTH=-90


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);a=ap.parse_args();root=a.root.resolve()
    sources={}
    def read(path):
        path=Path(path);sources[str(path)]=sha(path);return json.loads(path.read_text())
    execution=read(root/'EXECUTION.json')
    if execution['status']!='COMPLETE':raise RuntimeError('Wait for EXECUTION status COMPLETE; no partial-result rendering')
    original_manifest_hash=sha(root/'MANIFEST.json');sources[str(root/'MANIFEST.json')]=original_manifest_hash
    read(root/'RUN_PROTOCOL.json');read(root/'RUN_PROTOCOL_SHA256.json')
    ledger=read(root/'interventions_64/LEDGER.json');summary=read(root/'interventions_64/SUMMARY.json')
    slots={(r['sample_id'],r['combination'],r['domain']):r for r in ledger}
    expected={(i,c,d) for i in range(32) for c in COMBINATIONS for d in DOMAINS}
    if len(ledger)!=512 or set(slots)!=expected:raise AssertionError('Need all 512 unique primary slots, failures included')
    counts=np.array([[sum(bool(slots[i,c,d]['valid']) for i in range(32)) for c in COMBINATIONS] for d in DOMAINS])
    for j,d in enumerate(DOMAINS):
        for k,c in enumerate(COMBINATIONS):assert counts[j,k]==summary['by_domain'][d][c]['valid']
    ledger256_path=root/'interventions_256/LEDGER.json';slots256={}
    if ledger256_path.exists():
        extra=read(ledger256_path)
        slots256={(r['sample_id'],r['domain']):r for r in extra}
        if len(extra)!=64 or set(slots256)!={(i,d) for i in range(32) for d in DOMAINS}:raise AssertionError('Incomplete 256-step slots')
        assert all(r['combination']=='PPP' for r in extra)
    if (root/'STEP_TEST_DECISION.json').exists():read(root/'STEP_TEST_DECISION.json')
    out=root/'post_analysis/visuals';out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(__file__),out/'render_single_state.py')
    meshes={};requested=[]
    def mesh(path):
        path=Path(path)
        if not path.exists():return None
        sources[str(path)]=sha(path)
        with np.load(path,allow_pickle=False) as z:return z['vertices'].copy(),z['faces'].copy()
    target=mesh(root/'target/canonical_mesh.npz')
    if target is None:raise ValueError('Missing canonical target mesh')
    for d in DOMAINS:
        for c in COMBINATIONS:requested.append((64,0,c,d))
    if slots256:
        for nfe in [64,256]:
            for d in DOMAINS:
                for i in range(4):requested.append((nfe,i,'PPP',d))
    for key in sorted(set(requested)):
        nfe,i,c,d=key;meshes[key]=mesh(root/f'interventions_{nfe}'/f'{i:03d}_{c}_{d}'/'mesh.npz')
    all_vertices=[target[0]]+[m[0] for m in meshes.values() if m is not None and len(m[0])]
    vertices=np.concatenate(all_vertices);lo=vertices.min(axis=0);hi=vertices.max(axis=0);center=(lo+hi)/2
    span=float(np.max(hi-lo))*1.10
    if not np.isfinite(span) or span<=0:raise ValueError('Bad plotting range')
    limits=np.column_stack((center-span/2,center+span/2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'savefig.facecolor':'white'})
    figures=[]
    def save(fig,name):
        path=out/(name+'.png');fig.savefig(path,dpi=150,facecolor='white');plt.close(fig);figures.append(path.name)
    fig,ax=plt.subplots(figsize=(12,3.7));fig.subplots_adjust(left=.11,right=.92,top=.74,bottom=.24)
    plot=ax.imshow(counts,vmin=0,vmax=32,cmap='YlGnBu',aspect='auto')
    ax.set_xticks(range(8),[c+'\n'+('free' if c=='PPP' else 'assisted') for c in COMBINATIONS]);ax.set_yticks([0,1],DOMAINS)
    ax.tick_params(axis='both',length=0,pad=8)
    ax.set_xticks(np.arange(-.5,8,1),minor=True);ax.set_yticks(np.arange(-.5,2,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='minor',bottom=False,left=False)
    for j in range(2):
        for k in range(8):ax.text(k,j,f'{counts[j,k]}/32',ha='center',va='center',color='white' if counts[j,k]>=20 else '#17212b',fontsize=13,fontweight='bold')
    bar=fig.colorbar(plot,ax=ax,fraction=.024,pad=.025);bar.set_ticks([0,8,16,24,32]);bar.set_label('Joint geometry-valid samples')
    fig.suptitle('Single fixed thin-U target | 64-step direct FM | all 32 noise samples',fontsize=14,y=.97)
    fig.text(.5,.855,'Field order: position P / raw normal n / centered factor F.  P = predicted, O = oracle.',ha='center',fontsize=10)
    fig.text(.5,.055,'Only PPP is free generation. Every other column is oracle-assisted; OOO repeats one identity, not 32 independent shapes.\nDomains are reported separately; no union of successes. All failures remain in the denominator.',ha='center',fontsize=9)
    save(fig,'validity_by_field_combination')
    def panel(ax,data,title,row=None,is_target=False):
        ax.set_proj_type('ortho');ax.view_init(elev=ELEVATION,azim=AZIMUTH)
        ax.set_xlim(*limits[0]);ax.set_ylim(*limits[1]);ax.set_zlim(*limits[2]);ax.set_box_aspect((1,1,1));ax.set_axis_off()
        valid=True if is_target else bool(row.get('valid'))
        if data is not None and len(data[1]):
            v,f=data
            poly=Poly3DCollection(v[f],facecolors='#bbc6cd' if is_target else '#71adc5',edgecolors='#244554',linewidths=.19,alpha=1.)
            ax.add_collection3d(poly)
            text=f'{len(v)} vertices | {len(f)} faces'
            if row is not None:text+='\n'+('VALID' if valid else 'INVALID')+'  B='+str(tuple(row.get('beta'+str(k),'?') for k in range(3)))
        else:
            ax.text2D(.5,.5,'NO SAVED MESH\n'+str(row.get('status','missing')),transform=ax.transAxes,ha='center',va='center',fontsize=7,color='#a32222')
            text='Failure retained'
        ax.set_title(title,fontsize=10,pad=-2,color='#17212b')
        ax.text2D(.5,.04,text,transform=ax.transAxes,ha='center',va='bottom',fontsize=7.4,color='#236847' if valid else '#a32222')
    fig=plt.figure(figsize=(22,7));fig.subplots_adjust(left=.004,right=.996,bottom=.105,top=.82,wspace=-.015,hspace=.12)
    for j,d in enumerate(DOMAINS):
        ax=fig.add_subplot(2,9,j*9+1,projection='3d');panel(ax,target,d.upper()+'\nTarget reference',is_target=True)
        for k,c in enumerate(COMBINATIONS):
            ax=fig.add_subplot(2,9,j*9+k+2,projection='3d');panel(ax,meshes[64,0,c,d],c+'\n'+('free generation' if c=='PPP' else 'oracle-assisted'),slots[0,c,d])
    fig.suptitle('Preselected sample 0 | eight field combinations | same camera and world scale in every panel',fontsize=15,y=.975)
    fig.text(.5,.905,'Column code order = position / raw normal / centered F.  P: prediction; O: oracle.  Q is decoded at the chosen position.\nTop: fixed official corner cube [-1,1]^3.  Bottom: enclosing site AABB + frozen 10% margins.',ha='center',fontsize=10)
    fig.text(.5,.026,'No sample selection, vertex repositioning, mesh repair or view rescaling. OOO is an oracle identity check.\nValidity labels come from exact geometry/topology audits; images alone do not certify manifoldness.',ha='center',fontsize=10)
    save(fig,'sample_000_eight_combinations')
    if slots256:
        fig=plt.figure(figsize=(15,12));fig.subplots_adjust(left=.01,right=.99,bottom=.07,top=.90,wspace=-.02,hspace=.08)
        order=[(64,'fixed'),(64,'enclosing'),(256,'fixed'),(256,'enclosing')]
        for j,(nfe,d) in enumerate(order):
            ax=fig.add_subplot(4,5,j*5+1,projection='3d');panel(ax,target,f'{nfe} Euler steps | {d}\nTarget reference',is_target=True)
            for i in range(4):
                row=slots[i,'PPP',d] if nfe==64 else slots256[i,d]
                ax=fig.add_subplot(4,5,j*5+i+2,projection='3d');panel(ax,meshes[nfe,i,'PPP',d],f'Sample {i:03d} | PPP free',row)
        fig.suptitle('Same frozen model and noise | 64 vs 256 steps | preselected sample IDs 0–3',fontsize=14,y=.975)
        fig.text(.5,.935,'All panels share the same camera, orthographic projection and world scale as the eight-combination figure.',ha='center',fontsize=10)
        fig.text(.5,.025,'All displayed samples are fully predicted PPP. The target is shown only as a visual reference.\nNo best-sample selection; this is a development-set diagnostic, not a generalization result.',ha='center',fontsize=10)
        save(fig,'free_samples_000_to_003_64_vs_256')
    report=dict(status='COMPLETE',source_execution_status=execution['status'],primary_slots=512,combination_order=COMBINATIONS,
        domain_order=DOMAINS,valid_counts=counts.tolist(),fixed_sample_id=0,free_comparison_ids=list(range(4)) if slots256 else [],
        step256_available=bool(slots256),plots=figures,
        camera=dict(elevation=ELEVATION,azimuth=AZIMUTH,projection='orthographic',axis_limits=limits.tolist(),shared_across_all_mesh_panels=True),
        source_files=[dict(path=p,sha256=h) for p,h in sorted(sources.items())],
        interpretation='Static scientific illustrations. PPP is generated; all other field combinations are oracle-assisted. Invalid/missing meshes remain labeled failures.',
        original_manifest_unmodified=sha(root/'MANIFEST.json')==original_manifest_hash)
    assert report['original_manifest_unmodified']
    dump(out/'VISUALIZATION.json',report)
    dump(out/'SUPPLEMENTAL_MANIFEST.json',dict(parent_manifest_sha256=original_manifest_hash,
        files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file() and p.name!='SUPPLEMENTAL_MANIFEST.json']))
    print(json.dumps(dict(output=str(out),valid_counts=counts.tolist(),plots=figures,total_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file())),indent=2))
if __name__=='__main__':main()