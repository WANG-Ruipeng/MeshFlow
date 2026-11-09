"""CPU-only, unchanged independent scoring for the fixed solver pilot."""
from __future__ import annotations
from pathlib import Path
import argparse,csv,time
import numpy as np
from experiments.mf_h1_local0.jedge5_diagnosis.common import (
    ROOT,load_cases,read_json,write_json,file_sha256,EVALUATION_SHA,ASSETS,array_hash)
from experiments.mf_h1_local0.jedge5_diagnosis import scoring as archived

OLD=ROOT/'native_t1/runs/jedge5_diagnosis_20261002_102215'
DEFAULT=ROOT/'native_t1/runs/jedge5_solver_pilot_20261002_121000'
CASE_SPECS=((0,10401),(0,10402),(4,10401),(4,10402))
METHODS={'E_quad50':(50,50),'Midpoint25':(25,50),'Heun25':(25,50),'Heun50':(50,100)}
DISPLAY=('Euler50','E_quad50','Midpoint25','Heun25','Euler100','Heun50')
J_FILE_SHA='cc7a248fa9a8570059f225cd1c313f21f08585925407a79418df6d63c9cae0e9'
KEYS=('remainder_symmetric_RMS_bbox_pct','free_to_R12_RMS_bbox_pct','R12_to_free_RMS_bbox_pct',
      'boundary_RMS_bbox_pct','boundary_p95_bbox_pct','boundary_coverage_fraction',
      'FF_illegal_pair_count','CF_proper_pair_count','CF_proper_affected_free_fraction',
      'free_minimum_angle_lt5_fraction','free_q_gt10_fraction','free_total_surface_area')


def own_out(value):
    out=Path(value).resolve()
    if out.parent!=ROOT/'native_t1/runs' or not out.name.startswith('jedge5_solver_pilot_'):
        raise ValueError('Only this isolated pilot run is writable')
    return out


def initialize(out):
    import fcntl
    folder=out/'registration';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'scoring_init.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        source=OLD/'registration/static_context_audit.json';cache=folder/'static_context_audit.json'
        if not cache.exists():cache.write_bytes(source.read_bytes())
        if file_sha256(cache)!=file_sha256(source):raise ValueError('Archived CC cache changed')
        banks=archived.banks_with_context_audit(out)
        record=dict(status='PASS',CC_source=str(source),CC_sha256=file_sha256(source),new_CC_audits=0,
            scorer_sha256=file_sha256(archived.REFERENCE/'evaluation.py'),
            original_scorer_module_sha256=file_sha256(Path(archived.__file__)),
            evaluation_inputs_sha256=EVALUATION_SHA,model_forwards=0,
            CPU_scheduling='One thread per process; initialization lock only. No cross-run global CPU lock was found or claimed.')
        if not (folder/'scoring_identity.json').exists():write_json(folder/'scoring_identity.json',record)
    return banks


def score_cli(out,case_id):
    import torch
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():raise RuntimeError('CPU scoring must not initialize CUDA')
    out=own_out(out);case_id=int(case_id)
    patch,seed=CASE_SPECS[case_id];source=out/f'raw/case{case_id}/results.json'
    generated=read_json(source)
    if generated['status']!='PASS':raise ValueError('Complete successful worker required')
    rows=generated['rows']
    if len(rows)!=4 or {x['method'] for x in rows}!=set(METHODS):raise ValueError('Require exactly four registered methods')
    folder=out/f'scores/case{case_id}';folder.mkdir(parents=True,exist_ok=True)
    dest=folder/'results.json'
    if dest.exists():raise FileExistsError(dest)
    banks=initialize(out);case=load_cases()[0,patch];tick=time.perf_counter();ledger={};saved=[]
    result=dict(status='RUNNING',case_id=case_id,parent=0,patch=patch,seed=seed,rows=saved,ledger=ledger,
        model_forwards=0,optimizer_updates=0,OT_calls=0,repairs=0,source_sha256=file_sha256(source))
    write_json(dest,result)
    try:
        for method in METHODS:
            row=next(x for x in rows if x['method']==method);steps,nfe=METHODS[method]
            for k,want in dict(parent=0,patch=patch,seed=seed,precision='bf16',model='J',nfe=nfe,
                model_state_sha256=ASSETS['J']['state'],model_file_sha256=J_FILE_SHA).items():
                if row[k]!=want:raise ValueError(f'{method} identity mismatch {k}')
            path=Path(row['path']);path=path if path.is_absolute() else out/path;path=path.resolve()
            if out not in path.parents or file_sha256(path)!=row['sha256']:raise ValueError('Raw path/hash mismatch')
            with np.load(path,allow_pickle=False) as z:
                raw=z['raw'].copy().reshape(112,3,3);states=z['states'].copy().reshape(-1,112,3,3);C=z['C'].copy().reshape(12,3,3)
            if raw.dtype!=np.float32 or states.dtype!=np.float32 or C.dtype!=np.float32:raise ValueError('Non-FP32 geometry')
            if len(states)!=steps+1 or not np.isfinite(states).all():raise ValueError('Incomplete/nonfinite states')
            if states[-1].tobytes()!=raw.tobytes() or C.tobytes()!=case['C'].tobytes():raise ValueError('Endpoint/context identity mismatch')
            if any(s[:12].tobytes()!=C.tobytes() for s in states):raise ValueError('C changed at a state')
            old_noise_path=OLD/f'A/J/J_P0_patch{patch}_seed{seed}_bf16_Euler50.npz'
            with np.load(old_noise_path,allow_pickle=False) as old:initial=old['states'][0].reshape(112,3,3)
            if states[0].tobytes()!=initial.tobytes():raise ValueError('Initial C/free Gaussian differs from paired baseline')
            before=raw.tobytes();ts=time.perf_counter();metrics,scalars=archived.score_one(raw,banks[patch],0,patch,ledger)
            if raw.tobytes()!=before:raise ValueError('Offline evaluator mutated geometry')
            detail=folder/(method+'.json');write_json(detail,metrics)
            saved.append(dict(row,case_id=case_id,steps=steps,scalars=scalars,
                cross_remainder=metrics['remainder']['cross'],metric_path=str(detail.relative_to(out)),metric_sha256=file_sha256(detail),
                scoring_seconds=time.perf_counter()-ts,C_saved_states_bitwise=True,topology='NOT_RUN'))
            result['completed']=len(saved);result['seconds']=time.perf_counter()-tick;write_json(dest,result)
            print('score',case_id,method,'PASS',flush=True)
        if torch.cuda.is_initialized():raise RuntimeError('CPU score initialized CUDA')
        result['status']='PASS'
    except BaseException as exc:
        result['status']='FAIL';result['error']=repr(exc);raise
    finally:
        result['seconds']=time.perf_counter()-tick;write_json(dest,result)
    return result


def baseline_rows():
    source=OLD/'scores/A_J/results.json';record=read_json(source)
    state=read_json(OLD/'A/J/results.json')
    if state['status']!='PASS' or state['state_after']!=ASSETS['J']['state']:raise ValueError('Baseline model identity mismatch')
    rows=[]
    for cid,(patch,seed) in enumerate(CASE_SPECS):
        for nfe in (50,100):
            r=next(x for x in record['rows'] if x['parent']==0 and x['patch']==patch and x['seed']==seed and x['nfe']==nfe and x['precision']=='bf16')
            path=Path(r['path']);path=path if path.is_absolute() else OLD/path
            if file_sha256(path)!=r['sha256']:raise ValueError('Reused baseline raw hash changed')
            detail=OLD/r['score_path'];metrics=read_json(detail)
            rows.append(dict(model='J',method=f'Euler{nfe}',case_id=cid,parent=0,patch=patch,seed=seed,
                nfe=nfe,steps=nfe,precision='bf16',model_state_sha256=ASSETS['J']['state'],model_file_sha256=J_FILE_SHA,
                path=str(path),sha256=r['sha256'],scalars=r['scalars'],metric_path=str(detail),metric_sha256=file_sha256(detail),
                baseline_scores_source_sha256=file_sha256(source),reused=True,topology='NOT_RUN',cross_remainder=metrics['remainder']['cross']))
    return rows


def write_csv(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def aggregate(out):
    out=own_out(out);rows=baseline_rows();new=0;ledger={};timing=[]
    for cid in range(4):
        r=read_json(out/f'scores/case{cid}/results.json')
        if r['status']!='PASS' or len(r['rows'])!=4:raise ValueError('Complete fixed score matrix required')
        rows.extend(r['rows']);new+=len(r['rows']);timing.append(dict(case_id=cid,seconds=r['seconds']))
        for k,v in r['ledger'].items():ledger[k]=ledger.get(k,0)+v
    rows.sort(key=lambda r:(r['case_id'],DISPLAY.index(r['method'])))
    allnames=sorted(rows[0]['scalars']);by={(r['case_id'],r['method']):r for r in rows};comparisons=[]
    for cid in range(4):
        for candidate,baseline in (('E_quad50','Euler50'),('Midpoint25','Euler50'),('Heun25','Euler50'),('Heun50','Euler100')):
            a,b=by[cid,candidate],by[cid,baseline]
            if a['nfe']!=b['nfe']:raise AssertionError('Unequal-NFE comparison')
            changes={k:(a['scalars'][k]-b['scalars'][k]) if a['scalars'].get(k) is not None and b['scalars'].get(k) is not None else None for k in allnames}
            comparisons.append(dict(case_id=cid,parent=0,patch=a['patch'],seed=a['seed'],candidate=candidate,baseline=baseline,nfe=a['nfe'],differences=changes))
    macros={}
    for group,cids in (('normal_patch0',[0,1]),('sensitive_patch4',[2,3]),('all_selected',[0,1,2,3])):
        macros[group]={}
        for candidate in METHODS:
            rr=[r for r in comparisons if r['case_id'] in cids and r['candidate']==candidate]
            macros[group][candidate]={k:dict(mean_delta=float(np.mean([x['differences'][k] for x in rr])),
                median_paired_delta=float(np.median([x['differences'][k] for x in rr])),
                negative_count=sum(x['differences'][k]<0 for x in rr),positive_count=sum(x['differences'][k]>0 for x in rr),count=len(rr))
                for k in allnames if all(x['differences'][k] is not None and isinstance(x['differences'][k],(int,float)) for x in rr)}
    folder=out/'evaluation';folder.mkdir(exist_ok=True)
    write_csv(folder/'per_output.csv',[dict(case_id=r['case_id'],parent=0,patch=r['patch'],seed=r['seed'],method=r['method'],nfe=r['nfe'],**r['scalars']) for r in rows])
    write_csv(folder/'paired_equal_NFE.csv',[dict(**{k:v for k,v in r.items() if k!='differences'},**r['differences']) for r in comparisons])
    result=dict(status='PASS',rows=rows,comparisons=comparisons,macro_differences=macros,ledger=ledger,new_scored_outputs=new,
        reused_baselines=8,model_forwards=0,optimizer_updates=0,scoring_worker_seconds=timing,
        evaluation_inputs_sha256=EVALUATION_SHA,scoring_version_sha256=file_sha256(archived.REFERENCE/'evaluation.py'),
        caveat='All cases are preselected observed contexts/noises on P0. Equal-NFE comparisons only. Heun50 uses 100 NFE, not 50. Differences are descriptive, not convergence or generalization proof. No topology certification or repair.')
    write_json(folder/'results.json',result);plot_outputs(out,rows);print('aggregate PASS',flush=True)
    return result


def plot_outputs(out,rows):
    """Show GT coverage at fixed original query positions; never hide absent regions."""
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from experiments.mf_h1_local0.jedge5_blackbox.geometry import attribution
    folder=out/'evaluation';cases=load_cases();raws={};coverage={};ledger=[]
    # Small CPU-only directional-query export; original score formula is independently reconciled.
    for r in rows:
        path=Path(r['path']);path=path if path.is_absolute() else out/path
        if file_sha256(path)!=r['sha256']:raise ValueError('Raw changed before plotting')
        with np.load(path,allow_pickle=False) as z:raw=z['raw'].copy().reshape(112,3,3)
        arrays,summary=attribution(raw,cases[0,r['patch']]);key=(r['case_id'],r['method']);raws[key]=raw;coverage[key]=arrays
        if not np.isclose(summary['symmetric_RMS_bbox_pct'],r['scalars']['remainder_symmetric_RMS_bbox_pct'],atol=1e-12,rtol=1e-12):raise ValueError('Plot sample/score reconciliation failed')
        np.savez_compressed(folder/f"case{r['case_id']}_{r['method']}_directional_queries.npz",**arrays)
        ledger.append(dict(case_id=r['case_id'],method=r['method'],nfe=r['nfe'],precision=r['precision'],raw_sha256=r['sha256'],
            model_state_sha256=r['model_state_sha256'],C_sha256=array_hash(cases[0,r['patch']]['C']),
            target_points_sha256=array_hash(arrays['target_points']),GT_to_free_RMS_bbox_pct=summary['backward_RMS_bbox_pct']))
    pts=np.concatenate([v.reshape(-1,3) for v in raws.values()]+[cases[0,0]['full'].reshape(-1,3)])
    center=(pts.max(0)+pts.min(0))/2;radius=np.ptp(pts,axis=0).max()*.54
    vmax=max(float(a['target_squared_distances'].max()/cases[0,CASE_SPECS[cid][0]]['bbox']**2*1e4) for (cid,m),a in coverage.items())
    norm=PowerNorm(.35,vmin=0,vmax=vmax)
    for cid,(patch,seed) in enumerate(CASE_SPECS):
        fig=plt.figure(figsize=(21,11));case=cases[0,patch];L=case['bbox']
        for col,method in enumerate(DISPLAY):
            row=next(r for r in rows if r['case_id']==cid and r['method']==method);tri=raws[cid,method];a=coverage[cid,method]
            for view in range(3):
                ax=fig.add_subplot(3,6,view*6+col+1,projection='3d')
                if view<2:
                    ax.add_collection3d(Poly3DCollection(tri[12:],facecolors=(.70,.72,.77,.95),edgecolors=(.2,.2,.2,.3),linewidths=.25))
                    if view==0:ax.add_collection3d(Poly3DCollection(tri[:12],facecolors=(.2,.7,.9,.85),edgecolors='teal',linewidths=.4))
                else:
                    target=case['free'].reshape(100,3,3);q=a['target_points'];d2=a['target_squared_distances']/L**2*1e4
                    ax.add_collection3d(Poly3DCollection(target,facecolors=(.7,.7,.7,.04),edgecolors=(.3,.3,.3,.12),linewidths=.3))
                    ax.scatter(q[:,0],q[:,1],q[:,2],c=d2,cmap='viridis',norm=norm,s=8,depthshade=False,edgecolors='none')
                    ax.add_collection3d(Poly3DCollection(case['C'],facecolors=(.2,.7,.9,.1),edgecolors='cyan',linewidths=.3))
                ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
                ax.set_box_aspect((1,1,1));ax.view_init(elev=20,azim=-55,vertical_axis='y');ax.set_axis_off()
                if view==0:ax.set_title(f"{method} ({row['nfe']} NFE)\nRMS {row['scalars']['remainder_symmetric_RMS_bbox_pct']:.3f}%",fontsize=10)
                if col==0:ax.text2D(-.05,.5,('Full','Free only','GT coverage')[view],transform=ax.transAxes,rotation=90,va='center',fontsize=11)
        fig.suptitle(f'JEdge5 / BF16 / P0 patch {patch} / seed {seed}: same noise, fixed view and ranges',fontsize=14)
        fig.subplots_adjust(left=.025,right=.94,top=.93,bottom=.02,wspace=0,hspace=.02)
        bar=fig.add_axes([.955,.1,.01,.25]);fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),cax=bar,label='GT coverage point distance^2 [(% bbox)^2]')
        fig.savefig(folder/f'case{cid}_comparison.png',dpi=160);plt.close(fig)
    write_json(folder/'plot_manifest.json',dict(status='PASS',rows=ledger,extra_model_forwards=0,
        directional_surface_query_reconstructions=len(rows),points_each_direction=512,
        sampler='Exact original RandomState(9911), independently reconciled to all 24 original scores',
        camera=dict(elevation=20,azimuth=-55,vertical_axis='y',center=center.tolist(),radius=float(radius)),
        coverage_color=dict(transform='PowerNorm gamma=.35',minimum=0,maximum=vmax,units='squared percent bbox; pointwise, weight 1/512 in RMS score'),
        note='Full/free shapes and GT missing-region heatmaps; no GT correspondence enters generation. Fixed shared range across all four cases.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--case',type=int,choices=range(4));p.add_argument('--aggregate',action='store_true');a=p.parse_args()
    if a.aggregate:aggregate(a.out)
    elif a.case is not None:score_cli(a.out,a.case)
    else:p.error('--case or --aggregate required')

if __name__=='__main__':main()
