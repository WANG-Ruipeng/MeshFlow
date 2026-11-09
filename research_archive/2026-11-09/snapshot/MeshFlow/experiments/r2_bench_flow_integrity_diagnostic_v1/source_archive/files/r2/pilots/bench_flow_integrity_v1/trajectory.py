"""Frozen CPU diagnostics of stored Euler trajectories; never loads a model."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse,collections,datetime,hashlib,io,json,math,statistics,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_flow_integrity_diagnostic_v1'
INV=OUT/'audits/trajectory_inventory'
DEST=OUT/'trajectory'
PHASES={'FULL':(0,50),'EARLY':(0,15),'MID':(15,35),'LATE':(35,50)}
ROLES=('ALL_FREE','BULK_FREE','SEAM_TIP')
METRICS=('S','orthogonal_energy','parallel_speed_variance','parallel_global_offset_energy','path_chord_ratio','path_length_normalized','chord_length_normalized','B_velocity_variation_proxy')
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def immutable(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    raw=data if isinstance(data,bytes) else (json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode()
    if p.exists():assert p.read_bytes()==raw,str(p)
    else:p.write_bytes(raw)
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def csvwrite(p,rows):
    import csv
    text=io.StringIO(newline='');keys=list(dict.fromkeys(k for row in rows for k in row));w=csv.DictWriter(text,fieldnames=keys,lineterminator='\n');w.writeheader()
    w.writerows({k:('NA' if v is None else json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v) for k,v in r.items()} for r in rows)
    immutable(p,('\ufeff'+text.getvalue()).encode())
def num(x):
    if isinstance(x,bool):return None
    try:v=float(x)
    except (TypeError,ValueError):return None
    return v if math.isfinite(v) else None
def avg(v):
    v=[x for x in v if x is not None];return statistics.fmean(v) if v else None
def grouped(rows,keys):
    d=collections.defaultdict(list)
    for r in rows:d[tuple(r[k] for k in keys)].append(r)
    return d
def resources():
    def cmd(args):
        try:return subprocess.run(args,text=True,capture_output=True,timeout=10).stdout.strip()
        except Exception as e:return repr(e)
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cpu_logical=os.cpu_count(),cpu_threads=1,
        meminfo=Path('/proc/meminfo').read_text() if Path('/proc/meminfo').exists() else 'NA',
        loadavg=Path('/proc/loadavg').read_text() if Path('/proc/loadavg').exists() else 'NA',
        gpu=cmd(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader']),
        gpu_processes=cmd(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader']),
        CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],model_calls=0,scoring_calls=0)

def freeze():
    protocol=dict(schema='STORED_EULER_FREE_TRAJECTORY_V1',status='FROZEN_BEFORE_COMPUTATION',trajectories=108,
        scope='all current84 plus same-panel curriculum COMMON24; no historical COMMON/USER mixed into current comparison',
        input_inventory_sha256=sha(INV/'per_trajectory.json'),source_code_sha256=sha(__file__),
        dt=.02,nodes=51,L='per-trajectory prior frozen L (9.208138264587333), not parent bounding box',
        roles=ROLES,phases=PHASES,phase_interpretation='15/20/15 Euler increments; phase S contributions sum FULL and use the same global net displacement',
        coordinates='Canonical model/world frame. Fixed slot/corner identities. No temporal OT alignment.',
        velocity='v_k=(X[k+1]-X[k])/.02 in float64 from stored FP32 states. Realized discrete velocity, not recovered BF16 forward.',
        S='sum_k dt ||v_k-(X50-X0)||^2 / (free_coordinate_count * L^2)',
        orthogonal='Unit direction d=(X50-X0)/||X50-X0|| in the full selected state space; sum dt ||v-(v.dot(d))*d||^2/(count*L^2)',
        parallel_speed_variance='Within phase alpha=v.dot(d); sum dt (alpha-mean_phase(alpha))^2/(count*L^2)',
        parallel_global_offset_energy='duration*(mean_phase(alpha)-||X50-X0||)^2/(count*L^2); FULL nearly zero. S=orthogonal+parallel_variance+offset.',
        path_chord_ratio='sum ||X[k+1]-X[k]|| / ||X[end]-X[start]|| in full selected coordinate state; monotone speed change along one line leaves ratio1. Phase ratio uses phase chord.',
        normalization='S/energy divide count*L^2; path and chord divide sqrt(count)*L. Zero-mask or zero-chord is explicit NA.',
        B_velocity_variation_proxy='.5*dt*sum_k RMS(v[k+1]-v[k])/L over all free scalar coordinates; descriptive screening proxy, not a rigorous error upper bound.',
        risk_gate=dict(parent_prefixes=['a8e987','6b894c'],noise_index=0,labels=['NATIVE_S6000','COVERAGE_FINAL'],rows=4,threshold=.005,rule='Any one B>.005 triggers exactly the registered four Euler100 comparisons; no dispatch by this CPU script.',registered_by_root='OUT/PROTOCOL.md before current readings'),
        pairs=[dict(reference='NATIVE_S6000',candidate='COVERAGE_FINAL',requests=30,mask='identical'),dict(reference='COMMON_S12000',candidate='NATIVE_S6000',requests=24,mask='identical'),dict(reference='NATIVE_S6000',candidate='SEAM_S3000',requests=24,mask='different; compare per-model normalized scalars descriptively, no tensor distance')],
        conclusion_boundary='Curvature/speed variation alone does not prove fragmentation, OT failure or numerical error.',
        aggregate='average noises within condition then conditions within parent, then equal parent means',
        resources=resources(),model_calls=0,new_noise=0,new_OT=0)
    immutable(DEST/'definition.json',protocol);return protocol

def measures(state,mask,L,start=0,end=50):
    m=int(mask.sum())*3
    if not m:return dict(status='NA_EMPTY_ROLE',free_corner_count=0,free_coordinate_count=0,**dict.fromkeys(METRICS))
    x=state[:,mask,:].astype(np.float64).reshape(51,m);dv=np.diff(x,axis=0);v=dv/.02;delta=x[-1]-x[0];v=v[start:end];dv=dv[start:end]
    den=m*L*L;duration=(end-start)*.02;net=np.linalg.norm(delta);localnet=np.linalg.norm(x[end]-x[start]);path=np.linalg.norm(dv,axis=1).sum()
    result=dict(status='PASS',free_corner_count=int(mask.sum()),free_coordinate_count=m,S=float(.02*np.sum((v-delta)**2)/den),
        path_length_normalized=float(path/math.sqrt(m)/L),chord_length_normalized=float(localnet/math.sqrt(m)/L),
        path_chord_ratio=float(path/localnet) if localnet>1e-15 else None,
        B_velocity_variation_proxy=float(.5*.02*np.sqrt(np.mean(np.diff(v,axis=0)**2,axis=1)).sum()/L))
    if net<=1e-15:
        result.update(status='NA_ZERO_GLOBAL_CHORD_DIRECTION',orthogonal_energy=None,parallel_speed_variance=None,parallel_global_offset_energy=None)
    else:
        d=delta/net;alpha=v@d;perp=v-alpha[:,None]*d
        result.update(orthogonal_energy=float(.02*np.sum(perp**2)/den),parallel_speed_variance=float(.02*np.sum((alpha-alpha.mean())**2)/den),parallel_global_offset_energy=float(duration*(alpha.mean()-net)**2/den))
        assert math.isclose(result['S'],sum(result[k] for k in ('orthogonal_energy','parallel_speed_variance','parallel_global_offset_energy')),rel_tol=1e-9,abs_tol=1e-13)
    return result

def analytic_checks():
    t=np.arange(51,dtype=np.float64)/50;mask=np.ones((1,1),bool)
    straight=np.zeros((51,1,1,3));straight[:,0,0,0]=t
    a=measures(straight,mask,1);assert a['S']<1e-28 and abs(a['path_chord_ratio']-1)<1e-14
    speed=straight.copy();speed[:,0,0,0]=t*t;b=measures(speed,mask,1)
    assert b['S']>0 and b['orthogonal_energy']<1e-28 and abs(b['path_chord_ratio']-1)<1e-14
    bent=straight.copy();bent[:,0,0,1]=.2*np.sin(np.pi*t);c=measures(bent,mask,1)
    assert c['orthogonal_energy']>0 and c['path_chord_ratio']>1
    zero=measures(straight,np.zeros((1,1),bool),1);assert zero['status']=='NA_EMPTY_ROLE'
    return dict(status='PASS',uniform_straight=a,monotone_nonuniform_speed=b,bent=c,empty_role=zero)

def aggregate(rows,keys,metrics):
    out=[]
    for identity,gr in sorted(grouped(rows,keys).items()):
        r=dict(zip(keys,identity));r['rows']=len(gr)
        for m in metrics:
            vals=[num(x.get(m)) for x in gr];r[m]=avg(vals);r[m+'__valid_n']=sum(x is not None for x in vals)
        out.append(r)
    return out

def run():
    tick=time.perf_counter();definition=read(DEST/'definition.json')
    assert definition['source_code_sha256']==sha(__file__)
    assert definition['input_inventory_sha256']==sha(INV/'per_trajectory.json')
    entries=[r for r in read(INV/'per_trajectory.json') if r['cohort']!='OLD_COMMON_USER'];assert len(entries)==108
    checks=analytic_checks();rows=[]
    for e in entries:
        assert sha(e['trajectory_path'])==e['trajectory_sha256'] and sha(e['layout_path'])==e['layout_sha256']
        state=arrays(e['trajectory_path'])[e['trajectory_array_key']];lay=arrays(e['layout_path']);free=lay['anchor_ids']<0
        assert state.shape==(51,e['N'],3,3) and state.dtype==np.float32 and np.isfinite(state).all()
        masks={'ALL_FREE':free,'BULK_FREE':free&np.isin(np.arange(e['N']),lay['bulk_slots'])[:,None],
               'SEAM_TIP':free&np.isin(np.arange(e['N']),lay['seam_slots'])[:,None]}
        for role,mask in masks.items():
            part=[]
            for phase,(start,end) in PHASES.items():
                r={k:e[k] for k in ('label','parent_id','condition_id','request_id','eval_group','noise_index','N','K','L','trajectory_path','trajectory_sha256','layout_path','layout_sha256','noise_sha256')}
                r.update(role=role,phase=phase,start_step=start,end_step=end,**measures(state,mask,e['L'],start,end));rows.append(r);part.append(r)
            if mask.any():assert math.isclose(part[0]['S'],sum(r['S'] for r in part[1:]),rel_tol=1e-12,abs_tol=1e-13)
    allfull=[r for r in rows if r['role']=='ALL_FREE' and r['phase']=='FULL'];assert len(allfull)==108
    pairs=[]
    for p in definition['pairs']:
        by={(r['label'],r['request_id'],r['role'],r['phase']):r for r in rows}
        ref=[r for r in rows if r['label']==p['reference']]
        for a in ref:
            b=by.get((p['candidate'],a['request_id'],a['role'],a['phase']))
            if not b:continue
            same=a['layout_sha256']==b['layout_sha256']
            assert a['noise_sha256']==b['noise_sha256'] and a['N']==b['N'] and a['K']==b['K']
            if p['mask']=='identical':assert same
            row=dict(reference=p['reference'],candidate=p['candidate'],**{k:a[k] for k in ('request_id','condition_id','parent_id','eval_group','noise_index','role','phase')},identical_layout=same,comparison_scope=p['mask'],reference_free_coordinate_count=a['free_coordinate_count'],candidate_free_coordinate_count=b['free_coordinate_count'])
            for m in METRICS:
                av,bv=a[m],b[m];row[m+'__reference']=av;row[m+'__candidate']=bv;row[m+'__delta']=None if av is None or bv is None else bv-av
            pairs.append(row)
    keys=('label','eval_group','role','phase')
    conditions=aggregate(rows,keys+('parent_id','condition_id'),METRICS)
    parents=aggregate(conditions,keys+('parent_id',),METRICS);macro=aggregate(parents,keys,METRICS)
    pkeys=('reference','candidate','eval_group','role','phase');delta_metrics=tuple(m+'__delta' for m in METRICS)
    pc=aggregate(pairs,pkeys+('parent_id','condition_id'),delta_metrics);pp=aggregate(pc,pkeys+('parent_id',),delta_metrics);pm=aggregate(pp,pkeys,delta_metrics)
    gate_rows=[r for r in allfull if r['parent_id'][:6] in ('a8e987','6b894c') and int(r['noise_index'])==0 and r['label']in ('NATIVE_S6000','COVERAGE_FINAL')]
    assert len(gate_rows)==4
    gate=dict(status='TRIGGERED' if any(r['B_velocity_variation_proxy']>.005 for r in gate_rows) else 'NOT_TRIGGERED',threshold=.005,
        rule=definition['risk_gate'],rows=[{k:r[k] for k in ('label','request_id','B_velocity_variation_proxy','S','path_chord_ratio','free_coordinate_count')} for r in gate_rows],
        interpretation='Screening heuristic only; not a proven integration error bound and not an OT diagnosis.',new_model_dispatch_by_this_script=False)
    tables={'per_output_all_roles_phases.csv':rows,'per_output_allfree_full.csv':allfull,'per_condition.csv':conditions,'per_parent.csv':parents,'parent_macro.csv':macro,
        'paired_per_output.csv':pairs,'paired_per_condition.csv':pc,'paired_per_parent.csv':pp,'paired_parent_macro.csv':pm}
    for name,data in tables.items():csvwrite(DEST/name,data)
    immutable(DEST/'analytic_checks.json',checks);immutable(DEST/'risk_gate.json',gate)
    plots(rows,pairs,macro)
    receipt=dict(status='COMPLETE',trajectories=108,role_phase_rows=len(rows),per_output_pairs=len(pairs),model_calls=0,scoring_calls=0,new_noise=0,new_OT=0,
        source_code_sha256=sha(__file__),definition_sha256=sha(DEST/'definition.json'),risk_gate=gate['status'],seconds=time.perf_counter()-tick,
        source_trajectories={r['trajectory_path']:r['trajectory_sha256'] for r in entries},source_layouts={r['layout_path']:r['layout_sha256'] for r in entries},
        files={p.name:sha(p) for p in DEST.iterdir() if p.suffix in ('.csv','.png','.svg','.json')},resources_finished=resources())
    assert 'torch' not in sys.modules;immutable(DEST/'completion.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('source_trajectories','source_layouts','files','resources_finished')},indent=2));print(json.dumps(gate,indent=2))

def plots(rows,pairs,macro):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'CURRENT_TRAIN':'#3579a6','DEV':'#ce6c47','NEW_TRAIN':'#568c35','NEW_PARENT_HOLDOUT':'#9867ad'}
    selected=[p for p in pairs if p['reference']=='NATIVE_S6000' and p['candidate']=='COVERAGE_FINAL' and p['role']=='ALL_FREE' and p['phase']=='FULL']
    fig,axs=plt.subplots(2,2,figsize=(10,9),constrained_layout=True)
    for ax,m in zip(axs.flat,('S','path_chord_ratio','orthogonal_energy','parallel_speed_variance')):
        for group,color in colors.items():
            rr=[p for p in selected if p['eval_group']==group];ax.scatter([p[m+'__reference'] for p in rr],[p[m+'__candidate'] for p in rr],label=group,color=color,s=30,alpha=.8)
        lo=min(ax.get_xlim()[0],ax.get_ylim()[0]);hi=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([lo,hi],[lo,hi],color='#999999',lw=1,ls='--');ax.set_xlabel('Native6000');ax.set_ylabel('Coverage1500');ax.set_title(m);ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8);fig.suptitle('Same C / N / Gaussian / free mask, 30 fixed requests; stored Euler50')
    fig.savefig(DEST/'native_coverage_paired.png',dpi=160);fig.savefig(DEST/'native_coverage_paired.svg');plt.close(fig)
    labels=('COMMON_S12000','NATIVE_S6000','COVERAGE_FINAL','SEAM_S3000');fig,axs=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for ax,(group,metric) in zip(axs.flat,[(g,m) for g in ('CURRENT_TRAIN','DEV') for m in ('S','path_chord_ratio','orthogonal_energy')]):
        rr=[next(r for r in macro if r['label']==l and r['eval_group']==group and r['role']=='ALL_FREE' and r['phase']=='FULL') for l in labels]
        ax.bar(range(4),[r[metric] for r in rr],color=['#888888','#3579a6','#568c35','#9867ad']);ax.set_xticks(range(4),['COMMON','NATIVE','COVERAGE','SEAM'],rotation=25);ax.set_title(group+' / '+metric);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Parent-equal, own-free-mask normalized; SEAM mask differs (descriptive)')
    fig.savefig(DEST/'same_old_panel_models.png',dpi=160);fig.savefig(DEST/'same_old_panel_models.svg');plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('--run',action='store_true');a=ap.parse_args()
    if a.freeze:print(json.dumps(freeze(),ensure_ascii=False,indent=2))
    elif a.run:run()
    else:ap.error('--freeze or --run required')
