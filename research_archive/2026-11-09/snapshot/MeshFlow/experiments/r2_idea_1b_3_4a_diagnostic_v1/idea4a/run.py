"""CPU-only real-data curved FM path validation; no model/framework imports."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,csv,hashlib,time,datetime,importlib.abc
from pathlib import Path
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch','tensorflow','jax'):
            raise RuntimeError('Neural framework forbidden in CPU diagnostic: '+fullname)
sys.meta_path.insert(0,NoModels())
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'experiments/r2_full_seat_learning_curve_v1'
L=float(np.sqrt(3)*2/.3762)
SEED=2026101604
TIMES=[.1,.3,.5,.7,.9]
H=1e-5
SOURCES=['r2/pilots/full_seat_learning_curve_v1/train.py','r2/pilots/full_seat_learning_curve_v1/common.py','r2/pilots/user_data_scaling_32_67_v1/training_data.py','r2/src/meshflow_control/data/dataset.py','r2/src/meshflow_control/data/stream.py','r2/src/meshflow_control/training/coupling.py','r2/src/meshflow_control/training/losses.py','r2/pilots/user_condition_finetune_v1/trainer.py','r2/data_tools/user_conditions_v1/render.py']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def jhash(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf8')
def csvwrite(p,rows):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def prepare():
    if (OUT/'registration.json').exists():raise FileExistsError('Registration already frozen; use run or inspect')
    old=read(BASE/'registration.json');plan=read(BASE/'plans/FULL_SEAT.json');interface=read(old['interface_manifest_path'])
    assert len(old['train_task_ids'])==47 and interface['eligible_parent_count']==47
    assert (old['batch'],old['microbatch'],old['updates_per_arm'])==(8,1,6000)
    records=plan['records'][:4]
    assert len({r['uid'] for r in records})==4
    rng=np.random.default_rng(SEED);paths=[]
    for row in records:
        ax,az=rng.uniform(-.05,.05,2)
        theta=rng.uniform(0,2*np.pi);radius=.02*L*np.sqrt(rng.uniform())
        A=np.diag([ax,0.,az]);d=np.array([radius*np.cos(theta),0.,radius*np.sin(theta)])
        paths.append(dict(sample_index=row['sample_index'],A=A.tolist(),d=d.tolist(),translation_norm=float(np.linalg.norm(d))))
    protected={str(ROOT/s):sha(ROOT/s) for s in SOURCES}
    for p in (BASE/'registration.json',BASE/'plans/FULL_SEAT.json',BASE/'ot/FULL_SEAT/00000.npz',BASE/'ot/FULL_SEAT/00000.json',Path(old['source_data']),Path(old['interface_manifest_path'])):protected[str(p)]=sha(p)
    for row in records:
        m=interface['tasks'][row['task_id']]
        for key in ('parent_npz','condition_source','mask_path'):protected[m[key]]=sha(m[key])
    reg=dict(schema='r2_idea4a_curved_path_cpu_v1',status='FROZEN',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection='First four original FULL_SEAT plan records, before reading geometry or perturbation outcomes',seed=SEED,L=L,times=TIMES,finite_difference_h=H,finite_difference_max_abs_tolerance=1e-7,scale_diagonal_peak_bounds=[-.05,.05],horizontal_axes=[0,2],vertical_axis=1,axis_source='r2/data_tools/user_conditions_v1/render.py:101',translation_vector_norm_peak_bound=.02*L,translation_distribution='uniform disk in model X/Z, radius <= 0.02L; no Y translation',scale_center='world origin; no C-center subtraction',path='Xtilde=(I+g*A)X+g*d, g=4t(1-t); free only',velocity='Utilde=(I+g*A)U+4(1-2t)*(A*X+d)',endpoint_contract='same fixed HYBRID x0 and OT-matched x1; known C stays context at all t',records=records,paths=paths,source_data=old['source_data'],interface_manifest=old['interface_manifest_path'],old_registration_summary=dict(training_parents=47,conditions=47,condition_type='SEAT_FULL',K_min=interface['K_min'],K_max=interface['K_max'],batch=8,microbatch=1,updates=6000,source_global_step=6000,endpoint_global_step=12000,prior='OT first then HYBRID0.25',width_augmentation='per batch four alpha=1, four alpha in [.9,1.1], model X only'),protected_sha256=protected,budget=dict(cpu_workers=1,threads=1,real_paths=4,time_points_per_path=5,new_OT_calls=0,neural_framework_imports=0,weight_loads=0,model_forwards=0,backwards=0,optimizer_updates=0,mesh_generation=0))
    write(OUT/'registration.json',reg)
    print(json.dumps(dict(status='REGISTERED',real_paths=4,t_points=20,registration_sha256=sha(OUT/'registration.json'))))
def execute():
    if (OUT/'receipt.json').exists():raise FileExistsError('Completed evidence immutable')
    start=time.perf_counter();reg=read(OUT/'registration.json');assert reg['status']=='FROZEN'
    assert all(sha(p)==h for p,h in reg['protected_sha256'].items())
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'r2/src'))
    from r2.pilots.user_data_scaling_32_67_v1 import training_data as olddata
    from meshflow_control.data.io import bits
    def no_ot(*a,**kw):raise RuntimeError('No new OT authorized')
    olddata.coupling.nested_ot_map=no_ot
    data=olddata.Dataset(reg['source_data'])
    with np.load(BASE/'ot/FULL_SEAT/00000.npz',allow_pickle=False) as z:maps={k:z[k].copy() for k in z.files}
    assert sha(BASE/'ot/FULL_SEAT/00000.npz')==read(BASE/'ot/FULL_SEAT/00000.json')['sha256']
    builder=olddata.Builder(data,maps);interface=read(reg['interface_manifest']);rows=[];checks=[]
    for record,pathspec in zip(reg['records'],reg['paths']):
        sample=olddata.validate_sample(builder.sample(record),record)
        assert sample['ot_cache_status']=='HIT'
        n=int(sample['y']);known=sample['known_mask'];free=sample['valid_mask']&~known
        x0=sample['x0'].reshape(n,3,3).astype(np.float64);y=sample['x1'].reshape(n,3,3).astype(np.float64);u=y-x0
        C=sample['context'].reshape(n,3,3).astype(np.float64)
        A=np.asarray(pathspec['A']);d=np.asarray(pathspec['d']);I=np.eye(3)
        assert np.all(np.abs(np.diag(A))<=.05) and np.linalg.norm(d)<=.02*L and A[1].sum()==0 and d[1]==0
        def state(t):
            g=4*t*(1-t);x=(1-t)*x0+t*y;out=x.copy()
            out[free]=x[free]@(I+g*A).T+g*d;out[known]=C[known]
            return out
        def velocity(t):
            g=4*t*(1-t);gp=4*(1-2*t);x=(1-t)*x0+t*y;out=np.zeros_like(u)
            out[free]=u[free]@(I+g*A).T+gp*(x[free]@A.T+d)
            return out
        meta=interface['tasks'][record['task_id']]
        with np.load(meta['mask_path'],allow_pickle=False) as z:orig_mask=z[meta['mask_key']].copy()
        ids=sample['source_face_ids'];corners=sample['corner_permutations'];mapped_mask=orig_mask[ids][np.arange(n)[:,None],corners]
        assert not mapped_mask[known].any() and mapped_mask.sum()==orig_mask.sum()
        case=data.task(record['task_id'],record['alpha'])
        full=case['full_target'].reshape(n,3,3);expected=full[ids][np.arange(n)[:,None],corners]
        assert bits(expected[free].reshape(-1,9),sample['x1'][free]) and bits(expected[known].reshape(-1,9),sample['context'][known])
        endpoint0=state(0);endpoint1=state(1)
        assert np.array_equal(endpoint0[free],x0[free]) and np.array_equal(endpoint1[free],y[free])
        check=dict(sample_index=record['sample_index'],task_id=record['task_id'],N=n,K=int(known.sum()),alpha=record['alpha'],OT_cache='HIT',endpoint0_exact=True,endpoint1_exact=True,source_corner_transport_exact=True,interface_transport_exact=True,source_interface_corners=int(orig_mask.sum()),known_identity_exact=True,mask_changed=False,free_denominator=9*int(free.sum()),scale_x=float(A[0,0]),scale_z=float(A[2,2]),translation_norm_pct_L=float(np.linalg.norm(d)/L*100),original_u_fp32_rounding_max=float(np.max(np.abs(sample['u'].reshape(n,3,3)-u))))
        snapshots={}
        for t in reg['times']:
            xt=state(t);vt=velocity(t);xfd=(state(t+H)-state(t-H))/(2*H)
            fdmax=float(np.max(np.abs(xfd-vt)));assert fdmax<reg['finite_difference_max_abs_tolerance']
            base=(1-t)*x0+t*y;base[known]=C[known]
            assert np.array_equal(xt[known],C[known]) and np.array_equal(vt[known],np.zeros_like(vt[known]))
            assert np.array_equal(xt[:,:,1],base[:,:,1]) and np.array_equal(vt[free,:,1],u[free,:,1])
            g=4*t*(1-t);gp=4*(1-2*t);S=I+g*A
            xback=(xt[free]-g*d)@np.linalg.inv(S).T
            uback=(vt[free]-gp*(xback@A.T+d))@np.linalg.inv(S).T
            ycorrect=xback+(1-t)*uback
            ynaive=xt[free]+(1-t)*vt[free]
            corrected_error=float(np.max(np.abs(ycorrect-y[free])));assert corrected_error<1e-12
            # Fixed residual (not a model output) tests endpoint-error energy identity.
            residual=np.broadcast_to(np.array([.1,.0,-.1]),vt[free].shape)
            rcorrect=residual@np.linalg.inv(S).T*(1-t)
            naive_error=float(np.mean((residual*(1-t))**2));correct_error=float(np.mean(rcorrect**2))
            delta=(xt-base)[free];udiff=(vt-u)[free]
            row=dict(sample_index=record['sample_index'],task_id=record['task_id'],t=t,N=n,K=int(known.sum()),alpha=record['alpha'],delta_state_max_pct_L=float(np.linalg.norm(delta,axis=-1).max()/L*100),delta_state_rms=float(np.sqrt(np.mean(delta**2))),unchanged_u_wrong_velocity_rms=float(np.sqrt(np.mean(udiff**2))),unchanged_u_wrong_velocity_max=float(np.abs(udiff).max()),correct_target_energy=float(np.mean(vt[free]**2)),original_target_energy=float(np.mean(u[free]**2)),energy_ratio=float(np.mean(vt[free]**2)/np.mean(u[free]**2)),finite_difference_max_abs=fdmax,finite_difference_rms=float(np.sqrt(np.mean((xfd-vt)**2))),perfect_velocity_naive_endpoint_max_pct_L=float(np.linalg.norm(ynaive-y[free],axis=-1).max()/L*100),perfect_velocity_corrected_endpoint_max_abs=corrected_error,fixed_residual_old_endpoint_mse=naive_error,fixed_residual_correct_endpoint_mse=correct_error,known_C_exact=True,vertical_unchanged=True)
            rows.append(row);snapshots[f'xt_t{int(t*10)}']=xt.astype(np.float32);snapshots[f'u_t{int(t*10)}']=vt.astype(np.float32)
        # A coordinate-affine path must commute with all retained face/corner identities.
        source_order=np.empty_like(y);source_order[ids[:,None],corners]=y
        changed_source=source_order@(I+A).T+d;changed_order=changed_source[ids][np.arange(n)[:,None],corners]
        assert np.array_equal(changed_order,y@(I+A).T+d)
        check['affine_commutes_with_face_corner_permutation']=True
        # Padding is excluded by the same validity mask; no nonzero state/target there.
        padded_state=np.pad(state(.5),((0,7),(0,0),(0,0)));padded_velocity=np.pad(velocity(.5),((0,7),(0,0),(0,0)))
        assert not padded_state[n:].any() and not padded_velocity[n:].any()
        check['synthetic_padding_zero_preserved']=True
        checks.append(check)
        np.savez_compressed(OUT/f'path_{record["sample_index"]:02d}.npz',x0=sample['x0'],x1=sample['x1'],context=sample['context'],known_mask=known,valid_mask=sample['valid_mask'],source_face_ids=ids,corner_permutations=corners,interface_corner_mask=mapped_mask,A=A,d=d,**snapshots)
    assert builder.new_maps=={} and 'torch' not in sys.modules
    assert all(sha(p)==h for p,h in reg['protected_sha256'].items())
    csvwrite(OUT/'path_diagnostics.csv',rows);csvwrite(OUT/'data_contract_checks.csv',checks)
    write(OUT/'source_hashes.json',reg['protected_sha256'])
    receipt=dict(status='PASS',schema=reg['schema'],registration_sha256=sha(OUT/'registration.json'),paths=len(checks),time_points=len(rows),actual_OT_calls=0,OT_cache_hits=len(checks),weight_loads=0,model_forwards=0,backwards=0,optimizer_updates=0,mesh_generation=0,torch_imported=False,threads=1,protected_files_unchanged=True,wall_seconds=time.perf_counter()-start,finite_difference_max_abs=max(r['finite_difference_max_abs'] for r in rows),energy_ratio_range=[min(r['energy_ratio'] for r in rows),max(r['energy_ratio'] for r in rows)],max_state_displacement_pct_L=max(r['delta_state_max_pct_L'] for r in rows),wrong_old_target_rms_range=[min(r['unchanged_u_wrong_velocity_rms'] for r in rows),max(r['unchanged_u_wrong_velocity_rms'] for r in rows)],perfect_velocity_naive_endpoint_max_pct_L=max(r['perfect_velocity_naive_endpoint_max_pct_L'] for r in rows),corrected_endpoint_max_abs=max(r['perfect_velocity_corrected_endpoint_max_abs'] for r in rows),run_script_sha256=sha(__file__))
    write(OUT/'receipt.json',receipt);print(json.dumps(receipt,indent=2))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);a=p.parse_args()
    prepare() if a.stage=='prepare' else execute()