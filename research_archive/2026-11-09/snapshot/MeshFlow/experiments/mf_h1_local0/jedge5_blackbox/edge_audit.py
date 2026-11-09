"""CPU-only Edge mechanism audit. No model, OT, training, or repair calls.

Teacher: original fixed GT-source interface correspondence and original FP32
endpoint/loss. Generated trajectories: a separate FP64 nearest-Gamma probe;
its changing argmin is not the training D_edge correspondence or an ODE force.
"""
from pathlib import Path
import argparse,csv,hashlib,json,os,time
import numpy as np
import torch
from native_t1.losses.edge import prepare_edge_target,edge_loss,EPSILON,S_VALUES,QUADRATURE_WEIGHTS

ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'native_t1/runs/jedge5_diagnosis_20261002_102215'
EVAL=ROOT/'native_t1/runs/edge_factorial_20261001_123000/registration/evaluation_inputs.npz'
POLICY=dict(teacher_scalar_atol=2e-9,teacher_scalar_rtol=2e-5,
    fixture_atol=1e-12,fixture_rtol=1e-10,finite_difference_step=1e-6,
    finite_difference_atol=3e-10,finite_difference_rtol=1e-7,
    proxy_dtype='CPU_FP64 from saved FP32 states',proxy_normalization='one division by fixed evaluation bbox L',
    proxy_tie_distance_squared_normalized_atol=1e-12,
    projection_switch_parameter_atol=1e-12,
    proxy_candidates='physical C boundary Gamma only, source registration cross-checked with C exact-coordinate edge incidence==1',
    proxy='posthoc proximity only; not D_edge, no assigned GT free slots, no guidance force or causal stiffness claim')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def counts():return dict(model_constructions=0,model_forwards=0,model_backwards=0,OT_calls=0,
    optimizer_updates=0,repairs=0,public_target_preparations=0,public_edge_forward_calls=0,
    geometry_autograd_calls=0,finite_difference_loss_evaluations=0,
    corresponding_projection_direction_calls=0,nearest_boundary_state_calls=0)


def _csv(path,rows):
    with Path(path).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def _projection(points,segments):
    """Same FP32 arithmetic as public helper; adds observations only."""
    midpoint=(segments[:,0]+segments[:,1])*.5;direction=segments[:,1]-segments[:,0]
    length_sq=direction.square().sum(-1)
    raw=((points-midpoint[:,None])*direction[:,None]).sum(-1)/length_sq.clamp_min(EPSILON)[:,None]
    clipped=raw.clamp(-.5,.5);closest=midpoint[:,None]+clipped[...,None]*direction[:,None]
    d2=(points-closest).square().sum(-1)
    branch=torch.where(raw<-.5,-1,torch.where(raw>.5,1,0))
    return dict(distance_squared=d2,projection_raw=raw,projection_clamped=clipped,clamp_branch=branch,
        segment_length_squared=length_sq,closest=closest,
        exactly_on_projection_boundary=(raw==-.5)|(raw==.5))


def corresponding_details(pred,target,ledger):
    free=pred/target['bbox_L'];ix=target['free_faces'];corners=target['free_corners']
    predicted=free[ix[:,None],corners];known=target['GT_endpoints'];length=target['GT_lengths']
    s=pred.new_tensor(S_VALUES)[None,:,None];w=pred.new_tensor(QUADRATURE_WEIGHTS)[None]
    gtq=(1-s)*known[:,:1]+s*known[:,1:];pq=(1-s)*predicted[:,:1]+s*predicted[:,1:]
    ledger['corresponding_projection_direction_calls']+=2
    g=_projection(gtq,predicted);p=_projection(pq,known)
    gd=(g['distance_squared']*w).sum(-1);pd=(p['distance_squared']*w).sum(-1)
    gl=(length*gd).sum()/length.sum();pl=(length*pd).sum()/length.sum();loss=.5*(gl+pl)
    # Endpoint contribution uses actual fixed quadrature mass; no renormalization.
    weight=.5*length[:,None]/length.sum()*w
    gc=(weight*g['distance_squared']);pc=(weight*p['distance_squared'])
    total=float((gc+pc).double().sum());end=float((gc[:,[0,4]]+pc[:,[0,4]]).double().sum())
    return dict(GT_to_pred=g,pred_to_GT=p,loss=loss,GT_to_pred_loss=gl,pred_to_GT_loss=pl,
        GT_to_pred_point_contribution=gc,pred_to_GT_point_contribution=pc,
        endpoint_contribution_sum=end,all_point_contribution_sum=total,
        endpoint_contribution_fraction=None if total==0 else end/total,
        predicted_segments=predicted,GT_segments=known,GT_lengths=length,
        free_faces=ix,free_corners=corners)


def run_fixtures(out):
    out=Path(out)/'edge';out.mkdir(parents=True,exist_ok=True)
    dest=out/'fixture_record.json'
    if dest.exists():raise FileExistsError(dest)
    tick=time.perf_counter();ledger=counts();torch.set_num_threads(1)
    gt=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]],dtype=torch.float64)
    known=torch.tensor([[[1.,0.,0.],[0.,0.,0.],[0.,-1.,0.]]],dtype=torch.float64)
    target=prepare_edge_target(gt,known,[[0,1,2]],[[1,0,3]],1.);ledger['public_target_preparations']+=1
    historical=read(ROOT/'native_t1/runs/edge_factorial_20261001_123000/quadrature_diagnostic.json')
    hcase=next(x for x in historical['cases'] if x['name']=='collinear_delta_0.01')
    rows=[]
    for name,delta,direction in [('collinear',.01,[1.,0.,0.]),('orthogonal',.01,[0.,0.,1.]),('oracle',0.,[1.,0.,0.])]:
        d=torch.tensor(delta,dtype=torch.float64,requires_grad=True);axis=gt.new_tensor(direction)
        def value(v):
            ledger['public_edge_forward_calls']+=1
            return edge_loss(gt+v*axis,target)['loss']
        loss=value(d);ledger['geometry_autograd_calls']+=1
        grad=torch.autograd.grad(loss,d)[0]
        h=POLICY['finite_difference_step'];ledger['finite_difference_loss_evaluations']+=2
        fd=(float(value(delta+h))-float(value(delta-h)))/(2*h)
        expected=delta**2/8 if name!='orthogonal' else delta**2
        derivative=delta/4 if name!='orthogonal' else 2*delta
        continuous=abs(delta)**3/3 if name!='orthogonal' else delta**2
        cg=delta*abs(delta) if name!='orthogonal' else 2*delta
        assert np.isclose(float(loss),expected,atol=1e-12,rtol=1e-10)
        assert np.isclose(float(grad),derivative,atol=1e-12,rtol=1e-10)
        assert np.isclose(float(grad),fd,atol=3e-10,rtol=1e-7)
        rows.append(dict(name=name,delta=delta,ell=1.,bbox_L=1.,loss=float(loss),parameter_derivative=float(grad),
            finite_difference=fd,analytic_continuous_loss=continuous,analytic_continuous_derivative=cg,
            loss_ratio=None if continuous==0 else float(loss)/continuous,
            derivative_ratio=None if cg==0 else float(grad)/cg))
    assert np.isclose(rows[0]['loss'],hcase['quadrature']['5']['loss'],rtol=1e-12,atol=1e-15)
    result=dict(status='PASS',policy=POLICY,fixtures=rows,counts=ledger,CUDA_initialized=torch.cuda.is_initialized(),
        historical_source=dict(path=str(ROOT/'native_t1/runs/edge_factorial_20261001_123000/quadrature_diagnostic.json'),
            case='collinear_delta_0.01',analytic=hcase['analytic']),
        formulas=dict(collinear_validity='Equal length ell parallel collinear segments shifted by 0<=delta<ell/4; no GT multi-edge weighting change',
            continuous='D=delta^3/(3*ell*L^2)',five_point='D5=delta^2/(8*L^2)',loss_ratio='3*ell/(8*delta)',
            derivative_ratio='ell/(4*delta)',at_delta_001='loss ratio37.5; derivative ratio25',
            limit='The network sampler evaluates v_theta, not the derivative of this objective. Ratios do not estimate its Jacobian, stiffness, or numerical stability bound.'),
        source_sha256=sha(__file__),public_edge_sha256=sha(ROOT/'native_t1/losses/edge.py'),seconds=time.perf_counter()-tick)
    write(dest,result);return result


def _eval_arrays():
    reg=read(OLD/'registration/protocol.json')['inputs']['evaluation']
    if sha(EVAL)!=reg['sha256']:raise ValueError('Evaluation source changed')
    with np.load(EVAL,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files if k!='registration_json'}


def run_teacher(out):
    out=Path(out)/'edge';out.mkdir(parents=True,exist_ok=True)
    if (out/'teacher_summary.json').exists():raise FileExistsError(out/'teacher_summary.json')
    tick=time.perf_counter();ledger=counts();torch.set_num_threads(1)
    bank_meta=read(OLD/'stage_c/teacher_bank.json');vmeta=read(OLD/'stage_c/teacher/J_bf16.json')
    bankpath=OLD/'stage_c/teacher_bank.npz';vpath=OLD/'stage_c/teacher/J_bf16.npz'
    if sha(bankpath)!=bank_meta['artifact']['sha256'] or sha(vpath)!=vmeta['arrays']['sha256']:raise ValueError('Stored teacher assets changed')
    with np.load(bankpath,allow_pickle=False) as z:bank={k:z[k].copy() for k in z.files}
    with np.load(vpath,allow_pickle=False) as z:vel={k:z[k].copy() for k in z.files}
    ea=_eval_arrays();targets={};rows=[];points=[];arrays={}
    vrows={r['state_id']:r for r in vmeta['rows']}
    for state in bank_meta['states']:
        pair=bank_meta['pairs'][state['pair_index']]
        if pair['parent_index']!=0 or pair['patch_index'] not in (0,4):continue
        sample={k:bank[v] for k,v in pair['arrays'].items()};known=sample['known_mask'];valid=sample['valid_mask'];free=valid&~known
        ids=sample['source_face_ids'];corners=sample['corner_permutations'];F=ea['parent0_source_face_vertex_ids']
        vids=np.take_along_axis(F[ids],corners,axis=1)
        full=ea['parent0_target'].reshape(112,3,3);actual=full[ids][np.arange(112)[:,None],corners]
        gt=sample['x1'].reshape(112,3,3);C=sample['context'].reshape(112,3,3)
        if actual[free].tobytes()!=gt[free].tobytes() or actual[known].tobytes()!=C[known].tobytes():raise ValueError('Teacher trusted IDs do not map to target/C')
        L=float(np.linalg.norm(np.ptp(full.reshape(-1,3),axis=0)))
        if pair['pair_id'] not in targets:
            targets[pair['pair_id']]=prepare_edge_target(torch.from_numpy(gt[free]),torch.from_numpy(C[known]),vids[free],vids[known],L)
            ledger['public_target_preparations']+=1
        target=targets[pair['pair_id']];ref=vrows[state['state_id']];i=state['index'];prefix=f'state{i:03d}'
        xt=torch.from_numpy(vel[ref['actual_Xt_array']]);v=torch.from_numpy(vel[ref['velocity_array']]);t=torch.from_numpy(bank[prefix+'_t'])
        if xt[0].numpy().tobytes()!=bank[prefix+'_xt'].tobytes():raise ValueError('Saved actual teacher Xt differs')
        pred=(xt+(1-t)*v)[0,free].reshape(-1,3,3)
        ledger['public_edge_forward_calls']+=1
        loss=edge_loss(pred,target)['loss'];details=corresponding_details(pred,target,ledger)
        old=ref['losses']['Edge'];computed=float(loss);observed=float(details['loss'])
        if not np.isclose(computed,old,atol=2e-9,rtol=2e-5) or not np.isclose(observed,computed,atol=2e-9,rtol=2e-5):
            raise AssertionError(('Teacher Edge reconciliation failed',state['state_id'],old,computed,observed))
        summary=dict(state_id=state['state_id'],parent=0,patch=pair['patch_index'],noise_index=pair['noise_index'],
            actual_FP32_t=float(t),training_gate_active=bool(t>=.5),raw_Edge=old,recomputed_Edge=computed,
            recorded_absolute_error=abs(computed-old),observational_reconstruction_error=abs(observed-computed),
            interface_edges=len(target['free_faces']),bbox_L=L,endpoint_contribution_fraction=details['endpoint_contribution_fraction'],
            endpoint_contribution_sum=details['endpoint_contribution_sum'],all_point_contribution_sum=details['all_point_contribution_sum'])
        rows.append(summary)
        for direction in ('GT_to_pred','pred_to_GT'):
            detail=details[direction];d2=detail['distance_squared'].numpy();pr=detail['projection_raw'].numpy();branch=detail['clamp_branch'].numpy()
            for name,value in detail.items():arrays[prefix+'_'+direction+'_'+name]=value.detach().numpy()
            arrays[prefix+'_'+direction+'_point_contribution']=details[direction+'_point_contribution'].numpy()
            for e in range(len(d2)):
                for q,s in enumerate(S_VALUES):
                    points.append(dict(state_id=state['state_id'],patch=pair['patch_index'],noise_index=pair['noise_index'],t=float(t),
                        direction=direction,interface_index=e,free_face=int(target['free_faces'][e]),
                        free_corner0=int(target['free_corners'][e,0]),free_corner1=int(target['free_corners'][e,1]),
                        s=s,quadrature_weight=QUADRATURE_WEIGHTS[q],distance_normalized=float(np.sqrt(d2[e,q])),
                        projection_raw=float(pr[e,q]),projection_clamped=float(detail['projection_clamped'][e,q]),
                        clamp_branch=int(branch[e,q]),point_contribution=float(details[direction+'_point_contribution'][e,q])))
        for name in ('predicted_segments','GT_segments','GT_lengths','free_faces','free_corners'):arrays[prefix+'_'+name]=details[name].numpy()
    if len(rows)!=20:raise AssertionError('Expected 2patch x 2noise x 5t teacher states')
    np.savez_compressed(out/'teacher_projection_details.npz',**arrays);_csv(out/'teacher_states.csv',rows);_csv(out/'teacher_points.csv',points)
    result=dict(status='PASS',policy=POLICY,counts=ledger,states=len(rows),point_rows=len(points),rows=rows,
        original_bank=dict(path=str(bankpath),sha256=sha(bankpath)),original_velocities=dict(path=str(vpath),sha256=sha(vpath)),
        max_recorded_scalar_absolute_error=max(r['recorded_absolute_error'] for r in rows),
        max_observational_reconstruction_error=max(r['observational_reconstruction_error'] for r in rows),
        interpretation='Fixed teacher correspondence from GT source IDs. Below .5 the unweighted geometry is measured only; training multiplier is zero. No generated trajectory is assigned these GT slots.',
        geometry_autograd_calls=0,CUDA_initialized=torch.cuda.is_initialized(),seconds=time.perf_counter()-tick,source_sha256=sha(__file__))
    write(out/'teacher_summary.json',result);return result


def _canonical(segment):
    x=np.ascontiguousarray(segment,dtype=np.float32)
    order=sorted(range(2),key=lambda i:tuple(float(a) for a in x[i]))
    return np.ascontiguousarray(x[order]).tobytes()


def boundary_candidates(C,registered):
    edges={}
    for face in np.asarray(C,np.float32):
        for a,b in ((0,1),(1,2),(2,0)):
            key=_canonical(face[[a,b]]);edges[key]=edges.get(key,0)+1
    expected={k for k,v in edges.items() if v==1};got=[_canonical(x) for x in registered]
    if len(got)!=len(set(got)) or set(got)!=expected:raise ValueError('Registered Gamma differs from C own exact-coordinate boundary')
    # Canonical key order makes all recorded choices independent of input segment ordering.
    return np.stack([np.frombuffer(k,np.float32).reshape(2,3) for k in sorted(expected)]),dict(
        boundary_segments=len(expected),C_unique_edges=len(edges),C_nonmanifold_edge_incidence_gt2=sum(v>2 for v in edges.values()),
        registered_boundary_exact_match=True)


def nearest_projection(points,segments):
    """FP64 all point x finite Gamma segments. Coordinates already divided by L."""
    mid=(segments[:,0]+segments[:,1])*.5;direction=segments[:,1]-segments[:,0]
    ll=(direction*direction).sum(-1)
    raw=((points[:,None]-mid[None])*direction[None]).sum(-1)/np.maximum(ll,EPSILON)[None]
    clip=np.clip(raw,-.5,.5);closest=mid[None]+clip[...,None]*direction[None]
    d2=((points[:,None]-closest)**2).sum(-1)
    order=np.argsort(d2,axis=1,kind='stable');winner=order[:,0];index=np.arange(len(points))
    best=d2[index,winner];second=d2[index,order[:,1]] if len(segments)>1 else np.full(len(points),np.inf)
    chosen=raw[index,winner];margin=second-best
    return dict(winner=winner,projection_raw=chosen,projection_clamped=np.clip(chosen,-.5,.5),
        clamp_branch=np.where(chosen<-.5,-1,np.where(chosen>.5,1,0)).astype(np.int8),
        distance=np.sqrt(best),margin_squared=margin,tie=margin<=POLICY['proxy_tie_distance_squared_normalized_atol'],
        projection_boundary=np.abs(np.abs(chosen)-.5)<=POLICY['projection_switch_parameter_atol'])


def trajectory_probe(path,out,case,ledger):
    tick=time.perf_counter();path=Path(path)
    with np.load(path,allow_pickle=False) as z:states=z['states'].copy();savedC=z['C'].copy() if 'C' in z else case['C'].copy()
    if states.ndim==4 and states.shape[1]==1:states=states[:,0]
    if states.dtype!=np.float32 or states.shape[1:]!=(112,9) or not np.isfinite(states).all():raise ValueError('Expected finite saved FP32 [T,112,9]')
    C=np.asarray(savedC,np.float32).reshape(12,3,3)
    if C.tobytes()!=case['C'].tobytes():raise ValueError('Trajectory C differs from registered C')
    if not np.array_equal(states[:,:12].view(np.uint32),np.broadcast_to(C.reshape(1,12,9),(len(states),12,9)).view(np.uint32)):raise ValueError('Saved C clamp failed')
    segments,boundary_audit=boundary_candidates(C,case['boundary_segments']);segments64=segments.astype(np.float64)/case['L']
    s=np.asarray(S_VALUES)[None,None,:,None];details={};frames=[]
    for k,state in enumerate(states):
        free=state[12:].reshape(100,3,3).astype(np.float64)/case['L']
        edges=free[:,[[0,1],[1,2],[2,0]],:]
        points=(1-s)*edges[:,:,0,None,:]+s*edges[:,:,1,None,:]
        ledger['nearest_boundary_state_calls']+=1
        d=nearest_projection(points.reshape(-1,3),segments64)
        for name,value in d.items():details.setdefault(name,[]).append(value.reshape(100,3,5))
        frames.append(dict(k=k,t=k/(len(states)-1),point_count=1500,tie_count=int(d['tie'].sum()),
            projection_boundary_count=int(d['projection_boundary'].sum()),left_clamped=int((d['clamp_branch']==-1).sum()),
            right_clamped=int((d['clamp_branch']==1).sum()),interior=int((d['clamp_branch']==0).sum()),
            RMS_normalized=float(np.sqrt(np.mean(d['distance']**2))),mean_margin_squared_normalized=float(np.mean(d['margin_squared']))))
    arrays={k:np.stack(v) for k,v in details.items()};winner=arrays['winner'];ties=arrays['tie']
    switch=winner[1:]!=winner[:-1];aba=(winner[2:]==winner[:-2])&(winner[1:-1]!=winner[:-2]);clean=~(ties[2:]|ties[1:-1]|ties[:-2])
    arrays.update(Gamma_segments_model_coordinates=segments,consecutive_winner_change=switch,ABA=aba,ABA_without_any_tie=aba&clean)
    name=path.stem;np.savez_compressed(Path(out)/(name+'_probe.npz'),**arrays);_csv(Path(out)/(name+'_frames.csv'),frames)
    nfe=len(states)-1
    return dict(name=name,path=str(path),input_sha256=sha(path),nfe=nfe,states=len(states),queries_per_state=1500,
        boundary_audit=boundary_audit,known_all_states_bitwise=True,changes=int(switch.sum()),
        change_fraction=float(switch.mean()),ABA=int(aba.sum()),ABA_without_any_tie=int((aba&clean).sum()),
        all_query_states=len(states)*1500,tied_query_states=int(ties.sum()),tie_fraction=float(ties.mean()),
        projection_branch_changes=int((arrays['clamp_branch'][1:]!=arrays['clamp_branch'][:-1]).sum()),
        final_RMS_normalized=frames[-1]['RMS_normalized'],seconds=time.perf_counter()-tick,
        counts=dict(model_forwards=0,OT_calls=0,geometry_autograd_calls=0,optimizer_updates=0))


def run_trajectories(out):
    out=Path(out);dest=out/'edge';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'trajectory_summary.json').exists():raise FileExistsError(dest/'trajectory_summary.json')
    # No partial scan: all sixteen registered trajectories must already exist.
    planned=[(model,patch,seed,nfe,out/'trajectories'/model/f'{model}_P0_patch{patch}_seed{seed}_bf16_Euler{nfe}.npz')
             for model in ('B','J') for patch in (0,4) for seed in (10401,10402) for nfe in (50,100)]
    if any(not p.is_file() for *_,p in planned):raise FileNotFoundError('All16 registered raw trajectories are required before probe')
    ea=_eval_arrays();L=float(np.linalg.norm(np.ptp(ea['parent0_target'].reshape(-1,3),axis=0)))
    ledger=counts();rows=[];tick=time.perf_counter()
    for model,patch,seed,nfe,path in planned:
        case=dict(C=ea[f'parent0_patch{patch}_constraints'],boundary_segments=ea[f'parent0_patch{patch}_boundary_segments'],L=L)
        r=trajectory_probe(path,dest,case,ledger)
        if r['nfe']!=nfe:raise ValueError('Saved trajectory state count differs from planned NFE')
        rows.append(dict(model=model,parent=0,patch=patch,seed=seed,**r))
    result=dict(status='PASS',policy=POLICY,counts=ledger,trajectories=len(rows),rows=rows,
        units='Distances divided by fixed parent bbox L; margins are squared normalized distances',
        interpretation='Posthoc closest Gamma may switch by geometry/ties. D_edge training uses fixed GT free-corner correspondence; no such nearest-edge assignment or loss gradient is evaluated by sampling. A-B-A is not proof of ODE stiffness or learned attraction.',
        CPU_geometry_gradients=0,seconds=time.perf_counter()-tick,source_sha256=sha(__file__))
    write(dest/'trajectory_summary.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--stage',choices=['fixtures','teacher','trajectories'],required=True)
    a=p.parse_args();torch.set_num_threads(1)
    if torch.cuda.is_initialized():raise RuntimeError('CPU audit must not initialize CUDA')
    r={'fixtures':run_fixtures,'teacher':run_teacher,'trajectories':run_trajectories}[a.stage](a.out)
    print(json.dumps(dict(status=r['status'],counts=r['counts'])),flush=True)

if __name__=='__main__':main()
