"""CPU-only preflight. A failed mathematical gate prevents any GPU experiment."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time,hashlib,sys
from pathlib import Path
import numpy as np
import torch
from .loss import boundary_loss,prepare_target,uniform_probes,face_normals

OUT=Path('native_t1/runs/boundary_cancellation_20261001_161219')
DTYPE=torch.float64
SEED=20261001
COUNTS=dict(loss_evaluations=0,coordinate_autograd_calls=0,finite_difference_loss_evaluations=0,
            model_forwards=0,model_backwards=0,optimizer_updates=0,OT_calls=0,GPU_calls=0,independent_algebraic_reference_evaluations=0)
ARRAYS={}

def tensor(x):return torch.as_tensor(x,dtype=DTYPE)
def normals(x):return face_normals(x).detach()
def evaluate(name,pred,target,grad=False):
    COUNTS['loss_evaluations']+=1
    x=pred.detach().clone().requires_grad_(grad)
    o=boundary_loss(x,target)
    g=None
    if grad:
        COUNTS['coordinate_autograd_calls']+=1;g=torch.autograd.grad(o['loss'],x)[0]
    row=dict(name=name,loss=float(o['loss'].detach()),per_scale=o['per_scale'].detach().tolist(),
        residual_max_abs=float(o['residuals'].detach().abs().max()),
        free_faces=len(x),known_boundary_edges=o['known_boundary_edges'],
        normal_reversals=o['reversed_faces'],normal_ties=o['normal_alignment_ties'],
        context_audit=target.context_audit,
        gradient_norm=None if g is None else float(torch.linalg.vector_norm(g)),
        gradient_finite=None if g is None else bool(torch.isfinite(g).all()))
    for key,value in dict(pred=x,probes=target.probes,known_midpoints=target.known_midpoints,
        known_vectors=target.known_vectors,reference_normals=target.free_reference_normals,
        residuals=o['residuals'],gradient=g).items():
        if value is not None:ARRAYS[name+'_'+key]=value.detach().cpu().numpy()
    ARRAYS[name+'_bbox_L']=np.array(target.bbox_L)
    return row,g

def setup(full,ids,known_indices):
    known_indices=list(known_indices);free_indices=[i for i in range(len(full)) if i not in known_indices]
    probes,L=uniform_probes(full,SEED)
    target=prepare_target(full[known_indices],ids[known_indices],normals(full[known_indices]),
        normals(full[free_indices]),probes,L)
    return full[free_indices],target,free_indices

def outward(vertices,faces):
    faces=[list(f) for f in faces];center=vertices.mean(0)
    for f in faces:
        tri=vertices[f]
        if torch.dot(face_normals(tri[None])[0],tri.mean(0)-center)<0:f[1],f[2]=f[2],f[1]
    return torch.tensor(faces,dtype=torch.long)

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'cpu_fixtures.json').exists():raise FileExistsError('No automatic fixture rerun')
    tick=time.perf_counter();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rows=[];gates={}
    V=tensor([[0,0,0],[2,0,0],[0,1,0],[0,0,.7]])
    F=outward(V,[[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
    tri=F[0].tolist();bary=tensor([.15,.30,.55]);V=torch.cat((V,(V[tri]*bary[:,None]).sum(0)[None]),dim=0)
    F=torch.cat((torch.tensor([[tri[0],tri[1],4],[tri[1],tri[2],4],[tri[2],tri[0],4]]),F[1:]),dim=0)
    full=V[F];pred,target,freeidx=setup(full,F,[0,3])
    row,g=evaluate('case1_closed_nonuniform',pred,target,True)
    areas=torch.linalg.vector_norm(normals(full),dim=-1)/2
    row['face_area_ratio_max_min']=float(areas.max()/areas.min());rows.append(row)
    gates['case1_closed_nonuniform']=bool(row['loss']<1e-6 and all(x<1e-6 for x in row['per_scale']))
    # Independent algebraic reference: include ALL half-edges of C and F.
    mids=(full+full.roll(-1,1))/2;vec=full.roll(-1,1)-full
    ref=[]
    for sigma in (.05,.10):
        w=torch.exp(-(target.probes[:,None,:]-mids.reshape(-1,3)[None]).square().sum(-1)/(2*(sigma*target.bbox_L)**2))
        ref.append(float((w[:,:,None]*vec.reshape(-1,3)[None]).sum(1).square().sum(-1).mean()/target.bbox_L**2))
    COUNTS['independent_algebraic_reference_evaluations']+=1
    row['all_halfedges_reference_per_scale']=ref
    row['context_interior_removal_equivalent']=max(abs(a-b) for a,b in zip(ref,row['per_scale']))<1e-25
    # Odd corner permutations restore original geometric orientation via immutable labels.
    order=torch.tensor([2,0,3,1]);permuted=pred[order][:,[1,0,2]]
    ki=[0,3];C=full[ki][:,[2,1,0]];CI=F[ki][:,[2,1,0]]
    perm_target=prepare_target(C,CI,normals(full[ki]),normals(pred)[order],target.probes,target.bbox_L)
    p_row,_=evaluate('case1_odd_corner_and_face_permutation',permuted,perm_target)
    p_row['absolute_loss_delta']=abs(p_row['loss']-row['loss']);p_row['invariance_PASS']=p_row['absolute_loss_delta']<1e-25;rows.append(p_row)
    V=tensor([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
    F=outward(V,[[0,1,2],[0,2,3],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    pred,target,_=setup(V[F],F,[0,1,4]);row,g=evaluate('case2_closed_cube_sharp_ridges',pred,target,True);rows.append(row)
    gates['case2_sharp_legal_ridges']=bool(row['loss']<1e-6 and all(x<1e-6 for x in row['per_scale']))
    # Literal two adjacent triangles with an opening along their shared edge.
    V=tensor([[0,0,0],[1,0,0],[0,1,0],[1,1,0]]);F=torch.tensor([[0,1,2],[1,3,2]])
    full=V[F];pred,target,_=setup(full,F,[0]);direction=tensor([1,1,0])/2**.5;delta=.01*target.bbox_L
    opened=pred+delta*direction
    row,g=evaluate('case3_two_adjacent_faces_crack',opened,target,True)
    grad_proj=g[0,[0,2]]@direction
    row['gap_delta']=delta;row['shared_endpoint_closing_gradient_projections']=grad_proj.tolist()
    row['translation_directional_derivative']=float((g*direction).sum())
    rate=(1e-4*target.bbox_L)/(float(torch.linalg.vector_norm(g))+1e-30)
    nextpred=opened-rate*g
    newrow,_=evaluate('case3_single_coordinate_gradient_probe',nextpred,target)
    before_gap=torch.linalg.vector_norm(opened[0,[0,2]]-pred[0,[0,2]],dim=-1).square().mean().sqrt()
    after_gap=torch.linalg.vector_norm(nextpred[0,[0,2]]-pred[0,[0,2]],dim=-1).square().mean().sqrt()
    row.update(shared_endpoint_gap_RMS_before=float(before_gap),shared_endpoint_gap_RMS_after=float(after_gap),
        normalized_coordinate_probe_step=rate,loss_after_coordinate_probe=newrow['loss'])
    gates['case3_crack_closing_gradients']=bool(row['loss']>0 and (grad_proj>0).all() and after_gap<before_gap and newrow['loss']<row['loss'])
    eps=1e-6*target.bbox_L
    plus,_=evaluate('case3_FD_plus',opened+eps*direction,target)
    minus,_=evaluate('case3_FD_minus',opened-eps*direction,target)
    COUNTS['finite_difference_loss_evaluations']+=2
    row['FD_directional_derivative']=(plus['loss']-minus['loss'])/(2*eps)
    row['FD_absolute_error']=abs(row['FD_directional_derivative']-row['translation_directional_derivative'])
    rows.extend([row,newrow,plus,minus])
    # No gradient may reach C, normals or probes, including when callers mark them trainable.
    C=full[:1].detach().clone().requires_grad_();nr=normals(pred).clone().requires_grad_();q=target.probes.clone().requires_grad_()
    detached_target=prepare_target(C,F[:1],normals(full[:1]),nr,q,target.bbox_L)
    leaf=opened.clone().requires_grad_();COUNTS['loss_evaluations']+=1;o=boundary_loss(leaf,detached_target)
    COUNTS['coordinate_autograd_calls']+=1;detached_grads=torch.autograd.grad(o['loss'],(leaf,C,nr,q),allow_unused=True)
    detached_check=dict(status='PASS' if all(g is None for g in detached_grads[1:]) else 'FAIL',
        free_gradient_finite=bool(torch.isfinite(detached_grads[0]).all()),C_gradient_is_None=detached_grads[1] is None,
        GT_normal_gradient_is_None=detached_grads[2] is None,probe_gradient_is_None=detached_grads[3] is None)
    # Duplicate triangles: same GT direction versus opposite GT directions.
    tri=tensor([[[0,0,0],[1,0,0],[0,1,0]]]);up=normals(tri)
    gt=torch.cat((tri+tensor([0,0,.1]),tri[:,[0,2,1]]+tensor([0,0,-.1])),dim=0)
    probes,L=uniform_probes(gt,SEED)
    empty=tri[:0];ids=torch.empty((0,3),dtype=torch.long)
    single=prepare_target(empty,ids,up[:0],up,probes,L)
    one,_=evaluate('case4_single_reference',tri,single)
    same=prepare_target(empty,ids,up[:0],up.repeat(2,1),probes,L)
    wrong=torch.cat((tri,tri[:,[0,2,1]]),dim=0)
    row,g=evaluate('case4a_overlap_same_GT_normals',wrong,same,True)
    row['loss_over_single']=row['loss']/one['loss'];row['corrected_input_odd_winding']=row['normal_reversals']==1
    gates['case4a_same_GT_normals_prevent_false_cancellation']=bool(row['loss']>1e-12 and abs(row['loss_over_single']-4)<1e-12)
    rows.extend([one,row])
    opposite=prepare_target(empty,ids,up[:0],normals(gt),probes,L)
    coincident=torch.cat((tri,tri),dim=0)
    row,g=evaluate('case4b_overlap_opposite_GT_normals',coincident,opposite,True)
    row['predicted_triangle_areas']=(torch.linalg.vector_norm(normals(coincident),dim=-1)/2).tolist()
    row['GT_reference_plane_separation']=.2
    row['predicted_plane_separation']=0.
    row['GT_normals_dot']=float(torch.dot(normals(gt)[0],normals(gt)[1]))
    row['false_cancellation_detected']=row['loss']<1e-24 and row['gradient_norm']<1e-12
    gates['case4b_opposite_GT_normals_prevent_false_cancellation']=not row['false_cancellation_detected']
    rows.append(row)
    direction=tensor([0,0,1]);gap=.01*L
    separated=torch.cat((tri+gap*direction/2,tri-gap*direction/2),dim=0)
    row,g=evaluate('case4c_parallel_faces_small_separation',separated,opposite,True)
    row['plane_separation']=gap;row['separation_directional_derivative']=float(((g[0]-g[1])*direction/2).sum())
    row['gradient_descent_reduces_separation']=row['separation_directional_derivative']>0;rows.append(row)
    # The identical fixture is translated and scaled without retuning probes/scales.
    factor=7.;offset=tensor([13,-8,4])
    scaled_target=prepare_target(empty,ids,up[:0],up.repeat(2,1),probes*factor+offset,L*factor)
    scaled,_=evaluate('case4a_global_scale_translation',wrong*factor+offset,scaled_target)
    base=next(x for x in rows if x['name']=='case4a_overlap_same_GT_normals')
    scaled['scale_invariance_abs_error']=abs(scaled['loss']-base['loss']);rows.append(scaled)
    # Real GT oracle: raw orientation; naive archive orientation; restored source normals.
    oracle=[]
    with np.load(OUT/'cpu_oracle_geometry.npz',allow_pickle=False) as z:
        for parent in (0,1):
            raw=tensor(z[f'parent{parent}_raw_winding_target'].reshape(-1,3,3));arc=tensor(z[f'parent{parent}_archive_target'].reshape(-1,3,3))
            rawids=torch.tensor(z[f'parent{parent}_source_face_vertex_ids_raw_canonical']);arcids=torch.tensor(z[f'parent{parent}_source_face_vertex_ids_archive'])
            ki=z[f'parent{parent}_K12_source_face_ids'].tolist();fi=z[f'parent{parent}_K12_free_source_ids'].tolist()
            probes,L=uniform_probes(raw,SEED)
            for variant in ('raw','archive_naive','archive_restored'):
                geom=raw if variant=='raw' else arc
                ids=rawids if variant=='raw' else arcids
                refs=normals(arc) if variant=='archive_naive' else normals(raw)
                target=prepare_target(geom[ki],ids[ki],refs[ki],refs[fi],probes,L)
                row,_=evaluate(f'P{parent}_oracle_{variant}',geom[fi],target)
                row['interpretation']='Invalid inconsistent orientation diagnostic only' if variant=='archive_naive' else 'Consistent source GT oracle'
                oracle.append(row)
    numeric_checks=dict(context_boundary_equals_all_edges=rows[0]['context_interior_removal_equivalent'],
        permutation=rows[1]['invariance_PASS'],crack_finite_difference=abs(next(x for x in rows if x['name']=='case3_two_adjacent_faces_crack')['FD_absolute_error'])<1e-7,
        scale_translation=scaled['scale_invariance_abs_error']<1e-12,
        detached_supervision=detached_check['status']=='PASS',
        restored_real_GT_oracles=all(x['loss']<1e-6 for x in oracle if 'restored' in x['name'] or x['name'].endswith('_raw')))
    result=dict(schema='boundary_cancellation_CPU_preflight_v1',status='PASS' if all(gates.values()) and all(numeric_checks.values()) else 'FAIL_PRE_GPU_GATE',
        gate_checks=gates,numerical_implementation_checks=numeric_checks,fixtures=rows,real_GT_oracles=oracle,
        detached_supervision=detached_check,counts=COUNTS,
        policy=dict(dtype='CPU_FP64',probe_count=64,probe_seed=SEED,probe_region='uniform fixed GT axis-aligned bbox',
          sigma_fractions=[.05,.10],scale_combination='arithmetic_mean',tau='length-bearing oriented edge vector',
          orientation='negative dot with detached trusted source GT normal reverses half-edge direction; zero dot unchanged',
          threshold_closed_loss=1e-6,same_case4_probes_for_all_orientation_variants=True,
          coefficient_calibration='NOT_RUN_UNTIL_ALL_CPU_GATES_PASS',training='NOT_RUN_UNTIL_ALL_CPU_GATES_PASS',
          generation='NOT_RUN_UNTIL_ALL_CPU_GATES_PASS',parameter_sweep=False),
        model_calls=0,optimizer_updates=0,CUDA_initialized=torch.cuda.is_initialized(),seconds=time.perf_counter()-tick)
    np.savez_compressed(OUT/'fixture_arrays.npz',**ARRAYS)
    for f in [Path(__file__),Path(__file__).with_name('loss.py'),OUT/'fixture_arrays.npz',OUT/'cpu_oracle_geometry.npz']:
        result.setdefault('files',[]).append(dict(path=str(f.resolve()),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size))
    (OUT/'cpu_fixtures.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','gate_checks','numerical_implementation_checks','counts','seconds')},indent=2))
    for row in rows+oracle:print(row['name'],row['loss'],row.get('gradient_norm'),row.get('shared_endpoint_closing_gradient_projections'))
    return result

if __name__=='__main__':
    try:
        result=run()
    except Exception as exc:
        import traceback
        failure=dict(status='UNEXPECTED_IMPLEMENTATION_EXCEPTION',error=repr(exc),traceback=traceback.format_exc(),counts=COUNTS,CUDA_initialized=torch.cuda.is_initialized())
        (OUT/'cpu_fixture_exception.json').write_text(json.dumps(failure,indent=2)+'\n',encoding='utf-8')
        raise
    sys.exit(0 if result['status']=='PASS' else 2)
