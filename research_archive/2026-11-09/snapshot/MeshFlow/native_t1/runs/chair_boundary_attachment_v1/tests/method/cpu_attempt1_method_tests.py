"""Finite CPU algebra/topology/stream checks; never constructs a real model."""
from pathlib import Path
import ast
import copy
import hashlib
import inspect
import json
import textwrap
import time
import traceback
import numpy as np
import torch
from .method import (boundary_marks,inject_boundary,attach_boundary,seam_targets,seam_loss,
    analytic_ucorr,_boundary_forward)
from .stream import AttachmentStream,prepare_training_plan,make_plan,ARMS
from experiments.mf_h1_local0.control_mvp48.model_mvp import MVPModel
from experiments.mf_h1_local0.chair_hybrid_coupling_v1.stream import HybridStream

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'native_t1/runs/chair_boundary_attachment_v1'


def fixture(reuse=False):
    V=np.array([[0,0,0],[1,0,0],[0,1,0],[0,-1,0],[1,1,0]],np.float32)
    F=np.array([[0,1,2],[1,0,3]],np.int64) if not reuse else np.array([[1,0,3],[2,1,4],[0,1,2]],np.int64)
    k=1 if not reuse else 2;n=len(F); known=np.arange(n)<k;tri=V[F]
    C=np.zeros((n,9),np.float32);Y=C.copy();C[known]=tri[known].reshape(-1,9);Y[~known]=tri[~known].reshape(-1,9)
    s=dict(y=np.asarray(n),K=k,known_mask=known,valid_mask=np.ones(n,bool),source_face_ids=np.arange(n),
        corner_permutations=np.tile(np.arange(3),(n,1)),x1=Y,context=C,
        _geometry=dict(V=V,F=F,bbox_L=float(np.linalg.norm(np.ptp(tri.reshape(-1,3),axis=0)))),
        _seam_source=dict(source_faces=F,original_source_faces=F.copy(),retained_to_original_face=np.arange(n),
            trusted_interface=dict(edge_source_vertex_ids=np.array([[0,1]] if not reuse else [[0,1],[1,2]],np.int64))))
    s['_seam_targets']=seam_targets(s)
    return s


def run(out=OUT):
    torch.set_num_threads(2);started=time.monotonic();dest=Path(out)/'tests/method';dest.mkdir(parents=True,exist_ok=True)
    checks=[];counts=dict(CPU_leaf_gradient_calls=0,real_model_forwards=0,real_model_backwards=0,optimizer_updates=0)
    def check(name,ok,**details):
        checks.append(dict(name=name,status='PASS' if bool(ok) else 'FAIL',**details))
        if not bool(ok):raise AssertionError(name)
    def grad(value,inputs):
        counts['CPU_leaf_gradient_calls']+=1
        return torch.autograd.grad(value,inputs)
    def reject(fn):
        try:fn();return False
        except ValueError:return True
    paths=[ROOT/'experiments/mf_h1_local0/chair_hybrid_coupling_v1'/x for x in ('stream.py','coupling.py')]
    paths += [ROOT/p for p in ('experiments/mf_h1_local0/control_mvp48/model_mvp.py','experiments/mf_h1_local0/control_mvp48/train_stream.py','experiments/mf_h1_local0/chair_domain_control_v1/prepare.py','native_t1/losses/edge.py','native_t1/context_geometry.py','utils/ot_utils.py')]
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    before={str(p.relative_to(ROOT)):digest(p) for p in paths}
    failure=None;real_rows=[]
    try:
        # The original MVP body differs only by the explicitly declared injection.
        old=ast.parse(textwrap.dedent(inspect.getsource(MVPModel.forward))).body[0]
        new=ast.parse(textwrap.dedent(inspect.getsource(_boundary_forward))).body[0]
        new.name=old.name
        new.body=[n for n in new.body if not (isinstance(n,ast.Assign) and (isinstance(n.targets[0],ast.Tuple) and [getattr(t,'id',None) for t in n.targets[0].elts]==['marks','diagnostics'] or isinstance(n.targets[0],ast.Attribute) and n.targets[0].attr=='boundary_observations' or isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='inject_boundary')) and not (isinstance(n,ast.If) and 'site=\'boundary\'' in ast.unparse(n))]
        # ast.unparse renders keyword strings using either quote consistently here.
        check('forward_AST_identical_after_removing_only_boundary_statements',ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False))
        V=torch.tensor([[0.,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]])
        F=torch.tensor([[0,1,2],[0,2,3],[0,3,4],[0,4,1]])
        x=torch.zeros(1,7,9);x[0,:4]=V[F].reshape(4,9);known=torch.tensor([[1,1,1,1,0,0,0]],dtype=torch.bool);valid=torch.tensor([[1,1,1,1,1,0,0]],dtype=torch.bool)
        m,d=boundary_marks(x,known,valid)
        check('boundary_open_shared_center_identity',d[0]['open_edges']==4 and d[0]['shared_edges']==4 and not m[0,:4,0].any() and m[0,:4,1:].all() and not m[0,4:].any())
        xp=x.clone();xp[0,4:]=float('nan');mp,dp=boundary_marks(xp,known,valid)
        check('hidden_free_and_padding_never_read',torch.equal(m,mp) and d==dp)
        perm=torch.tensor([3,0,6,1,5,2,4]);cp=torch.tensor([[2,0,1],[1,2,0],[0,2,1],[2,1,0],[1,0,2],[0,1,2],[2,0,1]])
        xx=x[:,perm].reshape(1,7,3,3).gather(2,cp[None,:,:,None].expand(1,7,3,3)).reshape(1,7,9)
        mm,dd=boundary_marks(xx,known[:,perm],valid[:,perm]);expect=m[:,perm].gather(2,cp[None])
        check('boundary_face_corner_permutation_exact',torch.equal(mm,expect) and d==dd)
        xa=x.clone().reshape(1,7,3,3);xa[...,0]*=np.float32(1.1);ma,da=boundary_marks(xa.reshape(1,7,9),known,valid)
        check('boundary_recomputed_actual_alpha_C',torch.equal(ma,m) and da==d)
        tri=torch.tensor([[[0.,0,0],[1,0,0],[0,1,0]]]);dup=tri.repeat(3,1,1).reshape(1,3,9);mask=torch.ones(1,3,dtype=torch.bool)
        dm,di=boundary_marks(dup,mask,mask)
        check('nonmanifold_count_not_open',di[0]['ambiguous_edges']==3 and di[0]['open_edges']==0 and not dm.any())
        two=dup[:,:2].clone();two[0,1,0]=-0.;tm,td=boundary_marks(two,mask[:,:2],mask[:,:2])
        check('signed_zero_is_exact_shared',td[0]['shared_edges']==3 and not tm.any())
        two[0,1,0]=torch.nextafter(torch.tensor(0.),torch.tensor(1.));tm,td=boundary_marks(two,mask[:,:2],mask[:,:2])
        check('no_threshold_welding',td[0]['open_edges']==4)
        bad=tri.clone();bad[:,1]=bad[:,0]
        check('zero_edge_rejected_not_deleted',reject(lambda:boundary_marks(bad.reshape(1,1,9),mask[:,:1],mask[:,:1])))
        for dtype in (torch.float32,torch.bfloat16):
            hidden=torch.arange(7*3*768,dtype=torch.float32).reshape(1,21,768).to(dtype)/7; e=torch.zeros(768,requires_grad=True)
            inj=inject_boundary(hidden,m,e)
            check('zero_injection_exact_'+str(dtype),torch.equal(inj,hidden))
            eg=grad(inj.float().sum(),e)[0]
            check('zero_vector_gradient_connected_'+str(dtype),torch.equal(eg,torch.full_like(eg,float(m.sum()))))
        class Fake(torch.nn.Module):
            def __init__(self):
                super().__init__();self.p=torch.nn.Parameter(torch.ones(1));self.backbone=torch.nn.Module();self.backbone.hidden_size=768;self.experiment_trainable=True
        a,b=Fake(),Fake();am=attach_boundary(a,False);bm=attach_boundary(b,True)
        check('exact_one_vector_and_no_disabled_parameters',am['new_parameter_count']==0 and bm['new_parameter_names']==['e_boundary'] and bm['new_parameter_count']==768 and not b.e_boundary.detach().any())
        # Exact 1D polynomial integral, endpoint symmetry, scaling and gradients.
        gt=torch.tensor([[[0.,0,0],[1,0,0]],[[0,0,0],[4,0,0]]],dtype=torch.float64)
        pred=gt+torch.tensor([[[.2,.3,-.1],[-.4,.1,.2]],[[.1,-.2,.7],[.3,.2,-.5]]],dtype=torch.float64)
        nodes,weights=np.polynomial.legendre.leggauss(256);s=torch.tensor((nodes+1)/2)[None,:,None];w=torch.tensor(weights/2)[None]
        rr=(1-s)*(pred-gt)[:,:1]+s*(pred-gt)[:,1:];numerical=(rr.square().sum(-1)*w).sum(-1).mean()/9
        exact=analytic_ucorr(pred,gt,3.)
        check('analytic_matches_high_precision_1D_integral',abs(float(exact-numerical))<1e-14,error=float(exact-numerical))
        check('synchronous_endpoint_swap_invariance',torch.equal(exact,analytic_ucorr(pred.flip(1),gt.flip(1),3.)))
        perfect=gt.clone().requires_grad_();zero=analytic_ucorr(perfect,gt,1.);zg=grad(zero,perfect)[0]
        check('exact_coincidence_zero_loss_and_gradient',float(zero)==0 and not zg.any())
        one=gt[:1]
        translated=one+torch.tensor([.3,-.4,0.]);check('translation_expected_squared_norm',abs(float(analytic_ucorr(translated,one,1.))-.25)<1e-14)
        for factor in (.5,2.):
            changed=one.clone();changed[:,1]*=factor
            check('stretch_shrink_analytic_'+str(factor),abs(float(analytic_ucorr(changed,one,1.))-(factor-1)**2/3)<1e-14)
        rotated=one.clone();rotated[:,1]=torch.tensor([0.,1.,0.]);check('turning_analytic',abs(float(analytic_ucorr(rotated,one,1.))-2/3)<1e-14)
        # Equal-edge weighting, not GT edge length weighting.
        trans=gt.clone();trans[0]+=1.;check('equal_edge_weight',abs(float(analytic_ucorr(trans,gt,1.))-1.5)<1e-14)
        sample=fixture();target=sample['_seam_targets'];n=int(sample['y']);tt=.8
        xx=torch.tensor(sample['x1'],dtype=torch.float64)*.3;vel=((torch.tensor(sample['x1'],dtype=torch.float64)-xx)/(1-tt)).requires_grad_();xx.requires_grad_()
        E,diag=seam_loss(vel,xx,tt,sample);vg=grad(E,vel)[0]
        check('GT_velocity_recovers_endpoint_zero',abs(float(E))<1e-28 and float(vg.abs().max())<1e-13)
        vv=(vel.detach()+.5).requires_grad_();E,_=seam_loss(vv,xx,tt,sample);gg=grad(E,vv)[0]
        check('known_has_no_loss_gradient',not gg[torch.tensor(sample['known_mask'])].any() and xx.grad is None)
        pv=torch.cat((vv.detach(),torch.full((3,9),float('nan'),dtype=torch.float64))).requires_grad_();px=torch.cat((xx.detach(),torch.full((3,9),float('inf'),dtype=torch.float64)))
        pe,_=seam_loss(pv,px,tt,sample);pg=grad(pe,pv)[0]
        check('padding_ignored_by_loss_and_gradient',torch.equal(pe,E) and not pg[n:].any())
        # All six corner permutations and a face swap retain true GT mapping.
        import itertools
        for order in itertools.permutations(range(3)):
            z=copy.deepcopy(sample);fp=np.array([1,0]);cp=np.tile(np.array(order),(n,1))
            for key in ('known_mask','valid_mask','source_face_ids'):z[key]=z[key][fp]
            z['corner_permutations']=cp
            for key in ('context','x1'):z[key]=z[key][fp].reshape(n,3,3)[:,np.array(order)].reshape(n,9)
            z['_seam_targets']=seam_targets(z)
            check('post_OT_source_corner_permutation_'+str(order),np.array_equal(z['_seam_targets']['GT_endpoints'],target['GT_endpoints']))
        rs=fixture(reuse=True);rt=rs['_seam_targets'];check('shared_corner_reuse_recorded',rt['counts']['max_corner_reuse']==2 and rt['counts']['reused_free_corners']==1)
        rv=torch.zeros_like(torch.tensor(rs['x1'],dtype=torch.float64),requires_grad=True);rx=torch.tensor(rs['x1'],dtype=torch.float64)+.2
        re,_=seam_loss(rv,rx,.5,rs);rg=grad(re,rv)[0]
        manual=torch.zeros_like(rg)
        for f,cs in zip(rt['free_faces'],rt['free_corners']):
            for c in cs:manual.reshape(-1,3,3)[f,c]+=.5*.2/(rt['bbox_L']**2*len(rt['free_faces']))
        check('shared_corner_gradient_accumulates_without_degree_division',torch.allclose(rg,manual,atol=1e-14,rtol=1e-13))
        bad=copy.deepcopy(sample);bad['_seam_source']['original_source_faces']=np.r_[bad['_seam_source']['original_source_faces'],[[0,1,4]]];bad.pop('_seam_targets');bt=seam_targets(bad);bad['_seam_targets']=bt
        bv=torch.zeros(n,9,requires_grad=True);be,bd=seam_loss(bv,torch.zeros_like(bv),.6,bad);bg=grad(be,bv)[0]
        check('original_nonmanifold_excluded_no_valid_seam_zero',bt['status']=='NO_VALID_SEAM' and float(be)==0 and not bg.any())
        partial=fixture(reuse=True);partial['_seam_source']['trusted_interface']['edge_source_vertex_ids']=np.array([[0,1]]);pt=seam_targets(partial)
        check('partial_eligibility_keeps_good_edges_and_full_FM',pt['counts']['candidate_interface_edges']==2 and pt['counts']['eligible_interface_edges']==1 and len(partial['valid_mask'])==3)
        check('time_gate_boundary',seam_loss(vv,xx,.5,sample)[1]['gate'] and not seam_loss(vv,xx,.499,sample)[1]['gate'])
        # First frozen real batch only: eight OT assignments; other arms/cache
        # checks reuse those exact maps and never load any model checkpoint.
        source=ROOT/'native_t1/runs/control_mvp48_v1';chair=ROOT/'native_t1/runs/chair_domain_control_v1'
        plan=prepare_training_plan(source,chair,out);doc=json.loads((Path(out)/'train_plan.json').read_text())
        check('plan_32parents_250exposures_1000batch',len(doc['records'])==8000 and len(doc['object_counts'])==32 and set(doc['object_counts'].values())=={250} and all(len({r['uid'] for r in doc['records'][i:i+8]})==8 and sum(r['alpha']==1. for r in doc['records'][i:i+8])==4 for i in range(0,8000,8)))
        reference=None
        for arm in ARMS:
            stream=AttachmentStream(source,chair,out,arm);batch=stream.next_effective_batch()
            if reference is None:reference=batch
            check('actual_shared_first_batch_'+arm,all(s['shared_input_sha256']==r['shared_input_sha256'] and s['coupled_input_sha256']==r['coupled_input_sha256'] and s['_seam_targets']['target_sha256']==r['_seam_targets']['target_sha256'] for s,r in zip(batch,reference)))
            state=stream.state_dict();again=AttachmentStream(source,chair,out,arm);again.load_state_dict(state);check('restore_cursor_'+arm,again.batch_index==1)
            real_rows.append(dict(arm=arm,stream_state=state,OT_counts=stream.counts,first_batch_sha256=stream.last_audit['shared_batch_sha256'],samples=[dict(uid=s['object_id'],N=int(s['y']),K=s['K'],alpha=s['alpha'],boundary=s['boundary_diagnostics'],seam=s['_seam_targets']['counts'],target_sha256=s['_seam_targets']['target_sha256'],common_array_hashes=s['common_array_hashes']) for s in batch]))
        baseline=HybridStream._sample(stream,stream.records[0])
        check('new_stream_preserves_original_hybrid_tensor_math',all(np.array_equal(reference[0][k],baseline[k]) for k in ('x0','x1','u','xt','context','source_face_ids','corner_permutations','free_gaussian_before_OT','free_epsilon','ot_face_map','ot_corner_map')))
        check('source_files_unchanged',all(digest(ROOT/p)==h for p,h in before.items()))
        check('CUDA_never_initialized',not torch.cuda.is_initialized())
    except Exception:
        failure=traceback.format_exc()
    result=dict(status='PASS' if failure is None else 'FAIL',checks=checks,counts=counts,real_first_batch=real_rows,source_hashes=before,
        implementation_hashes={p.name:digest(p) for p in (Path(__file__),Path(__file__).with_name('method.py'),Path(__file__).with_name('stream.py'))},failure=failure,seconds=time.monotonic()-started)
    index=1
    while (dest/f'cpu_attempt{index}.json').exists():index+=1
    (dest/f'cpu_attempt{index}.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],checks=len(checks),counts=counts,seconds=result['seconds'],failure=failure),indent=2))
    if failure:raise RuntimeError('CPU method checks failed; see preserved attempt')
    (dest/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':run()
