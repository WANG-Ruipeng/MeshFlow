"""Small deterministic CPU/autograd checks; no model construction or CUDA use."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import copy,hashlib,itertools,json,math
import numpy as np
import torch
from .prepare import OUT,arrays,read,sha,write,SCALE
from .relative_loss import prepare_target,relative_loss

def run():
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 cfg=read(OUT/'loss_config.json');checks=[]
 def mark(name,**extra):checks.append(dict(name=name,status='PASS',**extra))
 tri=np.array([[[0,0,0],[1,0,0],[0,1,0]]],np.float32)
 mask=np.ones((1,3),bool);prepared=prepare_target(tri,mask,1,cfg)
 def loss(x,p=prepared):return relative_loss(x,p,cfg,torch)
 x=torch.tensor(tri,requires_grad=True);v=loss(x);v.backward()
 assert v.item()==0 and torch.equal(x.grad,torch.zeros_like(x));mark('identity_zero_loss_and_gradient')
 x=torch.tensor(tri*1.1,requires_grad=True);v=loss(x)
 expected=.5*(.1*math.sqrt(2/6))**2
 assert math.isclose(v.item(),expected,rel_tol=2e-6,abs_tol=1e-9);mark('uniform_scale_1p1_analytic_response',loss=v.item(),analytic=expected)
 translated=torch.tensor(tri+np.array([[[.25,-.5,1]]],np.float32))
 assert loss(translated).item()==0;mark('global_translation_not_penalized_by_edge_term')
 pred=tri.copy();pred[0,1]+=[.1,.025,.03];pred[0,2]+=[-.02,.04,-.01]
 theta=.7;rot=np.array([[math.cos(theta),-math.sin(theta),0],[math.sin(theta),math.cos(theta),0],[0,0,1]],np.float32)
 tr=tri@rot.T;pr=pred@rot.T
 rotated=prepare_target(tr,mask,1,cfg)
 a=loss(torch.tensor(pred)).item();b=loss(torch.tensor(pr),rotated).item()
 assert math.isclose(a,b,rel_tol=4e-6,abs_tol=2e-9);mark('joint_rigid_frame_rotation_invariance_unclipped',original=a,rotated=b)
 assert loss(torch.tensor(tr)).item()>0;mark('prediction_only_rotation_is_penalized')
 errs=[]
 for perm in itertools.permutations(range(3)):
  t=tri[:,perm,:].copy();p=pred[:,perm,:].copy();cache=prepare_target(t,mask,1,cfg)
  val=loss(torch.tensor(p),cache).item();errs.append(abs(val-a));assert math.isclose(val,a,rel_tol=5e-6,abs_tol=2e-9)
 mark('matched_corner_permutations_unclipped_only',max_absolute_error=max(errs))
 assert loss(torch.tensor(tri[:,[1,0,2],:].copy())).item()>0;mark('wrong_corner_correspondence_penalized')
 near=np.array([[[0,0,0],[1,0,0],[1,1e-5,0]],[[0,0,0],[1,0,0],[1,1e-8,0]]],np.float32)
 cache=prepare_target(near,np.ones((2,3),bool),1,cfg)
 assert cache['clipped_faces']==1 and cache['eligible_faces']==1 and cache['excluded_rank_faces']==1
 assert cache['maximum_inverse_norm']<=100.00001
 z=torch.tensor(near,requires_grad=True);z2=z+torch.tensor([[[0,0,0],[0,0,0],[0,.01,0]]]*2)
 v=loss(z2,cache);v.backward();assert torch.isfinite(v) and torch.isfinite(z.grad).all();mark('near_degenerate_bounded_inverse_and_finite_gradient',maximum_inverse_norm=cache['maximum_inverse_norm'])
 x=torch.tensor(pred,requires_grad=True);v=loss(x);v.backward();analytic=x.grad.numpy().copy();errors=[]
 for index in [(0,0,0),(0,0,2),(0,1,0),(0,1,1),(0,2,1),(0,2,2)]:
  eps=1e-3;xp=pred.copy();xm=pred.copy();xp[index]+=eps;xm[index]-=eps
  numeric=(loss(torch.tensor(xp)).item()-loss(torch.tensor(xm)).item())/(2*eps)
  errors.append(abs(numeric-float(analytic[index])));assert math.isclose(numeric,float(analytic[index]),rel_tol=.003,abs_tol=3e-6),(index,numeric,analytic[index])
 mark('CPU_finite_difference_gradient',max_absolute_error=max(errors))
 duplicated=np.repeat(pred,3,axis=0);base=np.repeat(tri,3,axis=0);m=np.array([[False]*3,[True]*3,[True,False,True]])
 base_before=base.copy();mask_before=m.copy();pc=prepare_target(base,m,1,cfg)
 assert pc['face_ids'].tolist()==[1] and np.array_equal(base,base_before) and np.array_equal(m,mask_before)
 z=torch.tensor(duplicated,requires_grad=True);v=loss(z,pc);v.backward()
 assert not z.grad[0].any() and not z.grad[2].any();mark('C_and_partial_face_mask_excluded_without_target_mutation')
 doubled=prepare_target(np.repeat(tri,2,0),np.ones((2,3),bool),1,cfg)
 pv=np.concatenate((pred,tri*1.1),axis=0)
 assert math.isclose(loss(torch.tensor(pv),doubled).item(),(a+expected)/2,rel_tol=3e-6,abs_tol=2e-9);mark('per_face_mean_reduction')
 m=read(OUT/'data/relative_cache_manifest.json');validation_parents=[]
 for p in m['parents']:
  assert sha(p['path'])==p['sha256'];raw=arrays(p['representative_input_path']);c=arrays(p['path']);ids=c['face_ids']
  td=raw['target_model'].astype(np.float32)*np.float32(SCALE);E=np.stack((td[:,1]-td[:,0],td[:,2]-td[:,0]),-1)
  assert np.array_equal(E[ids],c['target_edges'])
  assert raw['loss_mask'][ids].all() and (ids>=int(raw['known_faces'])).all()
  assert c['inverse'].dtype==np.float32 and np.isfinite(c['inverse']).all()
  # Independent SVD bound; no regenerated cache or noise.
  assert np.linalg.svd(c['inverse'].astype(float),compute_uv=False)[:,0].max()<=100.00001
  if p['split']=='validation':validation_parents.append(p['parent'])
 mark('all31_cache_targets_masks_and_inverse_caps',caches=len(m['parents']))
 trainparents={r['parent'] for r in m['train_rows']};assert not trainparents&set(validation_parents)
 assert len(m['train_rows'])==2000 and len(m['validation_rows'])==30;mark('original_stream_size_and_parent_isolation')
 assert not torch.cuda.is_initialized();mark('no_CUDA_initialized')
 result=dict(status='PASS',checks=checks,torch_version=torch.__version__,torch_threads=1,model_calls=0,GPU_calls=0,noise_draws=0,
  config_sha256=sha(OUT/'loss_config.json'),cache_manifest_sha256=sha(OUT/'data/relative_cache_manifest.json'),
  caveats=['Global translations are handled by the retained coordinate MSE, not this edge term.',
   'Only joint frame rotation, not prediction-only rotation, is invariant.',
   'Arbitrary corner permutation under clipping is not an invariance claim.',
   'The target-relative term measures deformation error; it does not guarantee self-intersection-free output.'])
 write(OUT/'audits/cpu_math_checks.json',result);return result

if __name__=='__main__':
 r=run();print(json.dumps(dict(status=r['status'],checks=len(r['checks']),cache_manifest_sha256=r['cache_manifest_sha256'])))
