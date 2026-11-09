"""Literal localized oriented boundary-current loss, training supervision only.

No model imports, nearest-point matching, welding, GT inference input or repairs.
Tau is the length-bearing directed edge vector confirmed by the user.
"""
from dataclasses import dataclass
import torch

SIGMA_FRACTIONS=(0.05,0.10)

@dataclass(frozen=True)
class BoundaryTarget:
    known_midpoints: torch.Tensor
    known_vectors: torch.Tensor
    free_reference_normals: torch.Tensor
    probes: torch.Tensor
    bbox_L: float
    context_audit: dict


def face_normals(triangles):
    return torch.linalg.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0],dim=-1)


def oriented_halfedges(triangles, reference_normals):
    if triangles.ndim!=3 or tuple(triangles.shape[1:])!=(3,3):
        raise ValueError('Expected [faces,3,3]')
    refs=reference_normals.detach().to(device=triangles.device,dtype=triangles.dtype)
    if refs.shape!=(len(triangles),3) or not bool(torch.isfinite(refs).all()):
        raise ValueError('GT normal labels must match target slots')
    if len(refs) and bool((refs.square().sum(-1)==0).any()):
        raise ValueError('Degenerate GT normal has no trusted orientation')
    dot=(face_normals(triangles)*refs).sum(-1)
    # Literal requested branch: no differentiable normal surrogate or tie repair.
    reverse=dot<0
    signs=torch.where(reverse,-torch.ones_like(dot),torch.ones_like(dot))
    other=triangles.roll(-1,dims=1)
    midpoint=(triangles+other)/2
    vectors=(other-triangles)*signs[:,None,None]
    return midpoint.reshape(-1,3),vectors.reshape(-1,3),reverse,dot==0


def prepare_target(known, known_ids, known_reference_normals,
                   free_reference_normals, probes, bbox_L):
    # All GT/context/query quantities are immutable supervision.
    known=known.detach().clone()
    ids=known_ids.detach().cpu().tolist()
    if known.shape!=(len(ids),3,3) or any(len(x)!=3 for x in ids):
        raise ValueError('Known source IDs must follow known corners')
    if len({int(g) for tri in ids for g in tri}) and min(g for tri in ids for g in tri)<0:
        raise ValueError('Invalid trusted vertex IDs')
    if not bbox_L>0 or probes.shape!=(64,3):
        raise ValueError('Require positive GT bbox diagonal and exactly 64 fixed probes')
    mid,vec,reverse,ties=oriented_halfedges(known,known_reference_normals)
    incidence={}
    for i,tri in enumerate(ids):
        for a in range(3):
            u,v=int(tri[a]),int(tri[(a+1)%3])
            if u==v:raise ValueError('Degenerate known source edge')
            if bool(reverse[i]):u,v=v,u
            incidence.setdefault(tuple(sorted((u,v))),[]).append((3*i+a,u,v))
    bad=[k for k,occ in incidence.items() if len(occ)==2 and occ[0][1:]==occ[1][1:]]
    over=[k for k,occ in incidence.items() if len(occ)>2]
    # Observe these pathologies, never silently weld or fix source geometry.
    take=[occ[0][0] for occ in incidence.values() if len(occ)==1]
    index=torch.tensor(take,dtype=torch.long,device=known.device)
    return BoundaryTarget(mid[index].detach(),vec[index].detach(),
        free_reference_normals.detach().clone(),probes.detach().clone(),float(bbox_L),
        dict(faces=len(known),boundary_edges=len(take),same_direction_internal_edges=len(bad),
             overincident_edges=len(over),normal_ties=int(ties.sum()),reversed_faces=int(reverse.sum())))


def boundary_loss(predicted_free, target):
    if predicted_free.dtype not in (torch.float32,torch.float64):
        raise TypeError('Geometry loss must use FP32 or FP64, not BF16')
    if not bool(torch.isfinite(predicted_free).all()):
        raise FloatingPointError('Nonfinite free prediction')
    mid,vec,reverse,ties=oriented_halfedges(predicted_free,target.free_reference_normals)
    convert=lambda t:t.detach().to(device=predicted_free.device,dtype=predicted_free.dtype)
    mid=torch.cat((convert(target.known_midpoints),mid),dim=0)
    vec=torch.cat((convert(target.known_vectors),vec),dim=0)
    probes=convert(target.probes)
    distances_sq=(probes[:,None,:]-mid[None,:,:]).square().sum(-1)
    per_scale=[];residual=[]
    for fraction in SIGMA_FRACTIONS:
        weights=torch.exp(-distances_sq/(2*(fraction*target.bbox_L)**2))
        B=(weights[:,:,None]*vec[None,:,:]).sum(1)
        per_scale.append(B.square().sum(-1).mean()/(target.bbox_L**2))
        residual.append(B)
    losses=torch.stack(per_scale)
    return dict(loss=losses.mean(),per_scale=losses,residuals=torch.stack(residual),
        free_halfedges=3*len(predicted_free),known_boundary_edges=len(target.known_vectors),
        reversed_faces=int(reverse.sum()),normal_alignment_ties=int(ties.sum()))


def uniform_probes(reference_geometry, seed=20261001):
    points=reference_geometry.detach().reshape(-1,3)
    low=points.min(0).values;high=points.max(0).values
    generator=torch.Generator(device='cpu').manual_seed(seed)
    u=torch.rand((64,3),generator=generator,dtype=points.dtype)
    return (low.cpu()+u*(high-low).cpu()).detach(),float(torch.linalg.vector_norm(high-low))
