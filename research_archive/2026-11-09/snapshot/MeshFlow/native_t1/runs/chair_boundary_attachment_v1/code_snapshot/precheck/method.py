"""C-only boundary input and training-only exact corresponding-segment UCorr.

No source topology is accepted by the model input branch. Training identities
are separately mapped to actual post-OT slots; old source files are read-only.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
from types import MethodType
import numpy as np
from native_t1 import runtime as _runtime
import torch
from native_t1.model import assert_native_runtime, native_block_forward, native_final_forward, _corners
from native_t1.context_geometry import inject_context

SPEC = dict(schema='chair_boundary_attachment_method_v1',
    boundary='exact_FP32_C_only_undirected_edge_incidence_1',
    boundary_nearly_constant_fraction_thresholds=[.01,.99],
    boundary_injection='after_coordinate_role_Geo_before_first_block',
    new_parameter='e_boundary',new_parameter_shape=[768],initialization='exact_FP32_zero',
    seam='mean_equal_edge((r0*r0).sum+(r0*r1).sum+(r1*r1).sum)/3',
    scale='augmented_full_GT_bbox_diagonal', gate='t>=0.5', batch_divisor=8,
    correspondence='original_degree2_trusted_source_edge_to_actual_post_OT_face_and_corner_slots',
    GT_detached=True,corner_degree_renormalization=False,quadrature=False)


def boundary_marks(x, known_mask, valid_mask):
    """FP32 x[B,N,9] -> bool[B,N,3] and detached C-only diagnostics.

    Only valid known coordinates are read. Exact zero-length/nonfinite C edges
    fail as in the original Geo contract; no face is deleted or welded.
    """
    if x.ndim != 3 or x.shape[-1] != 9 or x.dtype != torch.float32:
        raise ValueError('Boundary marking requires FP32 [B,N,9] actual input')
    for mask in (known_mask,valid_mask):
        if mask.shape != x.shape[:2] or mask.dtype != torch.bool or mask.device != x.device:
            raise ValueError('Separate colocated bool known/valid masks required')
    if bool((known_mask & ~valid_mask).any()): raise ValueError('Known must be valid')
    marks=torch.zeros((*x.shape[:2],3),device=x.device,dtype=torch.bool); rows=[]
    with torch.no_grad(), torch.autocast(device_type=x.device.type,enabled=False):
        for b in range(len(x)):
            slots=torch.where(known_mask[b] & valid_mask[b])[0]
            if not len(slots):raise ValueError('At least one valid known face required')
            C=x[b,slots].reshape(-1,3,3).detach()
            if not bool(torch.isfinite(C).all()):raise ValueError('Nonfinite known coordinate')
            starts=C[:,[0,1,2]].reshape(-1,3); ends=C[:,[1,2,0]].reshape(-1,3)
            if bool((starts==ends).all(-1).any()):raise ValueError('Known triangle has a zero-length edge')
            # Stable lexicographic endpoint order, with exact numeric equality.
            less=(starts[:,0]<ends[:,0]) | ((starts[:,0]==ends[:,0]) & ((starts[:,1]<ends[:,1]) | ((starts[:,1]==ends[:,1]) & (starts[:,2]<=ends[:,2]))))
            lo=torch.where(less[:,None],starts,ends); hi=torch.where(less[:,None],ends,starts)
            keys=torch.cat((lo,hi),1); keys=torch.where(keys==0,torch.zeros_like(keys),keys)
            _, inverse,counts=torch.unique(keys,dim=0,return_inverse=True,return_counts=True)
            opened=(counts[inverse]==1).reshape(-1,3)
            corner=torch.stack((opened[:,0]|opened[:,2],opened[:,0]|opened[:,1],opened[:,1]|opened[:,2]),1)
            marks[b,slots]=corner
            fraction=float(corner.float().mean())
            rows.append(dict(open_edges=int((counts==1).sum()),shared_edges=int((counts==2).sum()),
                ambiguous_edges=int((counts>2).sum()),unique_edges=len(counts),known_faces=len(slots),
                marked_known_corners=int(corner.sum()),known_corners=3*len(slots),marked_known_corner_fraction=fraction,
                nearly_constant=fraction<=.01 or fraction>=.99,exactly_constant=fraction in (0.,1.)))
    return marks,rows


def inject_boundary(hidden, marks, vector):
    """No zero-parameter shortcut: zero initialization still has gradients."""
    if hidden.ndim!=3 or marks.shape[:2]!=(hidden.shape[0],hidden.shape[1]//3) or marks.shape[-1]!=3:
        raise ValueError('Corner marker shape differs from hidden tokens')
    if vector.ndim!=1 or len(vector)!=hidden.shape[-1] or vector.dtype!=torch.float32:
        raise ValueError('Require FP32 vector matching actual hidden width')
    return hidden + marks.reshape(hidden.shape[0],hidden.shape[1],1).to(hidden.dtype)*vector.to(hidden.dtype)


def attach_boundary(model, enabled):
    """Call only after original model strict-load; adds exactly one vector."""
    if hasattr(model,'_boundary_attachment_enabled') or hasattr(model,'e_boundary'):
        raise ValueError('Boundary attachment already configured')
    model._boundary_attachment_enabled=bool(enabled)
    before=set(dict(model.named_parameters()))
    if enabled:
        width=int(model.backbone.hidden_size)
        if width!=768:raise ValueError('Registered hidden width must be 768')
        anchor=next(model.parameters())
        if anchor.dtype!=torch.float32:raise ValueError('Strict-loaded FP32 model required')
        model.register_parameter('e_boundary',torch.nn.Parameter(torch.zeros(width,device=anchor.device,dtype=torch.float32),requires_grad=bool(model.experiment_trainable)))
        model.forward=MethodType(_boundary_forward,model)
    added=sorted(set(dict(model.named_parameters()))-before)
    metadata=dict(spec=SPEC,enabled=bool(enabled),new_parameter_names=added,
        new_parameter_count=sum(dict(model.named_parameters())[k].numel() for k in added),
        added_after_strict_load=True,original_model_forward_reused=not enabled)
    model.boundary_attachment_metadata=metadata;model.boundary_observations=[]
    return metadata


def seam_targets(sample):
    """Map trustworthy original incidence to actual post-OT prediction slots.

    sample['_seam_source'] is training-only. Source face/corner permutations
    already include inverse nested OT and full randomized face order.
    """
    source=sample['_seam_source']; g=sample['_geometry']
    V=np.asarray(g['V'],np.float32);F=np.asarray(g['F'],np.int64)
    valid=np.asarray(sample['valid_mask'],bool); known=np.asarray(sample['known_mask'],bool)
    ids=np.asarray(sample['source_face_ids'],np.int64); corners=np.asarray(sample['corner_permutations'],np.int64)
    slots=np.flatnonzero(valid); n=len(F)
    if np.any(known & ~valid) or sorted(ids[valid].tolist())!=list(range(n)):
        raise ValueError('Source faces must form full valid partition')
    if not np.array_equal(np.sort(corners[valid],axis=1),np.tile(np.arange(3),(n,1))):raise ValueError('Source corner map is not bijective')
    source_slot={int(ids[s]):int(s) for s in slots}; known_source={int(ids[s]) for s in slots if known[s]}
    incidence,original=defaultdict(list),defaultdict(list)
    for target,faces in ((incidence,source['source_faces']),(original,source['original_source_faces'])):
        for face,row in enumerate(faces):
            for a,b in ((0,1),(1,2),(2,0)):target[tuple(sorted((int(row[a]),int(row[b]))))].append(face)
    retained=np.asarray(source['retained_to_original_face'],np.int64)
    allow={tuple(map(int,e)) for e in np.asarray(source['trusted_interface']['edge_source_vertex_ids']).reshape(-1,2)}
    if len(allow)!=len(source['trusted_interface']['edge_source_vertex_ids']):raise ValueError('Duplicate trusted interface identity')
    candidate=0;rejected=Counter();edges=[];free_faces=[];free_corners=[];known_faces=[];known_corners=[];endpoints=[]
    for edge,faces in sorted(incidence.items()):
        kin=[f for f in faces if f in known_source];fre=[f for f in faces if f not in known_source]
        if not kin or not fre:continue
        candidate+=1
        if len(faces)!=2 or len(kin)!=1 or len(fre)!=1 or len(original[edge])!=2 or set(original[edge])!={int(retained[f]) for f in faces}:
            rejected['nonmanifold_or_original_incidence']+=1;continue
        if edge not in allow:rejected['not_trusted_by_original_constructor']+=1;continue
        points=V[np.asarray(edge)]
        if not np.isfinite(points).all():raise ValueError('Nonfinite GT seam endpoints')
        if np.array_equal(points[0],points[1]):rejected['zero_length']+=1;continue
        pair_slots=[];pair_corners=[]
        for source_face in (kin[0],fre[0]):
            slot=source_slot[source_face]; vertex_ids=F[source_face][corners[slot]]
            matched=[np.flatnonzero(vertex_ids==v) for v in edge]
            if any(len(hit)!=1 for hit in matched):raise ValueError('Ambiguous source endpoint correspondence')
            pair_slots.append(slot);pair_corners.append([int(hit[0]) for hit in matched])
        ks,fs=pair_slots;kc,fc=pair_corners
        if not known[ks] or known[fs]:raise ValueError('Seam mapped outside known/free roles')
        actual_known=np.asarray(sample['context']).reshape(-1,3,3)[ks,kc]
        actual_free=np.asarray(sample['x1']).reshape(-1,3,3)[fs,fc]
        if actual_known.tobytes()!=points.tobytes() or actual_free.tobytes()!=points.tobytes():
            raise ValueError('Post-OT seam indices do not reconstruct exact GT endpoints')
        edges.append(edge);known_faces.append(ks);free_faces.append(fs);known_corners.append(kc);free_corners.append(fc);endpoints.append(points.copy())
    L=float(g['bbox_L'])
    if not np.isfinite(L) or L<=0:raise ValueError('Positive augmented full GT bbox diagonal required')
    uses=Counter((f,c) for f,pair in zip(free_faces,free_corners) for c in pair)
    result=dict(schema='equal_edge_analytic_UCorr_target_v1',free_faces=np.asarray(free_faces,np.int64),
        free_corners=np.asarray(free_corners,np.int64).reshape(-1,2),known_faces=np.asarray(known_faces,np.int64),
        known_corners=np.asarray(known_corners,np.int64).reshape(-1,2),GT_endpoints=np.asarray(endpoints,np.float32).reshape(-1,2,3),
        edge_source_vertex_ids=np.asarray(edges,np.int64).reshape(-1,2),bbox_L=L,
        counts=dict(candidate_interface_edges=candidate,eligible_interface_edges=len(edges),excluded=dict(rejected),
            reused_free_corners=sum(v>1 for v in uses.values()),max_corner_reuse=max(uses.values(),default=0),
            corner_reuse_histogram=dict(sorted(Counter(uses.values()).items())),
            equal_edge_weight=1/len(edges) if edges else 0.,corner_incidence_weights=[v/len(edges) for v in sorted(uses.values())] if edges else []),
        status='PASS' if edges else 'NO_VALID_SEAM',GT_and_C_detached=True)
    h=hashlib.sha256()
    for k in ('free_faces','free_corners','known_faces','known_corners','GT_endpoints','edge_source_vertex_ids'):
        a=result[k];h.update(k.encode());h.update(str((a.dtype,a.shape)).encode());h.update(a.tobytes())
    h.update(repr(L).encode());result['target_sha256']=h.hexdigest()
    return result


def analytic_ucorr(predicted, fixed, scale):
    """Exact corresponding-segment integral; FP32 production/FP64 CPU checks."""
    if predicted.dtype not in (torch.float32,torch.float64) or predicted.ndim!=3 or predicted.shape[1:]!=(2,3):
        raise ValueError('Expected floating [eligible_edges,2,3] endpoints')
    fixed=torch.as_tensor(fixed,device=predicted.device,dtype=predicted.dtype).detach()
    L=torch.as_tensor(scale,device=predicted.device,dtype=predicted.dtype).detach()
    if fixed.shape!=predicted.shape or L.numel()!=1 or not bool(torch.isfinite(L) & (L>0)):raise ValueError('Invalid detached GT/scale')
    if not bool(torch.isfinite(predicted).all() and torch.isfinite(fixed).all()):raise FloatingPointError('Nonfinite seam geometry')
    if not len(predicted):return predicted.sum()*0
    with torch.autocast(device_type=predicted.device.type,enabled=False):
        r=(predicted-fixed)/L;r0,r1=r[:,0],r[:,1]
        return ((r0.square().sum(-1)+r1.square().sum(-1)+(r0*r1).sum(-1))/3).mean()


def seam_loss(velocity,batch_xt,t,sample):
    """Return raw E (ungated, no /8) and diagnostics. Caller owns gate/lambda."""
    if velocity.ndim==3:
        if len(velocity)!=1 or batch_xt.shape!=velocity.shape:raise ValueError('One microbatch only')
        velocity=velocity[0];batch_xt=batch_xt[0]
    if velocity.ndim!=2 or velocity.shape[-1]!=9 or batch_xt.shape!=velocity.shape or velocity.dtype not in (torch.float32,torch.float64):
        raise ValueError('Require same-layout FP32 production velocity and X_t')
    target=sample.get('_seam_targets')
    if target is None:target=seam_targets(sample)
    time=torch.as_tensor(t,device=velocity.device,dtype=velocity.dtype).detach().reshape(())
    if not bool(torch.isfinite(time) & (time>=0) & (time<=1)):raise ValueError('Invalid flow time')
    faces=torch.as_tensor(target['free_faces'],device=velocity.device,dtype=torch.long)
    corners=torch.as_tensor(target['free_corners'],device=velocity.device,dtype=torch.long)
    with torch.autocast(device_type=velocity.device.type,enabled=False):
        endpoints=(batch_xt.detach().reshape(-1,3,3)[faces[:,None],corners]+(1-time)*velocity.reshape(-1,3,3)[faces[:,None],corners])
        value=analytic_ucorr(endpoints,target['GT_endpoints'],target['bbox_L'])
    return value,dict(status=target['status'],gate=bool(time>=.5),counts=target['counts'],target_sha256=target['target_sha256'],GT_and_C_detached=True)


def _boundary_forward(self, x, t, y, valid_mask=None, known_mask=None, *, observer=None):
    assert_native_runtime(self, coordinates=(x, t))
    if x.ndim != 3 or x.shape[-1] != 9:
        raise ValueError('Expected FP32 x[B,padded_N,9]')
    batch, faces, _ = x.shape
    if t.shape != (batch,) or y.shape != (batch,) or y.dtype != torch.int64:
        raise ValueError('Expected free t[B] and integer total N[B]')
    if not bool(torch.isfinite(x).all() and torch.isfinite(t).all()):
        raise ValueError('Nonfinite coordinates/time')
    if bool(((t < 0) | (t > 1)).any()):
        raise ValueError('Free time outside [0,1]')
    for mask in (valid_mask, known_mask):
        if mask is None or mask.dtype != torch.bool or mask.shape != (batch, faces) or mask.device != x.device:
            raise ValueError('Colocated separate bool[B,padded_N] masks required')
    if bool((known_mask & ~valid_mask).any()):
        raise ValueError('Known faces must be valid')
    if not torch.equal(valid_mask.sum(1), y) or bool(((y < 128) | (y > 256)).any()):
        raise ValueError('y must count real faces only, N128..256')
    if bool((known_mask.sum(1) == 0).any()) or bool(((valid_mask & ~known_mask).sum(1) == 0).any()):
        raise ValueError('Each sample requires known and free faces')
    _, hidden = self.backbone.x_embedder(x)
    hidden = hidden.view(batch, faces * 3, -1)
    role = self.role_embedding[known_mask.long()].to(hidden.dtype)
    hidden = hidden + _corners(role)
    if self.context_encoder is not None:
        delta = self.context_encoder(x, known_mask, valid_mask, observer=self.context_observer)
        hidden = inject_context(hidden, delta, known_mask, observer=self.context_observer)
    marks, diagnostics = boundary_marks(x, known_mask, valid_mask)
    hidden = inject_boundary(hidden, marks, self.e_boundary)
    self.boundary_observations = diagnostics
    if observer is not None:
        observer(dict(site='boundary', hidden=hidden, marks=marks, diagnostics=diagnostics))
    c_free = self.backbone.t_embedder(t)
    c_known = self.backbone.t_embedder(torch.ones_like(t))
    count = self.backbone.y_embedder(y // self.backbone.face_bin, False)
    c_free, c_known = c_free + count, c_known + count
    face_time = torch.where(known_mask, torch.ones_like(t[:, None]), t[:, None])
    if observer is not None:
        observer(dict(site='embedding', hidden=hidden, role=role,
            known_mask=known_mask, valid_mask=valid_mask, face_time=face_time,
            c_free=c_free, c_known=c_known))
    for index, block in enumerate(self.backbone.layers, 1):
        hidden = native_block_forward(block, hidden, c_free, c_known, known_mask,
            valid_mask, observer=observer, layer_index=index, face_time=face_time)
    output = native_final_forward(self.backbone.final_layer, hidden, c_free,
        c_known, known_mask, observer=observer, face_time=face_time)
    # Raw known velocity is intentionally unconstrained. Integration owns C.
    return output.view(batch, faces, 9)
