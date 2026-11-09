"""Frozen corner-domain intervention; no changes to sites, normals or quadrics.

Fixed delegates to the original safe extractor. Enclosing pre-appends the same
8 anchor attributes, replacing only their coordinates, then invokes the same
SafeMeshFromPoNQ with add_corners=False. The vendored downstream anchor rules
use the final 8 indices and do not inspect add_corners again.
"""
import copy
import numpy as np
import torch
from safe_extract import extract, SafeMeshFromPoNQ, SafeExtractionError
from mesh_tools import SIGNS

DOMAIN_VERSION='PONQ_DOMAIN_INTERVENTION_V1'
MARGIN_FRACTION=.1
ZERO_AXIS_MARGIN_FRACTION=1e-6
GRID_SCALE=32


def corner_domain(v,domain='fixed'):
    v=np.asarray(v,dtype=np.float64)
    if v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or not np.isfinite(v).all():
        raise SafeExtractionError('Domain needs at least four finite 3D sites')
    if domain not in ['fixed','enclosing']:raise ValueError('Unknown corner domain: '+str(domain))
    lower=v.min(axis=0);upper=v.max(axis=0);extent=upper-lower
    if domain=='fixed':
        corners=np.asarray(SIGNS,dtype=np.float64).copy();margin=None
    else:
        maximum=float(extent.max())
        if not np.isfinite(maximum) or maximum<=0:raise SafeExtractionError('Zero/nonfinite all-axis site extent; enclosing domain is undefined')
        margin=np.where(extent>0,MARGIN_FRACTION*extent,ZERO_AXIS_MARGIN_FRACTION*maximum)
        lo=lower-margin;hi=upper+margin
        if not np.isfinite(lo).all() or not np.isfinite(hi).all() or not np.all(lo<lower) or not np.all(hi>upper):
            raise SafeExtractionError('Enclosing margin is not finite and strictly positive in representable coordinates')
        corners=np.where(np.asarray(SIGNS)>0,hi[None,:],lo[None,:])
    bbox_lo=corners.min(axis=0);bbox_hi=corners.max(axis=0)
    audit=dict(domain_backend_version=DOMAIN_VERSION,extraction_domain=domain,grid_scale=GRID_SCALE,
        site_aabb_min=lower.tolist(),site_aabb_max=upper.tolist(),site_extent=extent.tolist(),
        anchor_aabb_min=bbox_lo.tolist(),anchor_aabb_max=bbox_hi.tolist(),anchor_coordinates=corners.tolist(),
        axis_margin=None if margin is None else margin.tolist(),margin_fraction=MARGIN_FRACTION,
        zero_axis_margin_fraction=ZERO_AXIS_MARGIN_FRACTION,
        input_sites_outside_actual_domain=int(np.any((v<bbox_lo)|(v>bbox_hi),axis=1).sum()),
        input_sites_outside_actual_fixed_cube=int(np.any((v<np.min(SIGNS,axis=0))|(v>np.max(SIGNS,axis=0)),axis=1).sum()),
        legacy_half_cube_exceedance_count=int(np.any(abs(v)>.5,axis=1).sum()),
        position_repair_performed=False,normal_normalization=False,quadric_change_performed=False,
        anchor_normals_policy='Unchanged official -SIGNS/sqrt(sum(SIGNS**2)); only anchor coordinates change',
        anchor_Q_and_eigs_policy='Unchanged official identity4 Q and ones3 eigs')
    return corners,audit


def extract_domain(params,domain='fixed'):
    """Return (v,f,input_vertex_ids,trace_arrays,audit), preserving original guards.

    Failure raises SafeExtractionError with original trace/audit plus domain
    metadata when available. grid_scale is fixed at 32 for both domains.
    """
    corners,domain_audit=corner_domain(params['v'],domain)
    try:
        if domain=='fixed':
            v,f,ids,trace,audit=extract(params,grid_scale=GRID_SCALE)
        else:
            arrays={};count=len(params['v'])
            for key,shape in dict(v=(count,3),n=(count,3),Q=(count,4,4),eigs=(count,3)).items():
                value=np.array(params[key],dtype=np.float64,copy=True)
                if value.shape!=shape or not np.isfinite(value).all():raise SafeExtractionError('Invalid shape or nonfinite input: '+key)
                arrays[key]=value
            if np.any(arrays['eigs'][:,0]<=0):raise SafeExtractionError('Positive normalization eigenvalues required')
            if np.any(np.linalg.norm(arrays['n'],axis=1)==0):raise SafeExtractionError('Zero input normal')
            # Match the exact official concatenation arithmetic and anchor dtypes.
            x=torch.from_numpy(arrays['v']);q=torch.from_numpy(arrays['Q']);n=torch.from_numpy(arrays['n']);e=torch.from_numpy(arrays['eigs'])
            x=torch.cat((x,torch.tensor(corners,device=x.device)))
            q=torch.cat((q,torch.eye(4,device=x.device).repeat(8,1,1)))
            n=torch.cat((n,-torch.tensor(SIGNS/np.sqrt((SIGNS**2).sum(-1,keepdims=True)),device=x.device)))
            e=torch.cat((e,torch.ones(8,3,device=x.device)))
            backend=SafeMeshFromPoNQ(x,e,q,n,grid_scale=GRID_SCALE,add_corners=False,correct_tet_color=True,compute_mincut=True)
            v,f,ids=backend.get_surface(return_indices=True)
            backend.trace_arrays['surface_input_vertex_ids']=ids.copy()
            backend.trace_arrays['surface_faces_input_ids']=ids[f].copy()
            backend.audit.update(status='EXTRACTED',input_elements=count,output_vertices=len(v),output_faces=len(f),
                anchor_vertices_in_output=int(np.sum(ids>=count)),arccos_clipped_count=sum(a['clipped_count'] for a in backend.audit['arccos_calls']))
            trace={k:np.array(a,copy=True) for k,a in backend.trace_arrays.items()};audit=copy.deepcopy(backend.audit)
        audit.update(domain_audit)
        trace['domain_corner_points']=corners.copy()
        return v.copy(),f.copy(),ids.copy(),trace,audit
    except Exception as exc:
        audit=copy.deepcopy(getattr(exc,'audit',getattr(locals().get('backend'), 'audit',{})))
        trace={k:np.array(a,copy=True) for k,a in getattr(exc,'trace_arrays',getattr(locals().get('backend'),'trace_arrays',{})).items()}
        audit.update(domain_audit);audit['status']='FAILED';trace['domain_corner_points']=corners.copy()
        raise SafeExtractionError(str(exc),audit,trace) from exc