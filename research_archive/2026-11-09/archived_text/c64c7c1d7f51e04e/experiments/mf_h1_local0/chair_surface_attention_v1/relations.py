"""Finite known-triangle relations in fixed Gaussian units.

No model, GT, object identity or loss is accepted. Production arithmetic is
FP32 inside disabled autocast. FP64 and denser quadrature are explicit numerical
reference modes. Queries and known faces are chunked; quadrature nodes remain
inside each known face and never become attention source tokens.
"""
from __future__ import annotations
from functools import lru_cache
import math
import torch

MODES = ('MOMENT_XY', 'SURFACE_XY')
PRODUCTION_N = 8
REFERENCE_NS = (8, 64, 128)
QUERY_CHUNK = 48
KNOWN_CHUNK = 8
BANDWIDTH_FLOOR = 0.1
FEATURE_NAMES = ('mu_minus_x_x', 'mu_minus_x_y', 'mu_minus_x_z',
                 'Sigma_xx', 'Sigma_yy', 'Sigma_zz', 'Sigma_xy', 'Sigma_xz', 'Sigma_yz',
                 'log1p_negative_logmass', 't')


@lru_cache(maxsize=3)
def quadrature_integer_nodes(n=PRODUCTION_N):
    """Exact barycentric numerators with denominator 6*n; S3 symmetry testable.

    Each of n^2 equal-area reference subtriangles receives the three degree-2
    barycentric nodes (4,1,1)/6 and cyclic permutations. These are equal positive
    weights, not vertex samples. Integer construction avoids approximate grid
    deduplication and preserves the exact six-permutation node multiset.
    """
    if type(n) is not int or n not in REFERENCE_NS:
        raise ValueError('Only the frozen n=8 and reference n=64/128 rules are allowed')
    nodes = []
    def vertex(i, j):
        return (n-i-j, i, j)
    def add_triangle(vertices):
        for dominant in range(3):
            nodes.append(tuple(sum((4 if a == dominant else 1)*vertices[a][d] for a in range(3)) for d in range(3)))
    for i in range(n):
        for j in range(n-i):
            add_triangle((vertex(i,j), vertex(i+1,j), vertex(i,j+1)))
            if i+j <= n-2:
                add_triangle((vertex(i+1,j), vertex(i+1,j+1), vertex(i,j+1)))
    if len(nodes) != 3*n*n or any(sum(row) != 6*n or min(row) <= 0 for row in nodes):
        raise RuntimeError('Fixed reference-triangle subdivision identity failed')
    return tuple(sorted(nodes))


def barycentric_quadrature(n=PRODUCTION_N, *, dtype=torch.float32, device='cpu'):
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('Geometry quadrature must be FP32 or explicit FP64 reference')
    integer = quadrature_integer_nodes(n)
    bary = torch.tensor(integer, dtype=dtype, device=device)/float(6*n)
    weights = torch.full((3*n*n,), 1.0/(3*n*n), dtype=dtype, device=device)
    return bary, weights


def _canonical_corners(triangles):
    """Value-preserving lexicographic corner order for reproducible S3 sums."""
    result = triangles
    for axis in (2,1,0):
        order = torch.argsort(result[...,axis], dim=1, stable=True)
        result = torch.gather(result, 1, order[...,None].expand(-1,-1,3))
    return result


def _dtype(reference):
    return torch.float64 if reference else torch.float32


def _configuration(mode, n, query_chunk, known_chunk, sigma_z, reference):
    if mode not in MODES:
        raise ValueError('Unknown relation mode')
    if type(n) is not int or n not in REFERENCE_NS or (not reference and n != PRODUCTION_N):
        raise ValueError('Production uses exactly 192 points; n64/128 are references only')
    if type(query_chunk) is not int or type(known_chunk) is not int or query_chunk <= 0 or known_chunk <= 0:
        raise ValueError('Query/known chunks must be positive integers')
    if not reference and (query_chunk,known_chunk) != (QUERY_CHUNK,KNOWN_CHUNK):
        raise ValueError('Production chunk sizes are frozen at query48/known8')
    if not isinstance(sigma_z,(int,float)) or not math.isfinite(sigma_z) or sigma_z <= 0:
        raise ValueError('sigma_Z must be one globally known positive scalar')


def _static_geometry(triangles, mode, n):
    canonical = _canonical_corners(triangles)
    centroid = canonical.mean(dim=1)
    centered = canonical-centroid[:,None,:]
    covariance = torch.einsum('kai,kaj->kij',centered,centered)/12.
    nodes = None
    if mode == 'SURFACE_XY':
        bary,_ = barycentric_quadrature(n,dtype=triangles.dtype,device=triangles.device)
        nodes = torch.einsum('ra,kad->krd',bary,canonical)
    return dict(source_C=triangles.detach().clone(), mode=mode, n=n,
                centroid=centroid, uniform_covariance=covariance, nodes=nodes)


def _cache_matches(cache, triangles, mode, n):
    if cache is None or triangles.requires_grad:
        return False
    old = cache.get('source_C')
    return (cache.get('mode') == mode and cache.get('n') == n and isinstance(old,torch.Tensor)
            and old.device == triangles.device and old.dtype == triangles.dtype
            and old.shape == triangles.shape and torch.equal(old,triangles))


def triangle_statistics(queries, triangles, t, mode, *, n=PRODUCTION_N,
                        query_chunk=QUERY_CHUNK, known_chunk=KNOWN_CHUNK,
                        sigma_z=1.0, reference=False, static=None):
    """Return mu[Q,K,3], Sigma[Q,K,3,3], z[Q,K] and static geometry.

    Inputs are model coordinates; only this function applies the constant
    sigma_Z conversion. Sigma uses the centered weighted covariance directly,
    avoiding subtraction of large uncentered second moments. Zero-area unit
    fixtures use the finite barycentric push-forward, not a legal-area claim.
    """
    _configuration(mode,n,query_chunk,known_chunk,sigma_z,reference)
    if not isinstance(queries,torch.Tensor) or not isinstance(triangles,torch.Tensor):
        raise ValueError('Relations require tensor coordinates')
    if queries.ndim != 2 or queries.shape[-1] != 3 or triangles.ndim != 3 or triangles.shape[1:] != (3,3):
        raise ValueError('Require queries[Q,3] and known triangles[K,3,3]')
    if queries.device != triangles.device:
        raise ValueError('Queries and known triangles must share a device')
    dtype = _dtype(reference)
    with torch.autocast(device_type=queries.device.type, enabled=False):
        x = queries.to(dtype=dtype)/float(sigma_z)
        C = triangles.to(dtype=dtype)/float(sigma_z)
        time = torch.as_tensor(t,dtype=dtype,device=x.device)
        if time.numel() != 1 or not bool(torch.isfinite(time).all()) or not 0 <= float(time.detach()) <= 1:
            raise ValueError('Each relation batch sample needs one finite scalar t in [0,1]')
        time = time.reshape(())
        if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(C).all()):
            raise ValueError('Nonfinite valid free or known relation coordinate')
        if not _cache_matches(static,C,mode,n):
            static = _static_geometry(C,mode,n)
        qcount,kcount = len(x),len(C)
        if not qcount or not kcount:
            return dict(mu=x.new_empty((qcount,kcount,3)),Sigma=x.new_empty((qcount,kcount,3,3)),
                        z=x.new_empty((qcount,kcount)),static=static,t=time)
        s2 = (1-time).square()+BANDWIDTH_FLOOR**2
        means=[];covariances=[];masses=[]
        for qstart in range(0,qcount,query_chunk):
            current=x[qstart:qstart+query_chunk]
            mu_parts=[];cov_parts=[];z_parts=[]
            for kstart in range(0,kcount,known_chunk):
                stop=kstart+known_chunk
                if mode == 'MOMENT_XY':
                    c=static['centroid'][kstart:stop]
                    covariance=static['uniform_covariance'][kstart:stop]
                    mu=c[None].expand(len(current),-1,-1)
                    covariance=covariance[None].expand(len(current),-1,-1,-1)
                    z=-(current[:,None,:]-c[None,:,:]).square().sum(dim=-1)/(2*s2)
                else:
                    nodes=static['nodes'][kstart:stop]
                    # This bounded [query_chunk,known_chunk,192,3] tensor is
                    # never the full B x 3F x K x quadrature object.
                    delta=current[:,None,None,:]-nodes[None,:,:,:]
                    log_weight=-delta.square().sum(dim=-1)/(2*s2)-math.log(3*n*n)
                    z=torch.logsumexp(log_weight,dim=-1)
                    p=torch.softmax(log_weight,dim=-1)
                    mu=torch.einsum('qkr,krd->qkd',p,nodes)
                    residual=nodes[None,:,:,:]-mu[:,:,None,:]
                    covariance=torch.einsum('qkr,qkri,qkrj->qkij',p,residual,residual)
                mu_parts.append(mu);cov_parts.append(covariance);z_parts.append(z)
            means.append(torch.cat(mu_parts,dim=1))
            covariances.append(torch.cat(cov_parts,dim=1))
            masses.append(torch.cat(z_parts,dim=1))
        return dict(mu=torch.cat(means,dim=0),Sigma=torch.cat(covariances,dim=0),
                    z=torch.cat(masses,dim=0),static=static,t=time)


def relation_features(queries, statistics, *, sigma_z=1.0):
    """Pack exactly mu-x, covariance(xx yy zz xy xz yz), log1p(-z)_+, t."""
    mu,S,z=statistics['mu'],statistics['Sigma'],statistics['z']
    with torch.autocast(device_type=mu.device.type,enabled=False):
        x=queries.to(dtype=mu.dtype)/float(sigma_z)
        covariance=torch.stack((S[...,0,0],S[...,1,1],S[...,2,2],S[...,0,1],S[...,0,2],S[...,1,2]),dim=-1)
        mass=torch.log1p(torch.clamp(-z,min=0))[...,None]
        time=statistics['t'].expand(z.shape)[...,None]
        return torch.cat((mu-x[:,None,:],covariance,mass,time),dim=-1)


def build_relations(x, t, known_mask, valid_mask, mode, sigma_z=1.0, *,
                    n=PRODUCTION_N, query_chunk=QUERY_CHUNK, known_chunk=KNOWN_CHUNK,
                    static_cache=None, reference=False):
    """One shared R per model forward; four attention layers reuse this list.

    Each batch entry has free_face_indices[F], known_indices[K], R[F,3,K,11].
    known and invalid query rows are omitted. Empty known sets yield a zero-size
    K dimension (the caller applies no correction), with no empty softmax.
    Optional static_cache is a caller-owned list, [] or [None]*B, updated in
    place only for coordinates without gradients; C changes refresh its entry.
    """
    _configuration(mode,n,query_chunk,known_chunk,sigma_z,reference)
    if x.ndim != 3 or x.shape[-1] != 9 or known_mask.shape != x.shape[:2] or valid_mask.shape != x.shape[:2]:
        raise ValueError('Require coordinates[B,N,9] and corresponding face masks')
    if known_mask.dtype != torch.bool or valid_mask.dtype != torch.bool:
        raise ValueError('Known and valid masks must be Boolean')
    if known_mask.device != x.device or valid_mask.device != x.device or bool((known_mask & ~valid_mask).any()):
        raise ValueError('Known/valid masks must be colocated and known implies valid')
    times=torch.as_tensor(t,device=x.device)
    if times.numel() not in (1,len(x)):
        raise ValueError('Require one t per batch item (or one shared scalar)')
    times=times.reshape(-1).expand(len(x)) if times.numel()==1 else times.reshape(-1)
    if static_cache is not None:
        if not isinstance(static_cache,list):
            raise ValueError('Static relation cache must be an explicit list')
        if len(static_cache)!=len(x):
            static_cache[:]=[None]*len(x)
    result=[]
    for batch in range(len(x)):
        free=torch.nonzero(valid_mask[batch] & ~known_mask[batch],as_tuple=False).flatten()
        known=torch.nonzero(valid_mask[batch] & known_mask[batch],as_tuple=False).flatten()
        queries=x[batch,free].reshape(-1,3)
        C=x[batch,known].reshape(-1,3,3)
        cached=None if static_cache is None else static_cache[batch]
        statistics=triangle_statistics(queries,C,times[batch],mode,n=n,query_chunk=query_chunk,
            known_chunk=known_chunk,sigma_z=sigma_z,reference=reference,static=cached)
        if static_cache is not None:
            static_cache[batch]=None if C.requires_grad else statistics['static']
        R=relation_features(queries,statistics,sigma_z=sigma_z).reshape(len(free),3,len(known),11)
        result.append(dict(free_face_indices=free,known_indices=known,R=R))
    return result


compute_relations = build_relations
