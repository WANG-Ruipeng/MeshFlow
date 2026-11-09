"""Deterministic geometry-only operations. Subdivision is NOT Loop smoothing."""
from __future__ import annotations
import collections
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from numba import njit

def topology(v,f):
    e=np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
    ue,iv,ct=np.unique(e,axis=0,return_inverse=True,return_counts=True)
    de=np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);sgn=np.where(de[:,0]<de[:,1],1,-1)
    oriented=np.bincount(iv,weights=sgn,minlength=len(ue))
    used=np.unique(f);gr=coo_matrix((np.ones(2*len(ue)),(np.r_[ue[:,0],ue[:,1]],np.r_[ue[:,1],ue[:,0]])),shape=(len(v),len(v))).tocsr()
    components=connected_components(gr[used][:,used],directed=False,return_labels=False)
    links=collections.defaultdict(list)
    for a,b,c in f:
        links[int(a)].append((int(b),int(c)));links[int(b)].append((int(c),int(a)));links[int(c)].append((int(a),int(b)))
    badlink=0
    for a,es in links.items():
        adj=collections.defaultdict(list)
        for b,c in es:adj[b].append(c);adj[c].append(b)
        visited=set();todo=[next(iter(adj))]
        while todo:
            x=todo.pop()
            if x in visited:continue
            visited.add(x);todo.extend(adj[x])
        badlink+=int(any(len(n)!=2 for n in adj.values()) or len(visited)!=len(adj) or a in adj)
    area2=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)
    diag=np.linalg.norm(np.ptp(v,axis=0));deg=int((area2<=max(diag**2,1e-300)*1e-14).sum())
    duplicates=len(f)-len(np.unique(np.sort(f,axis=1),axis=0));chi=len(used)-len(ue)+len(f)
    ok=bool(np.all(ct==2) and np.all(oriented==0) and badlink==0 and components==1 and chi==2 and deg==0 and duplicates==0)
    return {'components':int(components),'euler':int(chi),'boundary_edges':int((ct==1).sum()),'nonmanifold_edges':int((ct>2).sum()),'bad_vertex_links':badlink,'orientation_errors':int(((ct==2)&(oriented!=0)).sum()),'duplicate_faces':duplicates,'degenerate_faces_tolerance':deg,'closed_oriented_g0':ok}

def bisect_to_faces(vertices,faces,target=512):
    """Split both incident triangles on globally longest edge; deterministic lex tie.
    No vertex moves, projections, flips, deleted distinct surfaces, or capping.
    Carries per-output-face source-face provenance for independent preservation checks.
    """
    v=np.asarray(vertices,np.float64).copy();f=np.asarray(faces,np.int64).copy()
    if target<len(f) or (target-len(f))%2:raise ValueError('Need target>=input and even delta')
    if not topology(v,f)['closed_oriented_g0']:raise ValueError('Requires clean closed oriented g0 input; no silent repair')
    parent=np.arange(len(f));steps=[]
    while len(f)<target:
        edges=np.unique(np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
        lens=((v[edges[:,0]]-v[edges[:,1]])**2).sum(1)
        e=edges[int(np.argmax(lens))];a,b=map(int,e)
        hit=np.where(np.sum(np.isin(f,[a,b]),axis=1)==2)[0]
        if len(hit)!=2:raise ValueError('Edge no longer has two incident faces')
        mid=len(v);v=np.vstack([v,(v[a]+v[b])/2.])
        additions=[];pa=[]
        for h in hit:
            tri=f[h]
            k=next(k for k in range(3) if {int(tri[k]),int(tri[(k+1)%3])}=={a,b})
            x,y,z=map(int,[tri[k],tri[(k+1)%3],tri[(k+2)%3]])
            additions.extend([[x,mid,z],[mid,y,z]]);pa.extend([int(parent[h])]*2)
        keep=np.ones(len(f),bool);keep[hit]=False
        f=np.vstack([f[keep],np.asarray(additions,np.int64)]);parent=np.r_[parent[keep],pa]
        steps.append([a,b,mid])
    if len(f)!=target or not topology(v,f)['closed_oriented_g0']:raise RuntimeError('Subdivision topology failed')
    cert=check_surface_lineage(vertices,faces,v,f,parent)
    return v,f,parent,np.asarray(steps,np.int64).reshape(-1,3),cert

def check_surface_lineage(v0,f0,v,f,parent,tolerance=2e-11):
    tr=np.asarray(v0,np.float64)[f0];base=tr[:,0];e=tr[:,1]-base;g=tr[:,2]-base;n=np.cross(e,g)
    ar=np.linalg.norm(n,axis=1);unit=n/ar[:,None];q=v[f];pp=parent
    rel=q-base[pp,None];diag=np.linalg.norm(np.ptp(v0,axis=0))
    plane=np.abs(np.einsum('nij,nj->ni',rel,unit[pp]))/diag
    ee=(e*e).sum(1)[pp];gg=(g*g).sum(1)[pp];eg=(e*g).sum(1)[pp];det=ee*gg-eg*eg
    we=np.einsum('nkj,nj->nk',rel,e[pp]);wg=np.einsum('nkj,nj->nk',rel,g[pp])
    b=(we*gg[:,None]-wg*eg[:,None])/det[:,None];c=(wg*ee[:,None]-we*eg[:,None])/det[:,None]
    signed=np.einsum('ij,ij->i',np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),unit[pp])
    area=np.bincount(pp,weights=signed,minlength=len(f0));err=np.max(abs(area-ar))/max(ar.max(),1e-300)
    bary_min=float(np.minimum(np.minimum(b,c),1-b-c).min())
    check={'plane_distance_bbox_max':float(plane.max()),'barycentric_min':bary_min,'per_parent_oriented_area_relative_max':float(err),'positive_child_orientation':bool((signed>0).all()),'old_vertices_bitwise_preserved':bool(np.array_equal(v[:len(v0)],v0))}
    if not (plane.max()<tolerance and bary_min>-tolerance and err<tolerance and check['positive_child_orientation'] and check['old_vertices_bitwise_preserved']):raise AssertionError(check)
    return check

@njit(cache=True)
def ray_hits(v,f,dirs):
    """Moller-Trumbore + dedup of coincident indexed-face hits; origin must be0."""
    counts=np.zeros(len(dirs),np.int64);r=np.full(len(dirs),np.nan)
    for i in range(len(dirs)):
        d=dirs[i];hits=np.empty(len(f));n=0
        for j in range(len(f)):
            a=v[f[j,0]];e=v[f[j,1]]-a;g=v[f[j,2]]-a;h=np.cross(d,g);det=np.dot(e,h)
            if abs(det)<1e-14:continue
            inv=1/det;s=-a;u=np.dot(s,h)*inv
            if u < -1e-10 or u>1+1e-10:continue
            q=np.cross(s,e);vv=np.dot(d,q)*inv
            if vv < -1e-10 or u+vv>1+1e-10:continue
            t=np.dot(g,q)*inv
            if t>1e-10:hits[n]=t;n+=1
        if n:
            hh=np.sort(hits[:n]);ct=1
            for j in range(1,n):
                if hh[j]-hh[j-1]>1e-8*max(1.,abs(hh[j])):ct+=1
            counts[i]=ct
            if ct==1:r[i]=hh[0]
    return counts,r
