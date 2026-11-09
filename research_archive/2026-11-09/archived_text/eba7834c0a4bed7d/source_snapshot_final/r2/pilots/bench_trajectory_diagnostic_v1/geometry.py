import numpy as np
L=float(np.sqrt(3)*5.31488037109375)

def roles(s):
    free=np.asarray(s.get('free_corner_mask',s['anchor_ids']<0),bool)
    known=np.asarray(s['known_mask'],bool);valid=np.asarray(s['valid_mask'],bool)
    free=free & valid[:,None] & ~known[:,None]
    tip=free & (free.sum(1)==1)[:,None]
    bulk=free & (free.sum(1)==3)[:,None]
    return dict(all=free,tip=tip,bulk=bulk)

def projection(x,v,s):
    ids=s['shared_vertex_ids'];fixed=s['anchor_ids']>=0;p=x.copy();jv=v.copy()
    for identity in np.unique(ids):
        mask=ids==identity;anchors=mask&fixed
        if anchors.any():p[mask]=s['x1'][anchors][0];jv[mask]=0
        else:p[mask]=x[mask].mean(0);jv[mask]=v[mask].mean(0)
    e=x-p;out={}
    for name,mask in roles(s).items():
        out[name+'_free_scalar_count']=int(mask.sum()*3)
        out[name+'_split_mse']=float((e[mask]**2).mean()) if mask.any() else None
        out[name+'_split_derivative']=float(2*(e[mask]*(v-jv)[mask]).mean()) if mask.any() else None
    return out

def shape(x,s):
    target=s['x1'];mask=~s['known_mask'];a=x[mask];y=target[mask]
    edges=a[:,[1,2,0]]-a;ye=y[:,[1,2,0]]-y
    lens=np.linalg.norm(edges,axis=-1);yl=np.linalg.norm(ye,axis=-1)
    normals=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);yn=np.cross(y[:,1]-y[:,0],y[:,2]-y[:,0])
    area=np.linalg.norm(normals,axis=-1)/2;ya=np.linalg.norm(yn,axis=-1)/2
    ratios=area/np.maximum(ya,1e-12);valid=ya>1e-12
    flip=np.sum(normals*yn,axis=-1)<0
    return dict(free_faces=int(mask.sum()),endpoint_reference_mse=float(((a-y)**2).mean()),
        edge_ratio_median=float(np.median(lens/np.maximum(yl,1e-12))),edge_below_half_fraction=float(np.mean(lens<.5*yl)),
        area_ratio_median=float(np.median(ratios[valid])),area_below_quarter_fraction=float(np.mean(ratios[valid]<.25)),
        source_normal_flip_fraction=float(np.mean(flip[valid])),free_area=float(area.sum()),source_free_area=float(ya.sum()),
        support_scope='all free source faces; part-specific completeness requires images')
