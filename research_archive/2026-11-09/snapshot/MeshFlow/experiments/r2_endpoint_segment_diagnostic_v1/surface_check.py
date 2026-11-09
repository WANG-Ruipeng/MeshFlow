"""Exact feature-minimum finite segment/triangle distance; CPU only."""
from run import *

def point_segment(p,a,b):
    d=b-a;den=(d*d).sum(-1)
    t=np.divide(((p-a)*d).sum(-1),den,out=np.zeros_like(den),where=den>1e-24)
    return np.linalg.norm(p-(a+np.clip(t,0,1)[...,None]*d),axis=-1)

def bary_inside(p,tri):
    v=tri[:,1]-tri[:,0];w=tri[:,2]-tri[:,0];u=p-tri[:,0]
    vv=(v*v).sum(-1);ww=(w*w).sum(-1);vw=(v*w).sum(-1);uv=(u*v).sum(-1);uw=(u*w).sum(-1)
    den=vv*ww-vw*vw
    b=np.divide(ww*uv-vw*uw,den,out=np.full_like(den,np.inf),where=abs(den)>1e-24)
    c=np.divide(vv*uw-vw*uv,den,out=np.full_like(den,np.inf),where=abs(den)>1e-24)
    return (b>=-1e-12)&(c>=-1e-12)&(b+c<=1+1e-12)

def point_tri(p,tri):
    n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);nn=(n*n).sum(-1)
    h=((p-tri[:,0])*n).sum(-1)
    proj=p-np.divide(h,nn,out=np.zeros_like(h),where=nn>1e-24)[:,None]*n
    dist=np.min(np.stack([point_segment(p,tri[:,j],tri[:,(j+1)%3]) for j in range(3)]),axis=0)
    good=bary_inside(proj,tri)&(nn>1e-24)
    dist[good]=abs(h[good])/np.sqrt(nn[good])
    return dist

def seg_seg(a,b,c,d):
    u=b-a;v=d-c;w=a-c
    aa=float(u@u);bb=(u*v).sum(-1);cc=(v*v).sum(-1);dd=(u*w).sum(-1);ee=(v*w).sum(-1)
    den=aa*cc-bb*bb
    s=np.divide(bb*ee-cc*dd,den,out=np.full_like(den,np.inf),where=abs(den)>1e-24)
    t=np.divide(aa*ee-bb*dd,den,out=np.full_like(den,np.inf),where=abs(den)>1e-24)
    dist=np.min(np.stack([point_segment(a,c,d),point_segment(b,c,d),point_segment(c,np.broadcast_to(a,c.shape),np.broadcast_to(b,c.shape)),point_segment(d,np.broadcast_to(a,d.shape),np.broadcast_to(b,d.shape))]),axis=0)
    ok=(s>=0)&(s<=1)&(t>=0)&(t<=1)
    dist[ok]=np.minimum(dist[ok],np.linalg.norm(w[ok]+s[ok,None]*u-t[ok,None]*v[ok],axis=-1))
    return dist

def seg_tri(a,b,tri):
    dist=np.minimum(point_tri(a,tri),point_tri(b,tri))
    for j in range(3):dist=np.minimum(dist,seg_seg(a,b,tri[:,j],tri[:,(j+1)%3]))
    n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);den=(n*(b-a)).sum(-1)
    t=np.divide(((tri[:,0]-a)*n).sum(-1),den,out=np.zeros_like(den),where=abs(den)>1e-18)
    inside=bary_inside(a+t[:,None]*(b-a),tri)
    dist[inside&(abs(den)>1e-18)&(t>=0)&(t<=1)]=0
    return dist

def main():
    rows=json.loads((ROOT/'experiments/r2_partial_edge_binding_v1/input_manifest.json').read_text())['rows']
    out=[]
    for row in rows:
        if not row['short_id'].startswith(('608','610')):continue
        a=assets.load_asset(row);v=a['vertices'];f=a['faces'];K=row['K'];sf=a['source_face_ids']
        cb=[e for e,o in inc(f[:K]).items() if len(o)==1]
        fo=[(fi,j,k) for e,o in inc(f,K).items() if len(o)==1 for fi,j,k in o]
        for stage,tris in [('PRE_GEO',a['pre_geo'][sf]),('POST',v[f])]:
            for ei,edge in enumerate(cb):
                c0,c1=v[list(edge)].astype(float)
                distances=seg_tri(c0,c1,tris[K:].astype(float))
                edges=np.array([[tris[fi,j],tris[fi,k]] for fi,j,k in fo],float)
                ed=seg_seg(c0,c1,edges[:,0],edges[:,1])
                near=int(np.argmin(distances));near_e=int(np.argmin(ed))
                out.append(dict(short_id=row['short_id'],stage=stage,C_edge_index=ei,
                    min_C_segment_to_free_surface_pctL=float(distances[near]/L*100),closest_source_face_id=int(sf[K+near]),
                    min_C_segment_to_free_open_edge_pctL=float(ed[near_e]/L*100),closest_open_source_face_id=int(sf[fo[near_e][0]]),
                    free_faces_within2=int((distances<.02*L).sum())))
    save('surface_distance_by_C_edge.csv',out)
    for short in sorted(set(x['short_id'] for x in out)):
        rr=[x for x in out if x['short_id']==short]
        print(short,{stage:dict(surface_min=min(x['min_C_segment_to_free_surface_pctL'] for x in rr if x['stage']==stage),open_edge_min=min(x['min_C_segment_to_free_open_edge_pctL'] for x in rr if x['stage']==stage)) for stage in ('PRE_GEO','POST')})
if __name__=='__main__':main()
