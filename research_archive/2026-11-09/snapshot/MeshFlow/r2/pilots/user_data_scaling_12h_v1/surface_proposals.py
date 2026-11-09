"""Surface visibility only proposes regions; every accepted region still needs image review."""
from data_prepare import *

def visible(t,axis,sign=1):
    # Deterministic four interior samples. Visibility does not certify semantics or intersections.
    t=t.astype(np.float64); axes=[a for a in range(3) if a!=axis]
    a=t[:,0,axes];b=t[:,1,axes];c=t[:,2,axes];v0=b-a;v1=c-a
    det=v0[:,0]*v1[:,1]-v0[:,1]*v1[:,0];valid=np.abs(det)>1e-12
    weights=np.array([[1/3]*3,[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]])
    result=[];tol=np.ptp(t.reshape(-1,3),axis=0).max()*1e-6
    for i,face in enumerate(t):
        count=0
        for w in weights:
            p=w@face;v2=p[axes]-a
            d=np.where(valid,det,1.)
            u=(v2[:,0]*v1[:,1]-v2[:,1]*v1[:,0])/d;v=(v0[:,0]*v2[:,1]-v0[:,1]*v2[:,0])/d
            hit=valid&(u>=-1e-9)&(v>=-1e-9)&(u+v<=1+1e-9)
            depth=t[:,0,axis]+u*(t[:,1,axis]-t[:,0,axis])+v*(t[:,2,axis]-t[:,0,axis])
            hidden=np.any(hit&((depth-p[axis])*sign>tol))
            count+=not hidden
        result.append(count>=2)
    return np.array(result)

def propose_visible(row):
    with np.load(row['npz'],allow_pickle=False) as z:t=z['full_target'].reshape(-1,3,3).astype(float);f=z['source_faces']
    center=t.mean(1);cross=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);lens=np.linalg.norm(cross,axis=1);norm=np.abs(cross/np.maximum(lens[:,None],1e-20));area=lens/2
    lo=t.reshape(-1,3).min(0);hi=t.reshape(-1,3).max(0);height=hi[1]-lo[1]
    top=visible(t,1);mask=top&(norm[:,1]>.58)&(center[:,1]>lo[1]+.18*height)&(center[:,1]<lo[1]+.75*height)
    p=read(row['proposal_path']);adj={int(k):set(v) for k,v in p['adjacency'].items()}
    groups=proposal.components(np.flatnonzero(mask),adj)
    groups.sort(key=lambda g:(-area[g].sum(),min(g)))
    if groups:
        seed=groups[0];seat_y=float(np.average(center[seed,1],weights=area[seed]));seed_center=np.average(center[seed],axis=0,weights=area[seed])
        selected=[]
        for g in groups:
            gc=np.average(center[g],axis=0,weights=area[g]);
            if abs(gc[1]-seat_y)<.09*height and abs(gc[2]-seed_center[2])<.36*height:selected+=g
        seat=sorted(selected)
    else:seat=[];seat_y=float(lo[1]+.43*height);seed_center=(lo+hi)/2
    preliminary=(norm[:,2]>.45)&(center[:,1]>seat_y+.08*height)
    bc=np.flatnonzero(preliminary)
    back_z=float(np.average(center[bc,2],weights=area[bc])) if len(bc) else lo[2]
    sign=1 if seed_center[2]>=back_z else -1
    front=visible(t,2,sign)
    bmask=front&(norm[:,2]>.48)&(center[:,1]>seat_y+.06*height)&(t[:,:,1].min(1)>=seat_y-.04*height)
    # Supporting back belongs behind the central sitting area, not the front of an arm.
    bmask &= (center[:,2]-seed_center[2])*sign < .03*height
    back=sorted(np.flatnonzero(bmask).tolist())
    out=dict(p,geometric_proposal_version='visibility_v2',semantic_status='NOT_REVIEWED',
        note='Visibility/horizontal height are candidate aids only; original indexed faces, no cutting, repair or generated outputs.',
        seat_reference_height=seat_y,front_direction=sign,
        candidates=[dict(candidate_id=k,source_face_ids=ids,K=len(ids),area=float(area[ids].sum()),proposal_only=True) for k,ids in [('seat_surface',seat),('back_surface',back)]],
        proposed_regions=dict(SEAT='seat_surface' if seat else None,BACK='back_surface' if back else None,status='PROPOSAL_ONLY_NOT_REVIEWED'))
    return out

def run():
    rows=read(OUT/'data/parent_candidates.json')['rows'];new=[]
    for r in rows:
        p=propose_visible(r);dest=OUT/'data/proposals_v2'/f"{r['uid']}__{r['variant_uid']}.json";save(dest,p)
        new.append(dict(r,proposal_path=str(dest),proposal_sha256=sha(dest)))
    save(OUT/'data/parent_candidates_v2.json',dict(schema='r2_user_scaling_parent_candidates_v1',status='UNREVIEWED',rows=new))
    print('visibility proposals',len(new),flush=True)
if __name__=='__main__':run()
