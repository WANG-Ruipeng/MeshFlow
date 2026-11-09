"""CPU geometry recurrence diagnostics; area/bounds are not quality scores."""
import numpy as np
import common as C

def shape(x,K):
 f=x[K:].astype(np.float64);area=.5*np.linalg.norm(np.cross(f[:,1]-f[:,0],f[:,2]-f[:,0]),axis=-1);pts=f.reshape(-1,3);lo=pts.min(0);hi=pts.max(0)
 return dict(free_area=float(area.sum()),free_min_triangle_area=float(area.min()),free_zero_area_faces=int((area==0).sum()),free_bbox_x=float(hi[0]-lo[0]),free_bbox_y=float(hi[1]-lo[1]),free_bbox_z=float(hi[2]-lo[2]),free_centroid_x=float(pts[:,0].mean()),free_centroid_y=float(pts[:,1].mean()),free_centroid_z=float(pts[:,2].mean()))
def main():
 reg=C.prepare();rows=[]
 for r in reg['records']:
  K=int(r['K']);rid=r['request_id']
  with np.load(r['raw_path'],allow_pickle=False) as z:raw=z['raw'].reshape(-1,3,3).copy()
  previous=raw;base=None
  for n in range(6):
   path=r['raw_path'] if n==0 else r['S1_denoised_path'] if n==1 else str(C.OUT/'denoised'/f'S{n}'/rid/'denoised.npz')
   with np.load(path,allow_pickle=False) as z:x=z['raw'].reshape(-1,3,3).copy() if n==0 else z['output'].copy()
   d=np.linalg.norm((x[K:].astype(np.float64)-previous[K:].astype(np.float64)),axis=-1).ravel();total=np.linalg.norm(x[K:].astype(np.float64)-raw[K:].astype(np.float64),axis=-1).ravel();stats=shape(x,K)
   if n==1:base=stats
   rows.append(dict(request_id=rid,parent_id=r['parent_id'],region_type=r['region_type'],candidate=r['candidate'],round=n,stage='RAW' if n==0 else f'S{n}',K=K,N=int(r['N']),C_unchanged=x[:K].tobytes()==raw[:K].tobytes(),mean_step=float(d.mean()),p95_step=float(np.quantile(d,.95)),max_step=float(d.max()),mean_from_RAW=float(total.mean()),**stats,free_area_ratio_to_S1=stats['free_area']/base['free_area'] if base else None,input_path=path,input_sha256=C.sha(path)))
   previous=x
 C.csvwrite(C.OUT/'recurrence_geometry_per_output.csv',rows)
 summary=[]
 for n in range(6):
  sub=[r for r in rows if r['round']==n];summary.append(dict(round=n,outputs=len(sub),**{k:float(np.mean([r[k] for r in sub])) for k in ('mean_step','p95_step','max_step','mean_from_RAW','free_area','free_bbox_x','free_bbox_y','free_bbox_z')},free_area_ratio_to_S1=float(np.mean([r['free_area_ratio_to_S1'] for r in sub])) if n else None))
 C.csvwrite(C.OUT/'recurrence_geometry_summary.csv',summary)
 print(summary)
if __name__=='__main__':main()
