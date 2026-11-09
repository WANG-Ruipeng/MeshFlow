"""Read-only continuous-stage pair accounting and GEO face survival."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from .common import *
import numpy as np,concurrent.futures,multiprocessing,collections
from r2.tools.hinge_opposite_v1 import engine as H
from r2.pilots.seam_followup_v1.binding import ILLEGAL

def area(tri):return np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2

def one(job):
    label,q=job;rid=q['request_id'];dest=OUT/'stage_audit'/label/rid;done=dest/'complete.json'
    if done.exists():return read(done)
    base=OUT/'evaluation'/label/rid;raw=arrays(base/'raw/raw.npz');k=q['K'];d=arrays(base/'post/proposal.npz')['official_unconstrained_proposal'];p=arrays(base/'post/pre_geo.npz')['output']
    rows=[];scans={};keys={}
    for stage,x in [('RAW',raw['raw']),('D0_PROPOSAL',d),('PRE_GEO',p)]:
        path=dest/(stage+'_pairs.json')
        if path.exists():scan=read(path)
        else:scan=H.scan(x,k);save(path,scan)
        scans[stage]=scan;keys[stage]={kind:{tuple(r['pair']) for r in v['contacts'] if r['classification'] in ILLEGAL} for kind,v in scan.items()}
        row=dict(label=label,request_id=rid,split=q['split'],part=q['part_type'],stage=stage,free_faces=len(x)-k,free_area=float(area(x[k:]).sum()),C_corner_RMS_displacement=float(np.sqrt(np.mean((x[:k]-raw['raw'][:k])**2))))
        for kind in ('CF','FF'):
            affected=set()
            for i,j in keys[stage][kind]:affected.update((i,j) if kind=='FF' else (j,))
            row[kind+'_illegal_pairs']=len(keys[stage][kind]);row[kind+'_affected_free_count']=len(affected);row[kind+'_affected_free_fraction']=len(affected)/(len(x)-k)
        rows.append(row)
    transitions=[]
    for a,b in [('RAW','D0_PROPOSAL'),('D0_PROPOSAL','PRE_GEO')]:
        for kind in ('CF','FF'):
            transitions.append(dict(label=label,request_id=rid,split=q['split'],part=q['part_type'],transition=a+'->'+b,kind=kind,old=len(keys[a][kind]),new=len(keys[b][kind]),retained=len(keys[a][kind]&keys[b][kind]),removed=len(keys[a][kind]-keys[b][kind]),added=len(keys[b][kind]-keys[a][kind])))
    geo=read(base/'post/geometry.json');fr=[]
    for rec in geo['face_records']:
        if rec['known']:continue
        i=rec['source_face_id'];r=dict(label=label,request_id=rid,split=q['split'],part=q['part_type'],**rec,role='SEAM' if raw['raw_seam_slot_mask'][i] else 'BULK',RAW_area=float(area(raw['raw'][i:i+1])[0]),D0_area=float(area(d[i:i+1])[0]),PRE_GEO_area=float(area(p[i:i+1])[0]));fr.append(r)
    csvwrite(dest/'stage_pairs.csv',rows);csvwrite(dest/'pair_transitions.csv',transitions);csvwrite(dest/'GEO_face_survival.csv',fr)
    r=dict(status='COMPLETE',rows=rows,transitions=transitions,faces=fr,model_calls=0,mesh_edits=0,definition='Original exact finite triangle detector; D0 CF uses moving D0 C and is diagnostic only. Same RAW slots paired within request; no cross-model face match.');save(done,r);return r

def main():
    qs=read(OUT/'evaluation_inputs/manifest.json')['requests'];dest=OUT/'stage_audit';done=dest/'complete.json'
    if done.exists():return
    save(dest/'resources.json',resources());tick=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:rr=list(pool.map(one,[(l,q) for l in LABELS for q in qs]))
    for name,key in [('continuous_stage_pairs','rows'),('pair_transitions','transitions'),('GEO_face_survival','faces')]:csvwrite(dest/(name+'.csv'),[r for a in rr for r in a[key]])
    save(done,dict(status='COMPLETE',requests=len(rr),seconds=time.perf_counter()-tick,model_calls=0,workers=2,files={str(p):sha(p) for p in dest.glob('*.csv')}));print('STAGE_AUDIT_COMPLETE',len(rr))
if __name__=='__main__':main()
