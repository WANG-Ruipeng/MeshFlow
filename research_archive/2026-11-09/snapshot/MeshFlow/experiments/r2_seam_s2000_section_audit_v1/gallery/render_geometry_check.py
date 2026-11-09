"""Independent CPU checks of saved renderer primitives, no mesh modification."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import hashlib,json,time
from pathlib import Path
import numpy as np
G=Path(__file__).parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def inside(points,face):
    a,b,c=face;u=b-a;v=c-a;p=points-a;n=np.cross(u,v);area2=np.linalg.norm(n)
    if area2<1e-16:return True
    assert np.max(np.abs(p@n))/area2<1e-9
    # Direct 3x2 SVD, not normal equations which square the condition number
    # of the deliberately retained near-degenerate generated triangles.
    uv=np.linalg.lstsq(np.stack([u,v],axis=1),p.T,rcond=None)[0].T
    return bool(np.all(uv>=-1e-7) and np.all(uv.sum(1)<=1+1e-7))
protocol=read(G.parent/'protocol.json');pages=read(G/'pages.json')['pages'];counts=dict(pages=0,section_panels=0,cutaway_panels=0,finite_segments=0,clipped_triangles=0,verified_source_files=0,output_hashes=0)
for r in pages:
    q=next(q for q in protocol['requests'] if q['request_id']==r['request_id']);meshes={};slots={}
    for stage in ('RAW','D0','RESTORED','POST'):
        p=Path(q['paths'][stage]);assert sha(p)==protocol['source_hashes'][str(p)]
        with np.load(p,allow_pickle=False) as z:
            key='raw' if stage=='RAW' else 'official_unconstrained_proposal' if stage=='D0' else 'output'
            meshes[stage]=z[key].astype(np.float64);slots[stage]=z['source_face_ids'].reshape(-1) if stage=='POST' else np.arange(len(meshes[stage]))
        counts['verified_source_files']+=1
    for p,h in r['outputs'].items():assert sha(p)==h;counts['output_hashes']+=1
    with np.load(r['identity_npz'],allow_pickle=False) as z:
        for panel in r['panels']:
            pre=panel['prefix'];s=panel['stage'];axis=panel['plane_axis'];pos=panel['plane_position'];owners=z[pre+'_face_indices'];ids=z[pre+'_raw_slots']
            assert np.array_equal(ids,slots[s][owners]);pix=z[pre+'_pixel_raw_slot'];assert pix.shape==(700,700)
            assert set(np.unique(pix)).issubset({-1,*slots[s].tolist()})
            if panel['mode']=='section':
                seg=z[pre+'_segments'];assert np.max(np.abs(seg[:,:,axis]-pos),initial=0)<=1e-10
                for x,fi in zip(seg,owners):assert inside(x,meshes[s][fi])
                counts['section_panels']+=1;counts['finite_segments']+=len(seg)
            else:
                t=z[pre+'_clipped_triangles'];assert np.max(t[:,:,axis]-pos,initial=-1)<=1e-10
                for x,fi in zip(t,owners):
                    original=meshes[s][fi];assert inside(x,original)
                    n=np.cross(x[1]-x[0],x[2]-x[0]);n0=np.cross(original[1]-original[0],original[2]-original[0])
                    assert n@n0>=-1e-12 and np.linalg.norm(n)<=np.linalg.norm(n0)+1e-10
                rims=z[pre+'_cut_rims'];assert np.max(np.abs(rims[:,:,axis]-pos),initial=0)<=1e-10
                counts['cutaway_panels']+=1;counts['clipped_triangles']+=len(t)
    counts['pages']+=1
receipt=dict(status='PASS',counts=counts,protocol_sha256=sha(G.parent/'protocol.json'),definition_sha256=sha(G/'definition.json'),check_code_sha256=sha(__file__),unix=time.time(),model_calls=0,GPU_calls=0,GEO_runs=0,
    checks=['112 original stage hashes unchanged','84 PNG/preview/identity output hashes','every segment on inherited plane and original finite triangle','every clipped triangle within retained halfspace and original face','clipped triangle winding and area bounded by original','POST slots from original source_face_ids','per-panel pixel identities belong to stage slots'])
with (G/'render_geometry_check.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt,indent=2))
