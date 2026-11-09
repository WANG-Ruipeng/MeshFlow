"""CPU-only falsification of the proposed <60 degree C/seam normal rule.

Reads immutable cached sources, outputs, pair/edge classifications. No model,
training, mesh mutation, random draws, or reclassification of original scores.
The saved source normal is a winding convention, not an outwardness certificate.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT/'experiments/r2_seam_slot_c200_v1'
PRIOR = ROOT/'experiments/r2_idea_1b_3_4a_diagnostic_v1/idea3'
OUT = ROOT/'experiments/r2_seam_geometry_guard_v1/normal'
L = math.sqrt(3)*2/.3762
HASHES = {}


def path(p):
    p = str(p).replace('\\', '/')
    if p.startswith('E:/') and os.name != 'nt': p = '/mnt/e/'+p[3:]
    if p.startswith('/mnt/e/') and os.name == 'nt': p = 'E:/'+p[7:]
    return Path(p)


def sha(p): return hashlib.sha256(path(p).read_bytes()).hexdigest()
def track(p):
    p = path(p); HASHES[str(p)] = sha(p); return p
def read(p): return json.loads(track(p).read_text(encoding='utf-8-sig'))
def arrays(p):
    with np.load(track(p), allow_pickle=False) as f: return {k:f[k].copy() for k in f.files}
def rows(p):
    with track(p).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def write(p, obj):
    with path(p).open('x', encoding='utf-8') as f: json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
def table(p, data):
    fields = list(dict.fromkeys(k for r in data for k in r))
    with path(p).open('x', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in data: w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})


def normal(tri):
    tri=np.asarray(tri,float); z=np.cross(tri[1]-tri[0],tri[2]-tri[0]); norm=np.linalg.norm(z)
    return None if norm<=1e-12*L*L else z/norm


def geometry(c_tri, c0, c1, third):
    """t follows C winding; m=n_C x t points into incident C triangle."""
    c0,c1,third=map(lambda x:np.asarray(x,float),(c0,c1,third))
    n=normal(c_tri); s=normal([c1,c0,third]); e=c1-c0; length=np.linalg.norm(e)
    if n is None or s is None or length<=1e-12*L:
        return dict(status='NA_DEGENERATE',normal_cosine=None,normal_angle_deg=None,
                    reject_normal_lt60=None,tangent_model=None,inward_conormal_model=None,height_model=None)
    t=e/length; m=np.cross(n,t); p=third-(c0+c1)*.5
    cosine=float(np.clip(n@s,-1,1)); inward=float(p@m); height=float(p@n)
    analytic=-inward/math.hypot(inward,height)
    if not np.isclose(cosine,analytic,atol=2e-6): raise ValueError('Normal/local-coordinate identity failed')
    return dict(status='PASS',normal_cosine=cosine,normal_angle_deg=math.degrees(math.acos(cosine)),
        reject_normal_lt60=cosine<=.5,tangent_model=float(p@t),inward_conormal_model=inward,
        height_model=height,inward_conormal_fraction_L=inward/L,height_fraction_L=height/L,
        reverse_base_area_model2=float(np.linalg.norm(np.cross(c0-c1,third-c1))*.5),
        identity_error=abs(cosine-analytic))


def summary(data, keys):
    groups={}
    for r in data: groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    result=[]
    for kk,rr in sorted(groups.items()):
        ok=[r for r in rr if r['status']=='PASS']; rejected=sum(bool(r['reject_normal_lt60']) for r in ok)
        angles=[r['normal_angle_deg'] for r in ok]
        result.append(dict(zip(keys,kk),edges=len(rr),valid_normals=len(ok),reject_lt60=rejected,
            reject_fraction=rejected/len(ok) if ok else None,
            angle_min_deg=min(angles) if angles else None,angle_median_deg=float(np.median(angles)) if angles else None,
            angle_max_deg=max(angles) if angles else None))
    return result


def source_audit():
    reg=read(PRIOR/'registration.json'); oldrows=rows(PRIOR/'per_source_interface.csv')
    records={r['condition_id']:r for r in reg['records']}
    pilot={r['condition_id'] for r in read(EXP/'eval_manifest.json')['conditions']}
    parents={}; result=[]
    for r in oldrows:
        cid=r['condition_id']; rec=records[cid]
        if cid not in parents:
            p=arrays(rec['parent_npz'])
            assert sha(rec['parent_npz'])==rec['parent_sha256']
            parents[cid]=p
        p=parents[cid]; v=p['model_vertices']; cf=int(r['C_source_face']); yf=int(r['Y_source_face'])
        u,w=map(int,json.loads(r['C_directed_edge'])); third=int(r['third_vertex_id'])
        tri=v[p['source_faces'][cf]]; z=geometry(tri,v[u],v[w],v[third]); yn=normal(v[p['source_faces'][yf]])
        cn=normal(tri); source_cos=float(cn@yn) if cn is not None and yn is not None else None
        result.append(dict(condition_id=cid,parent_id=r['parent_id'],group=rec['group'],
            in_C200_four_conditions=cid in pilot,edge_id=int(r['source_edge_id']),source_C_face=cf,source_Y_face=yf,
            source_C_directed_edge=[u,w],third_vertex_id=third,source_classification=r['classification'],
            source_winding_opposite=r['winding_opposite']=='True',assignment=r['assignment'],
            source_Y_legal_C_edges=int(r['Y_legal_C_edges']),source_normal_cosine=source_cos,
            normal_reference='saved_original_source_C_winding_outwardness_not_recertified',**z))
    return result


def generated_audit():
    m=read(EXP/'eval_manifest.json'); result=[]; legal=[]
    for q in m['requests']:
        for arm in m['arms']:
            base=EXP/'evaluation'/arm/q['request_id']; raw=arrays(base/'raw/raw.npz'); post=arrays(base/'post/post.npz')
            scores=EXP/'evaluation/scores'/arm/q['request_id']/'POST'
            edge_rows=rows(scores/'valid_edges.csv')
            k=q['K']; c=raw['C_vertices']; cf=raw['C_faces_original']
            # All existing legal CF edges, independent of seam owner identity.
            for e in edge_rows:
                if e['valid_edge']!='True' or e['is_C_F']!='True': continue
                faces=json.loads(e['incident_face_ids']); ci=min(faces); fi=max(faces)
                cn=normal(post['output'][ci]); fn=normal(post['output'][fi]); z=None if cn is None or fn is None else float(np.clip(cn@fn,-1,1))
                legal.append(dict(arm=arm,request_id=q['request_id'],parent_id=q['parent_id'],C_face=ci,free_face=fi,
                    raw_slot_id=int(post['raw_slot_ids'][fi]),is_seam=bool(post['retained_seam_mask'][fi]),
                    vertex_u=int(e['vertex_u']),vertex_v=int(e['vertex_v']),length_model=float(e['length_model']),
                    winding_opposite=e['winding_opposite']=='True',status='NA_DEGENERATE' if z is None else 'PASS',
                    normal_cosine=z,normal_angle_deg=None if z is None else math.degrees(math.acos(z)),
                    reject_normal_lt60=None if z is None else z<=.5))
            if arm!='SEAM_SLOT_C200': continue
            pre=arrays(base/'post/pre_geo.npz'); reach=read(scores/'seam_to_bulk.json')
            for slot in np.flatnonzero(raw['raw_seam_slot_mask']):
                slot=int(slot); ei=int(raw['raw_slot_edge_id'][slot]); ci,u,w=map(int,raw['edge_records'][ei]); ctri=c[cf[ci]]
                candidates=np.flatnonzero(raw['raw_anchor_ids'][slot]<0); assert len(candidates)==1; third_corner=int(candidates[0])
                loc=np.flatnonzero(post['raw_slot_ids']==slot)
                postlegal=[]
                if len(loc):
                    fi=int(loc[0]); postlegal=[e for e in edge_rows if e['valid_edge']=='True' and e['is_C_F']=='True' and fi in json.loads(e['incident_face_ids'])]
                for stage in ('RAW','PRE_GEO','POST'):
                    metadata=dict(arm=arm,request_id=q['request_id'],parent_id=q['parent_id'],noise_index=q['noise_index'],
                        stage=stage,raw_slot_id=slot,edge_id=ei,C_face=ci,post_seam_has_any_legal_CF=bool(postlegal),
                        post_bulk_reach_fraction=reach['free_only_reached_bulk_area_fraction'])
                    if stage=='POST' and not len(loc):
                        result.append(dict(**metadata,status='NA_DELETED_BY_ORIGINAL_GEO',normal_cosine=None,
                            normal_angle_deg=None,reject_normal_lt60=None)); continue
                    if stage=='RAW': third=raw['raw'][slot,third_corner]
                    elif stage=='PRE_GEO': third=pre['output'][slot,third_corner]
                    else:
                        output_corner=np.flatnonzero(post['source_corner_ids'][fi]==third_corner)
                        assert len(output_corner)==1; third=post['output'][fi,int(output_corner[0])]
                    result.append(dict(**metadata,third_vertex_xyz=third.tolist(),**geometry(ctri,c[u],c[w],third)))
    return result,legal


def analytic_audit():
    c=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    tests=[('outward_coplanar',[.5,-1.,0.],'LEGAL_LOCAL_CONTINUATION'),
           ('vertical_down',[.5,0.,-1.],'LEGAL_LOCAL_RIGHT_ANGLE_SIDEWALL'),
           ('slanted_under_inward',[.5,.2,-1.],'LEGAL_LOCAL_UNDER_SEAT_SIDEWALL'),
           ('inward_coplanar',[.25,.25,0.],'ILLEGAL_POSITIVE_AREA_OVERLAP'),
           ('outward_but_wrong_height',[.5,-1.,.2],'ANGLE_RULE_PASSES_BUT_TASK_ROLE_UNCERTIFIED')]
    return [dict(case=name,third_vertex=x,local_interpretation=meaning,**geometry(c,c[0],c[1],x)) for name,x,meaning in tests]


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'completion.json').exists():
        done=json.loads((OUT/'completion.json').read_text())
        for p,h in done['input_hashes'].items(): assert sha(p)==h,p
        for p,h in done['output_hashes'].items(): assert sha(p)==h,p
        print(json.dumps(dict(status='COMPLETE_VERIFIED_NO_RECOMPUTE',summary=done['summary']))); return
    if (OUT/'attempt.json').exists(): raise RuntimeError('Previous incomplete attempt: inspect before explicit recovery')
    def cmd(args):
        try:
            p=subprocess.run(args,capture_output=True,text=True,timeout=15);return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
        except Exception as e:return dict(error=repr(e))
    resources=dict(cpu_logical=os.cpu_count(),loadavg=list(os.getloadavg()),cpu_threads=1,workers=1,
        memory=Path('/proc/meminfo').read_text(),gpu=cmd(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader']),
        gpu_processes=cmd(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv']),
        CPU_processes=cmd(['ps','-eo','pid,pcpu,pmem,args','--sort=-pcpu']))
    write(OUT/'attempt.json',dict(unix=time.time(),model_calls=0,new_geometry=0,resources=resources,entry_sha256=sha(__file__)))
    start=time.perf_counter()
    source=source_audit(); generated,legal=generated_audit(); analytic=analytic_audit()
    table(OUT/'source_per_edge.csv',source);table(OUT/'generated_per_seam_stage.csv',generated)
    table(OUT/'existing_legal_CF_normal_rejections.csv',legal);table(OUT/'analytic_examples.csv',analytic)
    ss=summary(source,['group','in_C200_four_conditions']); gs=summary(generated,['stage']); ls=summary(legal,['arm'])
    table(OUT/'source_summary.csv',ss);table(OUT/'generated_stage_summary.csv',gs);table(OUT/'legal_CF_summary.csv',ls)
    pilot=[r for r in source if r['in_C200_four_conditions']]
    source4=summary(pilot,[])[0]; sourceall=summary(source,[])[0]
    good=[r for r in generated if r['stage']=='POST' and r.get('post_bulk_reach_fraction',0)>=.30 and r['post_seam_has_any_legal_CF']]
    goodsum=summary(good,[])[0]
    outlier=[r for r in generated if r['request_id']=='3f741759_n2' and r['raw_slot_id']==16]
    sums=dict(source_all=sourceall,source_four_C=source4,source_groups=ss,generated_stages=gs,
        existing_legal_CF=ls,legal_seams_in_bulk_ge30_outputs=goodsum,outlier_3f_n2_slot16=outlier,
        decision='REJECT_AS_STATED_NORMAL_LT60_RULE',source_outwardness='Original source winding only; no global outwardness recertification',
        alternative='Keep source-opposed edge winding; distinguish in-plane inward conormal from surface normal; reject actual illegal pair geometry, not all dihedral angles above60. No training rule is certified by this CPU audit.')
    write(OUT/'summary.json',sums)
    report=f'''# Third-point normal guard: CPU falsification\n\nThe proposed normal angle <60 degrees guard is **not safe as stated**. It rejects {source4['reject_lt60']}/{source4['valid_normals']} legal source interfaces in the four-C200 training conditions, and {sourceall['reject_lt60']}/{sourceall['valid_normals']} across the existing source interface table. These are existing verified local interface contacts, not certification of whole-source quality.\n\nAmong retained seam faces with legal CF in outputs with bulk reach >=30%, it rejects {goodsum['reject_lt60']}/{goodsum['valid_normals']}. Existing legal CF table also evaluates both control and candidate. No faces are changed or deleted by this audit.\n\n## Geometric contract\n\nLet t be the directed C boundary edge, n the incident original-C face normal, and m=n cross t its inward in-plane conormal. Write x3-edge_midpoint = a t + r m + h n. The seam uses reversed base (c1,c0,x3). Then dot(n_seam,n_C) = -r/sqrt(r^2+h^2). Requiring angle<60 means r < -sqrt(3)*abs(h): it forces a narrow outward in-plane wedge. A valid vertical sidewall has r=0, h!=0 and angle=90, so it is rejected. A valid underside slant can have r>0,h<0 and angle>90. The plane normal is not the boundary outward conormal.\n\nAn inward coplanar extension (r>0,h=0) is indeed rejected, but detecting this one bad case does not validate the broader rule. Even a passing angle does not certify semantic use, absence of intersections with nonincident C faces, FF quality or bulk connection. Analytic examples are saved separately.\n\n## Allowed next design inference\n\nPreserve reversed owner-edge winding. If a geometric penalty is designed, target finite forbidden overlap/penetration with actual C, or a specifically validated near-coplanar inward wedge, while allowing right-angle sidewalls. C alone does not specify one unique sidewall normal or dihedral target. Using source-owner normals would be supervised training information, not an inference oracle, and requires a separate experiment. Nothing here trains that loss.\n\nThe 3f/n2 slot16 RAW/PRE_GEO/POST local coordinates are in summary.json and generated_per_seam_stage.csv. A temporal trajectory claim cannot be inferred from these endpoints.\n\n## Scope\n\nSingle CPU worker/thread; zero model imports/calls, no GPU computation, no new noise, no mesh edits, no old scoring changes. The existing source classifications and actual indexed-edge cache are read-only. Normal orientation references original saved C winding; global outwardness is not independently certified. Null is not treated as a required connection.\n'''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    for p,h in HASHES.items(): assert sha(p)==h,p
    files={str(p):sha(p) for p in OUT.iterdir() if p.is_file()}
    write(OUT/'completion.json',dict(status='COMPLETE',seconds=time.perf_counter()-start,model_calls=0,GPU_calls=0,
        CPU_workers=1,CPU_threads=1,input_hashes=HASHES,output_hashes=files,summary=sums,entry_sha256=sha(__file__)))
    print(json.dumps(dict(status='COMPLETE',seconds=time.perf_counter()-start,summary=sums),ensure_ascii=False))


if __name__=='__main__': run()
