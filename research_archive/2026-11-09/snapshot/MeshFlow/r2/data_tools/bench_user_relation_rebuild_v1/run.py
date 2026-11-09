"""Bounded CPU-only Bench relation data rebuild; no target edits or model calls."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
import argparse, csv, hashlib, io, json, subprocess, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from r2.data_tools.bench_parts_v1 import quality as Q
from r2.data_tools.bench_parts_v1.construct import partition

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_user_relation_rebuild_v1'
OLD=ROOT/'experiments/r2_bench_user_data_v2_expansion_v1/manifest.json'
PLAN=ROOT/'experiments/r2_bench_relation_coverage_v1/render_plan.json'
PIDS=['a8e987a6a6486b962d6c8e51365a5a87','7b58a3e5d7f4dddcad1d789f3b2120d0','9e429b7f77646b4ccc8f8e7f17e9d7b6','943dde2754ddc7822e8ff3556a90169']
VERSION='BENCH_USER_RELATION_REBUILD_V1'
def path(p):
    p=Q.local_path(p)
    return p if p.is_absolute() else ROOT/p
def read(p):return json.loads(path(p).read_text('utf-8-sig'))
def sha(p):return Q.sha(path(p))
def write(p,x):Q.write_new(path(p),x)
def arrays(p):
    with np.load(path(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def event(kind,**kw):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'ledger.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),event=kind,**kw),ensure_ascii=False)+'\n')
def bytes_new(p,data):
    p=path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        if p.read_bytes()!=data:raise FileExistsError(str(p))
    else:p.write_bytes(data)
def npz(p,a):
    p=path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        old=arrays(p)
        assert set(old)==set(a) and all(np.array_equal(old[k],a[k]) for k in a),str(p)
    else:np.savez_compressed(p,**a)
def csvout(p,rows):
    if not rows:return
    cols=list(dict.fromkeys(k for r in rows for k in r));buf=io.StringIO(newline='')
    w=csv.DictWriter(buf,fieldnames=cols);w.writeheader()
    for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
    bytes_new(p,buf.getvalue().encode('utf-8-sig'))
def obj(p,v,f):
    text=''.join('v '+' '.join(format(float(x),'.9g') for x in r)+'\n' for r in v)
    text+=''.join('f '+' '.join(str(int(x)+1) for x in r)+'\n' for r in f)
    bytes_new(p,text.encode('ascii'))
def resources():
    import psutil
    def cmd(args):
        try:
            p=subprocess.run(args,capture_output=True,text=True,timeout=15)
            return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
        except Exception as e:return dict(status='NA',reason=str(e))
    return dict(utc=datetime.now(timezone.utc).isoformat(),logical_cores=os.cpu_count(),RAM_available=psutil.virtual_memory().available,
      CPU_percent=psutil.cpu_percent(interval=.2),load=os.getloadavg(),numeric_threads=1,
      gpu=cmd(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv']),
      gpu_processes=cmd(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv']),
      cpu_jobs=[dict(pid=p.pid,name=p.info['name'],command=p.info['cmdline']) for p in psutil.process_iter(['name','cmdline']) if any(s in (p.info['name'] or '') for s in ('python','train'))])
def parents():
    rows=read(PLAN)['parents'];return [r for r in rows if r['parent_id'] in PIDS]
def init():
    if (OUT/'protocol.json').exists():verify();return
    pr=parents();assert len(pr)==4
    old=read(OLD);protected={str(OLD):sha(OLD),str(PLAN):sha(PLAN)}
    protected.update(old['protected_source_hashes'])
    for r in pr:
        for k in ('target_mesh_path','target_case_path','lineage_path','geometry_path'):
            if r.get(k):protected[str(path(r[k]))]=sha(r[k])
    write(OUT/'protected_source_hashes.json',protected)
    write(OUT/'resources_before.json',resources())
    write(OUT/'protocol.json',dict(schema=VERSION,parent_ids=PIDS,max_candidates_per_parent=3,max_candidates=12,
      target_policy='Frozen audited target M reused byte-for-byte; select whole existing target faces only; C union Y exact disjoint partition',
      scope='Complete slat/slat assembly, transverse end frames, curved upper seat and inner back; no random face patches',
      qualification='Original frozen SourceAuditor + actual AI_PROXY multiview review + unchanged production CPU Native/SEAM packing',
      duplicate_policy='Identical source-parent/version/face-subset never counted as new; same parent variants are not independent parents',
      model_input='C/N only; semantic names and source owner/alias labels are curation/supervision only',
      CPU_budget='2 independent face-design workers, 1 thread each; packaging/OT sequential; at most one production OT per final accepted new condition',
      noise_policy='Only CPU data packing validation arrays; no new model output or evaluation noise',
      training_updates=0,model_forwards=0,GPU_compute=0,default_pool_changed=False,commit_push=False,
      old_train_conditions=len(old['train']),old_train_parents=len(old['parents']),old_dev_conditions=len(old['dev']),
      old_manifest_sha256=sha(OLD),protected_source_hashes_sha256=sha(OUT/'protected_source_hashes.json')))
    event('INITIALIZED',parents=4,CPU_design_workers=2,CPU_pack_workers=1)
    print(json.dumps(dict(status='INITIALIZED',parents=[{k:r.get(k) for k in ('parent_id','status','review_status','reasons','eligibility')} for r in pr],resources=read(OUT/'resources_before.json')),ensure_ascii=False))
def verify():
    for p,h in read(OUT/'protected_source_hashes.json').items():assert sha(p)==h,'Protected changed: '+p
    print('PROTECTED_INPUTS_UNCHANGED')
def package(selection_files):
    verify();start=time.perf_counter();pm={r['parent_id']:r for r in parents()};old=read(OLD)
    proposals=[]
    for p in selection_files:
        data=read(p)
        rows=data if isinstance(data,list) else data['candidates']
        for r in rows:proposals.append(dict(r,selection_path=str(path(p)),selection_sha256=sha(p)))
    assert len(proposals)<=12 and max(Counter(r['parent_id'] for r in proposals).values())<=3
    write(OUT/'frozen_selections.json',proposals)
    write(OUT/'resources_pack.json',resources());event('PACK_START',candidate_count=len(proposals))
    results=[];auditors={};seen=set()
    for p in proposals:
        uid=p['parent_id'];pr=pm[uid];key=p.get('key',p.get('region_name'))
        ids=sorted(p.get('target_face_ids',p.get('face_ids')));identity=(uid,tuple(ids))
        assert identity not in seen;seen.add(identity)
        cid=uid+'__'+key+'__REL_V1';dest=OUT/'conditions'/uid/cid
        if uid not in auditors:
            a=arrays(pr['target_case_path']);auditors[uid]=(a,Q.SourceAuditor(a,sha(pr['target_case_path'])))
        a,aud=auditors[uid];pair=partition(a,ids)
        q,sup=aud.candidate(np.asarray(ids,np.int64),dict(candidate_id=cid,parent_id=uid,group='TRAIN',proposed_part_type=p['part_type']))
        case=dict(a,**pair);case.update(sup)
        cvid=set(case['C_source_vertex_ids'].tolist())
        case['source_third_in_C']=np.asarray([bool(ok and tr[2] in cvid) for ok,tr in zip(case['source_supported_labels'],case['edge_source_triple'])])
        inp={k:case[k] for k in ('C','C_vertices','C_faces_original','C_faces_model','C_source_vertex_ids','C_source_corner_permutations','N','K','target_version_id')}
        for name,aa in (('case',case),('input',inp),('paired',pair),('supervision_only',sup)):npz(dest/(name+'.npz'),aa)
        obj(dest/'C.obj',case['C_vertices'],case['C_faces_original']);obj(dest/'Y.obj',a['model_vertices'],a['source_faces'][case['free_ids']])
        write(dest/'quality.json',q);write(dest/'selection.json',p)
        n,k,b=int(case['N']),int(case['K']),len(sup['edge_records'])
        duplicate=[r['condition_id'] for r in old['train'] if r['parent_id']==uid and sorted(r['target_face_ids'])==ids]
        r=dict(condition_id=cid,parent_id=uid,group='TRAIN',official_split='train',target_version_id=str(a['target_version_id'].item()),
          condition_version=VERSION,part_type=p['part_type'],region_name=key,extent_description=p.get('reason',p.get('semantic_description',p.get('extent_description',p.get('description')))),
          target_face_ids=ids,source_face_ids=ids,N=n,K=k,boundary_edges=b,source_owner_count=int(sup['occupied_labels'].sum()),source_alias_count=int(sup['source_alias_labels'].sum()),
          source_null_count=int((~sup['source_supported_labels']).sum()),area_fraction=q['area_fraction_source_face_area'],
          source_edge_components=len(q['topology']['edge_connected_components']),geometry_status=q['geometry_status'],geometry_reasons=q['reasons'],
          adapter_compatible=bool(128<=n<=256 and 2<=k<n and b<=n-k and not q['reasons']),
          adapter_semantics='CLOSED_NATIVE_NO_SEAM' if not b else ('OPEN_WITH_SEAM_OWNER' if sup['occupied_labels'].any() else 'OPEN_NATIVE_NO_OWNER'),
          parent_target_status=pr['status'],review_status='NOT_REVIEWED',eligibility='CANDIDATE_ONLY',duplicate_old_conditions=duplicate,
          selection_path=str(dest/'selection.json'),selection_sha256=sha(dest/'selection.json'),
          model_input_policy='C and N only; part, full target, source owner/alias and lineage are supervision/curation metadata')
        for field,name in (('case','case'),('input','input'),('paired','paired')):r[field+'_path']=str(dest/(name+'.npz'));r[field+'_sha256']=sha(r[field+'_path'])
        for field,name in (('supervision','supervision_only.npz'),('quality','quality.json')):r[field+'_file']=str(dest/name);r[field+'_sha256']=sha(r[field+'_file'])
        results.append(r);event('CANDIDATE_PACKED',condition_id=cid,K=k,N=n,geometry_status=q['geometry_status'],reasons=q['reasons'],duplicate_old=duplicate)
    manifest=dict(schema=VERSION,parents=list(pm.values()),conditions=results,train=[],dev=old['dev'],
      dev_source_manifest_path=old['dev_source_manifest_path'],dev_source_manifest_sha256=old['dev_source_manifest_sha256'],
      protected_source_hashes=read(OUT/'protected_source_hashes.json'),status='CANDIDATES_NEED_ACTUAL_REVIEW',training_registered=False)
    write(OUT/'candidate_manifest.json',manifest);csvout(OUT/'candidate_conditions.csv',results)
    verify();event('PACK_COMPLETE',seconds=time.perf_counter()-start,count=len(results),model_calls=0)
    print(json.dumps([{k:r[k] for k in ('condition_id','N','K','boundary_edges','geometry_status','geometry_reasons','adapter_compatible','duplicate_old_conditions')} for r in results],indent=2))
def render():
    from PIL import Image,ImageDraw
    from r2.data_tools.bench_user_v2.render import cell,basis,shared_bounds,font,text_lines
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
    m=read(OUT/'candidate_manifest.json');cams=[basis(-60,22),basis(120,22)];s=440;rows=[]
    for r in m['conditions']:
        a=arrays(r['case_path']);t=a['model_vertices'][a['source_faces']];ct=t[a['known_ids']];yt=t[a['free_ids']]
        center,span=shared_bounds([t],cams);im=Image.new('RGB',(s*4,110+(s+40)*2),'white');d=ImageDraw.Draw(im)
        text_lines(d,(10,8),r['condition_id']+'\n'+str(r['extent_description'])+' | K='+str(r['K'])+' N='+str(r['N']),s*4-20,16,4)
        for vi,cam in enumerate(cams):
            for col,(name,tri,k) in enumerate([('M: whole unchanged target',t,0),('C: supplied part',ct,len(ct)),('Y: exact complement',yt,0),('C + Y',np.concatenate([ct,yt]),len(ct))]):
                x,y=col*s,110+vi*(s+40);d.text((x+8,y),name,font=font(17),fill='black');tile,_=cell(tri,k,cam,center,span,s);im.paste(tile,(x,y+32))
        out=OUT/'gallery'/(r['condition_id']+'.png');buf=io.BytesIO();im.save(buf,format='PNG');bytes_new(out,buf.getvalue());rows.append(dict(condition_id=r['condition_id'],path=str(out),sha256=sha(out)))
    write(OUT/'gallery_manifest.json',rows)
    event('RENDERED_EXACT_PACKED_ARRAYS',pages=len(rows),views=2)
    print(json.dumps(dict(rendered=len(rows))))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['init','package','render','verify']);ap.add_argument('--selections',nargs='*');args=ap.parse_args()
    if args.stage=='package':package(args.selections)
    else:globals()[args.stage]()
if __name__=='__main__':main()
