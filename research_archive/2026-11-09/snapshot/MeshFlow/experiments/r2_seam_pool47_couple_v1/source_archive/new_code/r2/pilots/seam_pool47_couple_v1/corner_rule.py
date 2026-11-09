"""Frozen CPU C-only corner closure proposals; source labels are diagnostic only.

No torch, generator, denoiser, training, noise, geometry edits or old-file writes.
This proposes optional finite triangles, not a required-interface classifier.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, collections, csv, hashlib, json, sys, time, traceback, subprocess
from pathlib import Path
import numpy as np
from r2.tools.partial_edge_binding_v1 import metrics as Q, engine as PE

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_seam_pool47_couple_v1/corner_rule'
SOURCE=ROOT/'experiments/r2_seam_abc_v1/B_DATA_REGISTRATION.json'
VERSION='C_only_adjacent_boundary_corner_proposal_v1'
L=float(np.sqrt(3.)*2/.3762)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def csvwrite(p,rows,fields=None):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if fields is None:fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def resources():
    import psutil
    commands={}
    for name,cmd in [('gpu',['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader']),('gpu_processes',['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])]:
        r=subprocess.run(cmd,capture_output=True,text=True);commands[name]=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
    return dict(unix=time.time(),cpu_cores=os.cpu_count(),cpu_percent=psutil.cpu_percent(.1),available_ram_bytes=psutil.virtual_memory().available,
        loadavg=list(os.getloadavg()) if hasattr(os,'getloadavg') else None,process_rss=psutil.Process().memory_info().rss,
        dispatch='1 CPU process x1 math thread; small independent 50-C diagnostic; no GPU worker',**commands)

def incidence(f):
    result=collections.defaultdict(list)
    for fi,t in enumerate(f):
        for j in range(3):
            a,b=int(t[j]),int(t[(j+1)%3]);result[tuple(sorted((a,b)))].append((fi,a,b))
    return dict(result)
def topology_reason(f,affected_vertices=None):
    inc=incidence(f)
    if any(len(o)>2 for o in inc.values()):return 'NONMANIFOLD_EDGE'
    if any(len(o)==2 and o[0][1:]!=o[1][1:][::-1] for o in inc.values()):return 'SHARED_EDGE_WINDING'
    vs=set(map(int,np.asarray(f).reshape(-1))) if affected_vertices is None else set(affected_vertices)
    for v in vs:
        links=collections.defaultdict(set)
        for tri in f:
            if v not in tri:continue
            a,b=[int(x) for x in tri if x!=v];links[a].add(b);links[b].add(a)
        if any(len(n)>2 for n in links.values()):return 'NONMANIFOLD_VERTEX_LINK'
        if links:
            reached=set();stack=[next(iter(links))]
            while stack:
                a=stack.pop()
                if a in reached:continue
                reached.add(a);stack.extend(links[a]-reached)
            if len(reached)!=len(links):return 'DISCONNECTED_VERTEX_LINK'
    return None
def canon(t):
    t=tuple(map(int,t));return min(t,t[1:]+t[:1],t[2:]+t[:2])
def scan(first,second):
    value=Q.pair_scan(first,second,L,independent=True)
    bad=[r for r in value['contacts'] if r['classification'] in Q.ILLEGAL]
    return value,bad

def propose(inp):
    """Only C vertices/faces and requested N can enter this function."""
    v=np.asarray(inp['C_vertices'],np.float32);f=np.asarray(inp['C_faces_original'],np.int64)
    if f.ndim!=2 or f.shape[1]!=3 or not np.isfinite(v).all() or any(len(set(map(int,t)))!=3 for t in f):
        raise ValueError('Invalid indexed C')
    baseline=topology_reason(f)
    if baseline: return [],dict(status='BLOCKED_INPUT_TOPOLOGY',reason=baseline,generated_paths=0,unique_candidates=0,feasible=0,selected=0)
    inc=incidence(f);opened=[o[0] for _,o in sorted(inc.items()) if len(o)==1]
    eids={tuple(sorted((a,b))):i for i,(_,a,b) in enumerate(opened)}
    outgoing=collections.defaultdict(list)
    for ei,(_,a,b) in enumerate(opened):outgoing[a].append((ei,b))
    raw=collections.defaultdict(list)
    for e0,(_,u,m) in enumerate(opened):
        for e1,w in outgoing[m]:
            if len({u,m,w})==3:raw[tuple(sorted((u,m,w)))].append(dict(face=canon((w,m,u)),center=m,path_edges=[e0,e1]))
    rows=[];C=v[f];orig_sets={tuple(sorted(map(int,t))) for t in f}
    for ci,(key,origins) in enumerate(sorted(raw.items())):
        faces={x['face'] for x in origins};face=list(sorted(faces)[0]);tri=v[face][None]
        shape=PE.shape(tri);area=float(shape['area'][0]);angle=float(shape['angle'][0]);q=float(shape['q'][0])
        r=dict(candidate_id=ci,face_vertex_ids=face,corner_centers=sorted({x['center'] for x in origins}),
            originating_paths=[x['path_edges'] for x in origins],area=area,min_angle=angle,q=q if np.isfinite(q) else None,
            source_labels_used=False,feasible=False,selected=False,rejection=None,C_contacts=[],shared_C_edge_ids=[])
        reason='AMBIGUOUS_ORIENTATION' if len(faces)>1 else 'DUPLICATE_C_FACE' if key in orig_sets else None
        if reason is None and area<=1e-12*L*L:reason='AREA_DEGENERATE'
        if reason is None and angle<5:reason='MIN_ANGLE_LT5'
        if reason is None and q>10:reason='Q_GT10'
        if reason is None:reason=topology_reason(np.vstack((f,face)),face)
        shared=[eids[e] for e in incidence(np.array([face])) if e in eids]
        r['shared_C_edge_ids']=sorted(shared)
        if reason is None and len(shared)<2:reason='NOT_TWO_C_OPEN_EDGES'
        if reason is None:
            result,bad=scan(C,tri);r['C_contacts']=[dict(C_face=x['pair'][0],classification=x['classification']) for x in result['contacts']]
            if result['unresolved_or_degenerate_pairs']:reason='UNRESOLVED_C_PAIR'
            elif bad:reason='ILLEGAL_C_FINITE_SURFACE_CONTACT'
        if reason is None:r['feasible']=True
        r['rejection']=reason;rows.append(r)
    accepted=[];used_centers=set();used_C_edges=set()
    for r in sorted((r for r in rows if r['feasible']),key=lambda r:(r['q'],-r['area'],r['face_vertex_ids'])):
        reason=None
        if used_centers.intersection(r['corner_centers']):reason='CORNER_CAPACITY'
        elif used_C_edges.intersection(r['shared_C_edge_ids']):reason='C_OPEN_EDGE_CAPACITY'
        ff=np.asarray([x['face_vertex_ids'] for x in accepted],np.int64).reshape(-1,3)
        if reason is None:reason=topology_reason(np.vstack((f,ff,[r['face_vertex_ids']])),r['face_vertex_ids'])
        if reason is None and accepted:
            value,bad=scan(v[ff],v[r['face_vertex_ids']][None])
            if value['unresolved_or_degenerate_pairs']:reason='UNRESOLVED_PROPOSAL_PAIR'
            elif bad:reason='PROPOSAL_INTERSECTION_OR_OVERLAP'
        if reason is None:
            r['selected']=True;r['owner_edge_id']=min(r['shared_C_edge_ids']);r['alias_edge_ids']=[e for e in r['shared_C_edge_ids'] if e!=r['owner_edge_id']]
            accepted.append(r);used_centers.update(r['corner_centers']);used_C_edges.update(r['shared_C_edge_ids'])
        else:r['rejection']=reason
    eligible=collections.Counter(c for r in rows if r['feasible'] for c in r['corner_centers'])
    return rows,dict(status='COMPLETE',boundary_edges=len(opened),generated_paths=sum(len(a) for a in raw.values()),unique_candidates=len(rows),
        feasible=sum(r['feasible'] for r in rows),selected=len(accepted),ambiguous_corners=sum(n>1 for n in eligible.values()),
        rejections=dict(collections.Counter(r['rejection'] for r in rows if r['rejection'])),
        no_required_interface_claim=True,source_labels_used=False)

def freeze():
    p=OUT/'registration.json'
    if p.exists():return read(p)
    source=read(SOURCE);records=[dict(r,diagnostic_split='TRAIN') for r in source['records']]+[dict(r,diagnostic_split='DEV') for r in source['DEV_records']]
    assert len(records)==50
    protected={str(SOURCE):sha(SOURCE)}
    for r in records:
        for k in ('inference_npz','labels_npz'):protected[r[k]]=sha(r[k])
    code={str(Path(__file__)):sha(__file__),str(Path(PE.__file__)):sha(PE.__file__),**Q.definition_hashes()}
    reg=dict(status='FROZEN_BEFORE_GEOMETRY_AND_SOURCE_LABEL_DIAGNOSTIC',version=VERSION,records=records,protected=protected,code=code,L=L,
        rule='C directed open path u->v->w proposes reversed triangle(w,v,u), all vertices C; area>1e-12L2,minangle>=5,q<=10; exact finite-C pairs resolved/legal; no duplicate,edge/vertex nonmanifold or winding contradiction',
        selection='Unique unordered triangles; oriented cyclic consistency; sort q ascending,area descending,local face IDs; one proposal per corner center and C open edge; final edge/vertex topology and proposal pairs checked',
        interpretation='Optional geometric corner closure proposal; not proven source alias, required interface, universal all-open layout, or new generator input',
        source_diagnostic='After ALL50 C-only proposals saved: exact original C-vertex identity against unique source alias-owner faces; selected face precision/recall and alias-edge precision/recall; wrong proposals and missed truth preserved',
        budget=dict(conditions=50,CPU_workers=1,CPU_threads=1,model_calls=0,GPU_calls=0),
        no_threshold_tuning=True,no_default_integration=True)
    write(p,reg);write(OUT/'resources_before.json',resources());return reg
def verify(r):
    for p,h in {**r['protected'],**r['code']}.items():assert sha(p)==h,p
def source_diagnostic(r,allrows):
    """First semantic label reads occur only after geometry_complete.json exists."""
    assert (OUT/'geometry_complete.json').exists()
    summaries=[];proposal=[];missed=[]
    for record in r['records']:
        short=record['parent_id'][:8];d=arrays(record['labels_npz']);rr=allrows[short]
        gt_faces={int(d['edge_owner_face'][e]) for e in np.flatnonzero(d['source_alias_labels'])}
        truth={tuple(sorted(map(int,d['source_face_vertex_ids'][fi]))):fi for fi in gt_faces}
        true_alias=set(map(int,np.flatnonzero(d['source_alias_labels'])));pred_alias=set();matches=set();selected=[]
        for row in rr:
            srctri=tuple(sorted(map(int,d['C_source_vertex_ids'][row['face_vertex_ids']])));match=srctri in truth
            if row['selected']:
                selected.append(row);pred_alias.update(row['alias_edge_ids'])
                if match:matches.add(srctri)
            proposal.append(dict(parent_id=record['parent_id'],split=record['diagnostic_split'],**row,
                source_vertex_ids=list(srctri),exact_source_alias_face_match=match,source_alias_face_id=truth.get(srctri),
                exact_any_source_Y_face_match=any(tuple(sorted(map(int,d['source_face_vertex_ids'][fi])))==srctri for fi in d['free_ids'])))
        for key,fi in truth.items():
            if key not in matches:missed.append(dict(parent_id=record['parent_id'],split=record['diagnostic_split'],source_face_id=fi,source_vertex_ids=list(key),
                alias_edge_ids=[int(e) for e in np.flatnonzero(d['source_alias_labels']) if int(d['edge_owner_face'][e])==fi]))
        tp=len(matches);ap=len(pred_alias&true_alias)
        summaries.append(dict(parent_id=record['parent_id'],split=record['diagnostic_split'],K=int(d['K']),N=int(d['N']),
            source_alias_faces=len(truth),geometric_feasible=sum(x['feasible'] for x in rr),selected_proposals=len(selected),matched_alias_faces=tp,
            proposal_precision=tp/len(selected) if selected else None,alias_face_recall=tp/len(truth) if truth else None,
            true_alias_edges=len(true_alias),predicted_alias_edges=len(pred_alias),correct_alias_edges=ap,
            alias_edge_precision=ap/len(pred_alias) if pred_alias else None,alias_edge_recall=ap/len(true_alias) if true_alias else None,
            false_proposals=len(selected)-tp,missed_alias_faces=len(truth)-tp,
            false_alias_edges=sorted(pred_alias-true_alias),missed_alias_edges=sorted(true_alias-pred_alias)))
    totals=[]
    for split in ('TRAIN','DEV'):
        rows=[x for x in summaries if x['split']==split]
        sums={k:sum(x[k] for x in rows) for k in ('source_alias_faces','geometric_feasible','selected_proposals','matched_alias_faces','true_alias_edges','predicted_alias_edges','correct_alias_edges','false_proposals','missed_alias_faces')}
        for name,num,den in [('proposal_precision','matched_alias_faces','selected_proposals'),('alias_face_recall','matched_alias_faces','source_alias_faces'),('alias_edge_precision','correct_alias_edges','predicted_alias_edges'),('alias_edge_recall','correct_alias_edges','true_alias_edges')]:sums[name]=sums[num]/sums[den] if sums[den] else None
        totals.append(dict(split=split,conditions=len(rows),**sums))
    csvwrite(OUT/'per_condition.csv',summaries);csvwrite(OUT/'proposal_diagnostic.csv',proposal)
    csvwrite(OUT/'false_proposals.csv',[x for x in proposal if x['selected'] and not x['exact_source_alias_face_match']],list(proposal[0]))
    csvwrite(OUT/'missed_source_alias_faces.csv',missed,['parent_id','split','source_face_id','source_vertex_ids','alias_edge_ids'])
    write(OUT/'diagnostic_summary.json',dict(scope='Source completion correspondence diagnostic, not geometric impossibility or required-interface success',groups=totals))
    return totals
def run():
    r=freeze();verify(r)
    if (OUT/'completion.json').exists():return read(OUT/'completion.json')
    if (OUT/'started.json').exists():raise RuntimeError('Existing attempt; no silent repeat')
    write(OUT/'started.json',dict(unix=time.time(),pid=os.getpid(),model_calls=0));start=time.perf_counter();allrows={};summaries=[]
    try:
        for record in r['records']:
            key=record['parent_id'][:8];inp=arrays(record['inference_npz']);rows,summary=propose(inp);allrows[key]=rows
            write(OUT/'conditions'/key/'geometry.json',dict(parent_id=record['parent_id'],split=record['diagnostic_split'],summary=summary,candidates=rows))
            summaries.append(dict(parent_id=record['parent_id'],split=record['diagnostic_split'],**summary))
        write(OUT/'geometry_complete.json',dict(status='COMPLETE',conditions=50,source_label_arrays_read=0,results=summaries))
        totals=source_diagnostic(r,allrows);verify(r)
        assert 'torch' not in sys.modules
        write(OUT/'resources_after.json',resources())
        result=dict(status='COMPLETE',version=VERSION,conditions=50,seconds=time.perf_counter()-start,groups=totals,
            protected_hashes_verified=len(r['protected']),code_hashes_verified=len(r['code']),model_calls=0,GPU_calls=0,new_noise=0,
            no_rule_tuning=True,no_default_integration=True,torch_imported=False,source_diagnostic_after_all_geometry=True,
            files={str(p):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='completion.json'})
        write(OUT/'completion.json',result);return result
    except BaseException as e:
        write(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),model_calls=0,no_silent_retry=True));raise
def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','run','status']);a=p.parse_args()
    result=freeze() if a.stage=='freeze' else run() if a.stage=='run' else (read(OUT/'completion.json') if (OUT/'completion.json').exists() else dict(status='NOT_COMPLETE'))
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','records','protected','code')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
