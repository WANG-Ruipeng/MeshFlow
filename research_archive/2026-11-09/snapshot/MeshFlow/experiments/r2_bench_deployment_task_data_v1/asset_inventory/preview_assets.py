"""Read-only prior-source inventory and bounded 16-parent CPU preview.

No simplification, finite-pair rescoring, noise, model imports or GPU work.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import csv, json, hashlib, time, textwrap
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
EXP=ROOT/'experiments/r2_bench_user_data_v2_expansion_v1'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def path(p):
    s=str(p).replace('\\','/')
    if s.startswith('/mnt/e/MeshFlow/'):
        return ROOT/s[len('/mnt/e/MeshFlow/'):]
    return Path(s)
def sha(p):
    return hashlib.sha256(path(p).read_bytes()).hexdigest()
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def table(p,rows):
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
def arrays(p):
    with np.load(path(p),allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}
def canonical_hash(a):
    v=a['model_vertices'];f=a['source_faces']
    # Exact unordered geometric triangles; no tolerance, no similarity inference.
    triangles=sorted(tuple(sorted(tuple(float(c) for c in pt) for pt in tri)) for tri in v[f])
    return hashlib.sha256(np.asarray(triangles,dtype='<f8').tobytes()).hexdigest()
def components(f):
    parent=list(range(len(f)));byedge={}
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,tri in enumerate(f):
        for j in range(3):
            e=tuple(sorted((int(tri[j]),int(tri[(j+1)%3]))))
            if e in byedge:
                a,b=root(i),root(byedge[e]);parent[a]=b
            else:byedge[e]=i
    c=Counter(root(i) for i in range(len(f)))
    return sorted(c.values(),reverse=True)

def render(row):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    a=arrays(row['clean_case_path']);b=arrays(row['agg7_target_case_path'])
    fig=plt.figure(figsize=(16,10),dpi=120)
    axes=[]
    for idx,(data,title,az) in enumerate([(a,'CLEAN SOURCE',-60),(a,'CLEAN SOURCE',120),(b,'OLD QEM agg7 TARGET (HELD)',-60),(b,'OLD QEM agg7 TARGET (HELD)',120)]):
        ax=fig.add_subplot(2,2,idx+1,projection='3d');axes.append(ax)
        v=data['model_vertices'];f=data['source_faces'];t=v[f][:,:,[0,2,1]]
        mesh=Poly3DCollection(t,facecolor='#aebcc5',edgecolor='#5e6e75',linewidth=.20,alpha=1)
        ax.add_collection3d(mesh)
        ax.set_xlim(-2.9,2.9);ax.set_ylim(-2.9,2.9);ax.set_zlim(-2.9,2.9)
        ax.set_box_aspect((1,1,1),zoom=1.3);ax.view_init(elev=20,azim=az);ax.set_axis_off()
        ax.set_title(title+' | N='+str(len(f)),fontsize=12)
    fig.suptitle(row['parent_id']+'\n'+str(row['N_clean'])+' source faces; '+str(row['edge_component_face_counts'])+' edge components\n'+textwrap.fill('agg7 held: '+', '.join(row['agg7_reasons']),120),fontsize=11)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=.88,wspace=-.05,hspace=-.05)
    dest=OUT/'gallery_v2'/(row['parent_id']+'.png');dest.parent.mkdir(exist_ok=True)
    fig.savefig(dest,facecolor='white');plt.close(fig)
    return dict(parent_id=row['parent_id'],path=str(dest),sha256=sha(dest),view_contract='Source and held QEM: (X,Z,Y), elev20 az-60/120, all axes [-2.9,2.9], zoom1.3, identical scale all parents',input_hashes={row['clean_case_path']:row['clean_case_sha256'],row['agg7_target_case_path']:row['agg7_target_case_sha256']})

def main():
    start=time.monotonic()
    merged=read(ROOT/'experiments/r2_bench_user_relation_rebuild_v1/proposed_merged_manifest.json')
    common=read(ROOT/'experiments/r2_bench_parts_convergence_v1/data/manifest.json')
    prior=read(EXP/'all_candidate_manifest.json')
    high=read(EXP/'high_source/audited.json')['rows']
    supply=read(EXP/'supply/manifest.json')
    current={r['parent_id'] for r in merged['train']}
    dev={r['parent_id'] for r in merged['dev']}
    common_train={r['parent_id'] for r in common['train']}
    old_eval=read(ROOT/'experiments/r2_bench_joint_shift_v1/eval_manifest.json')
    if isinstance(old_eval,dict):
        old_eval=old_eval.get('requests',old_eval.get('rows',[]))
    evaluation={r['parent_id'] for r in old_eval}
    pmap={r['parent_id']:r for r in prior['parents']}
    clean_high=[r for r in high if r['source_finite_pass']]
    rows=[]
    for r in clean_high:
        uid=r['parent_id']
        if uid in current:continue
        p2=read(EXP/'parents'/uid/'parent.json')
        p7=read(EXP/'standard_parents'/uid/'parent.json')
        fid2=read(p2['fidelity_path']);fid7=read(p7['fidelity_path'])
        a=arrays(r['clean_case_path'])
        assert sha(r['clean_case_path'])==r['clean_case_sha256']
        assert sha(p7['target_case_path'])==p7['target_case_sha256']
        q=read(r['source_geometry_path'])
        counts=components(a['source_faces'])
        rows.append(dict(r,common12000_exposed=uid in common_train,old_dev=uid in dev,historical_eval_parent=uid in evaluation,current_relation_parent=uid in current,
            edge_components=len(counts),edge_component_face_counts=counts,source_finite_pairs=len(q['illegal_pairs']),source_unresolved_pairs=len(q['unresolved_pairs']),
            exact_clean_geometry_sha256=canonical_hash(a),
            agg2_reasons=p2['reasons'],agg2_fidelity=fid2,agg2_target_case_path=p2['target_case_path'],agg2_target_case_sha256=p2['target_case_sha256'],
            agg7_reasons=p7['reasons'],agg7_fidelity=fid7,agg7_target_case_path=p7['target_case_path'],agg7_target_case_sha256=p7['target_case_sha256'],
            new_structure_relation='UNREVIEWED: inspect complete source; no semantics inferred from UID',eligible_for_immediate_training=False,
            candidate_action='BOUNDED_COMPONENT_PRESERVING_REBUILD_REQUIRES_ROOT_PROTOCOL',independent_shape_family='UNKNOWN'))
    assert len(rows)==16
    hashes={r['exact_clean_geometry_sha256']:[] for r in rows}
    for r in rows:hashes[r['exact_clean_geometry_sha256']].append(r['parent_id'])
    for r in rows:r['exact_duplicate_source_uids']=hashes[r['exact_clean_geometry_sha256']]
    table(OUT/'high_source_candidates.csv',rows);save(OUT/'high_source_candidates.json',dict(rows=rows))
    table(OUT/'current_reusable_parents.csv',[dict(parent_id=uid,condition_count=sum(r['parent_id']==uid for r in merged['train']),common12000_exposed=uid in common_train,historical_eval_parent=uid in evaluation,old_dev=uid in dev,independent_shape_family='UNKNOWN',**{k:pmap[uid].get(k) for k in ['target_case_path','target_case_sha256','geometry_path','lineage_path','N']}) for uid in sorted(current)])
    reasons=Counter(reason for r in rows for reason in r['agg7_reasons'])
    summary=dict(current_parents=len(current),current_conditions=len(merged['train']),old_dev_parents=len(dev),historical_eval_parents=len(evaluation),common12000_parents=len(common_train),
      archive_uid_count=560,archive_existing_versions_per_uid=1,previous_all_target_proposals=184,
      previous_high_source_checked=104,previous_high_source_finite_pass=len(clean_high),remaining_clean_high_source_parents=len(rows),
      candidate_N_clean_min=min(r['N_clean'] for r in rows),candidate_N_clean_max=max(r['N_clean'] for r in rows),agg7_reasons=dict(reasons),
      new_current_source_ids=len(rows),new_common12000_source_ids=sum(not r['common12000_exposed'] for r in rows),old_dev_overlap=sum(r['old_dev'] for r in rows),historical_eval_overlap=sum(r['historical_eval_parent'] for r in rows),
      exact_duplicate_groups=[v for v in hashes.values() if len(v)>1],
      supply_exhaustion='Known N128..256 existing candidates exhausted under previous gates; high-N source geometry is available but no accepted compact reconstruction yet; this does not prove all 560 assets irreparable.',
      semantic_independence='Source-ID novelty only. Shape-family and novel structural-relation count require actual review.',model_calls=0,new_geometry_scorings=0,simplifications=0,render_workers=2)
    save(OUT/'summary.json',summary)
    with ProcessPoolExecutor(max_workers=2) as pool:pages=list(pool.map(render,rows))
    save(OUT/'source_gallery_manifest_v2.json',dict(pages=pages,source_manifest_sha256=sha(OUT/'high_source_candidates.json'),reason='Improve viewing scale and wrap long titles; initial previews preserved.'))
    inputs=[ROOT/'experiments/r2_bench_user_relation_rebuild_v1/proposed_merged_manifest.json',ROOT/'experiments/r2_bench_parts_convergence_v1/data/manifest.json',EXP/'all_candidate_manifest.json',EXP/'high_source/audited.json',EXP/'supply/manifest.json']
    save(OUT/'receipt.json',dict(status='INVENTORY_AND_PREVIEW_COMPLETE_PENDING_ACTUAL_REVIEW',source_hashes={str(p):sha(p) for p in inputs},code_sha256=sha(__file__),seconds=time.monotonic()-start,models=0,training=0,new_geometry_scoring=0,simplifications=0))
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
