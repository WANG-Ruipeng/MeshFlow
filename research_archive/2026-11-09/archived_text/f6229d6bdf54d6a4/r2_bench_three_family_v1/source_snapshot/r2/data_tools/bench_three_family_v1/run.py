"""Close or verify this bounded data pilot; never dispatch a neural call on resume."""
from __future__ import annotations
from .common import *
import argparse, csv, difflib, io, zipfile

CODE = Path(__file__).parent

def write_once(path, text):
    path=resolve(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert path.read_text('utf-8-sig')==text, str(path)
    else:path.write_text(text,'utf-8-sig' if path.suffix=='.csv' else 'utf-8')

def table(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=fields,lineterminator='\n')
    w.writeheader();w.writerows(rows);write_once(path,buf.getvalue())

def verify():
    c=read(OUT/'COMPLETE.json')
    for path,h in c['files'].items():assert sha(path)==h,path
    for path,h in c['protected_sources'].items():assert sha(path)==h,path
    assert c['generator_calls']==c['D0_calls']==c['training_updates']==0
    assert c['head_forward_calls']==1 and c['new_training_parents']==0
    print(json.dumps(dict(status=c['status'],verified_files=len(c['files']),
        protected_sources=len(c['protected_sources']),new_training_parents=0,
        head_forward_calls_this_resume=0,generator_calls_this_resume=0),ensure_ascii=False))
    return c

def closeout():
    if (OUT/'COMPLETE.json').exists():return verify()
    protocol=freeze();targets=read(OUT/'selected_targets.json')['rows']
    conditions=read(OUT/'tasks/reviewed_manifest.json')['rows']
    contract=read(OUT/'production_contract/COMPLETE.json')
    audit=read(OUT/'boundary_method_check/final_array_audit.json')
    assert audit['status']=='PASS_ARRAY_LINEAGE_AND_TOPOLOGY_CONSISTENCY'
    results={r['condition_id']:r for r in contract['rows']}
    assert len(targets)==3 and len(conditions)==6 and contract['total_head_forward_calls']==1
    assert all(contract[k]==0 for k in ('generator_calls','D0_calls','OT_calls','backward_calls'))
    # These frozen outcomes are a construction/encoding result, not a trained-model comparison.
    passed=[r for r in conditions if results[r['condition_id']]['status']=='EXACT_TARGET_BIJECTION_PASS']
    assert len(passed)==1 and passed[0]['recipe_id']=='R2'
    assert not any(r['recipe_id']=='R1' for r in passed)
    parent_rows=[]
    for r in targets:
        f=r['fidelity'];prefix=r['parent_id'][:6]
        outcome={'27eb79':'LOW_N_GEOMETRY_PASS_R1_LAYOUT_BLOCKED_R2_HELDOUT_PASS',
                 '945214':'BLOCKED_N448','585673':'HOLD_FIDELITY_AND_AREA'}[prefix]
        parent_rows.append(dict(parent_id=r['parent_id'],family=r['source_parent']['family'],
            source_N=r['N_original'],target_N=r['N'],method=r['method'],
            geometry_pass=r['geometry_pass'],N_pass=r['current_N_pass'],
            source_components=f['source_components'],target_components=f['target_components'],
            finite_illegal=r['finite_illegal'],finite_unresolved=r['finite_unresolved'],
            max_bidirectional_probe_L=max(f['source_to_target_max_L'],f['target_to_source_max_L']),
            area_ratio=f['area_ratio'],actual_visual_review='AI_PROXY_PLAUSIBLE',
            final_status=outcome,new_training_parent=False,target_path=r['target_case_path'],
            target_sha256=r['target_case_sha256']))
    task_rows=[];detailed=[];edge_rows=[]
    for r in conditions:
        cid=r['condition_id'];z=results[cid];eligible=r['eligible_for_production_contract']
        packing=read(OUT/'production_contract/conditions'/cid/'packing.json') if eligible else {}
        status=(z['status'] if eligible else ('BLOCKED_N' if r['N']>256 else 'HOLD_TARGET_GEOMETRY'))
        if status=='EXACT_TARGET_BIJECTION_PASS':status='ENCODABLE_HELDOUT_PENDING_TRAIN_C'
        tr=dict(condition_id=cid,parent_id=r['parent_id'],recipe_id=r['recipe_id'],region=r['proposed_region'],
            N=r['N'],K=r['K'],free_faces=r['N']-r['K'],C_open_edges=r['C_open_boundary_count'],
            source_owner_edges=r['source_owner_count'],source_alias_edges=r['source_alias_count'],
            head_forward_calls=z['head_forward_calls'],predicted_active_edges=z.get('active','NA_NOT_RUN'),
            production_status=z['status'],final_status=status,training_pool_membership=False,
            effective_eval_group='SAME_PARENT_PENDING_NO_ACCEPTED_TRAIN_C' if r['recipe_id']=='R2' else 'TRAIN_STAGING_BLOCKED',
            C_Y_exact_complement=r['C_Y_exact_complement'],
            pack_roundtrip=packing.get('source_winding_roundtrip','NA_INCOMPATIBLE_OR_NOT_RUN'),
            input_sha256=r['input_sha256'],case_sha256=r['case_sha256'])
        task_rows.append(tr)
        detailed.append(dict(r,**{k:v for k,v in tr.items() if k not in ('source_owner_edges','source_alias_edges')},
            status=status,training_ready=False,future_training_pool_membership=False))
        for e in packing.get('active_edge_source_diagnostics',[]):
            edge_rows.append(dict(condition_id=cid,edge_id=e['edge_id'],
                C_vertex_ids=json.dumps(e['C_vertex_ids']),source_endpoint_ids=json.dumps(e['source_endpoint_ids']),
                legal_source_supported=e['source_legal_supported'],source_alias=e['source_alias'],
                incidence_candidate_count=len(e['broader_source_incidence_face_ids']),source_face=e['source_face']))
    assert len(edge_rows)==20 and all(r['incidence_candidate_count']==0 for r in edge_rows)
    save(OUT/'results/per_parent.json',dict(rows=parent_rows))
    save(OUT/'results/per_condition.json',dict(rows=task_rows))
    # Machine-readable result exports; their owning calculations remain in JSON receipts.
    table(OUT/'results/per_parent.csv',parent_rows);table(OUT/'results/per_condition.csv',task_rows)
    table(OUT/'results/active_edge_incompatibility.csv',edge_rows)
    save(OUT/'tasks/final_manifest.json',dict(rows=detailed,training_pool=[],new_training_parents=0,
        encodable_heldout_conditions=1,valid_same_parent_transfer_conditions=0,
        authoritative_over_intermediate_status=True,production_default_unchanged=True))
    cov=read(OUT/'n_budget/coverage.json');nrows=[]
    for cap in (256,384,512,799,800):
        rs=[r for r in cov['per_condition'] if r['recipe_id']=='R1' and r['final_source_task_status']=='ACCEPT_SOURCE_TASK_PENDING_ADAPTER'
            and r['effective_eval_group']=='TRAIN_KNOWN_C' and 128<=r['N']<=cap]
        nrows.append(dict(cap=cap,source_qualified_train_parents=len({r['parent_id'] for r in rs}),
            visual_families=len({r['family_id'] for r in rs}),source_qualified_train_C=len(rs),
            positive_owner_parents=sum(r['source_owner_count']>0 for r in rs),
            positive_owner_families=len({r['family_id'] for r in rs if r['source_owner_count']>0}),
            production_ready='NOT_TESTED_HIGH_N',budget_boundary='800_hits_CFG_slot' if cap==800 else 'static_only'))
    assert [r['source_qualified_train_parents'] for r in nrows]==[0,3,6,9,10],nrows
    table(OUT/'results/N_coverage.csv',nrows);save(OUT/'results/N_coverage.json',dict(rows=nrows))
    parent_images=read(OUT/'gallery/parents_fullframe_manifest.json')['rows']
    condition_images=read(OUT/'gallery/conditions_manifest.json')['rows']
    def pic(r):
        p=resolve(r['image_path']);return f'<figure><img loading="lazy" src="{p.name}"><figcaption>{p.stem}</figcaption></figure>'
    html='<!doctype html><meta charset="utf-8"><title>Bench三家族完整目标与条件</title><style>body{font:16px sans-serif;max-width:1500px;margin:auto;padding:20px}img{max-width:100%}figure{margin:28px 0}</style><h1>三家族：实际完整目标与C/Y</h1><p>蓝色C；完整目标页为来源与新版对照。AI_PROXY实际图审；无生成器输出。以下仅为完整画幅正式评阅图。</p>'
    html+=''.join(pic(r) for r in parent_images+condition_images)
    write_once(OUT/'gallery/review_index.html',html)
    assert (OUT/'REPORT.md').exists() and (CODE/'README.md').exists()
    snap=OUT/'source_snapshot';changes=[];hashes={}
    files=list(CODE.glob('*.py'))+[CODE/'README.md']
    hp=read(OUT/'production_contract/HEAD_PROTOCOL.json')
    dependencies={resolve(p) for p in hp['code_sha256'] if resolve(p).suffix=='.py'}
    dependencies.add(ROOT/'r2/data_tools/bench_parts_v1/quality.py')
    # Include the actual finite-face scorer used by quality.py, with frozen hashes.
    from r2.data_tools.bench_parts_v1 import quality as Q
    detector=Q.detector()
    for name in Q.FROZEN_HASHES:
        p=Path(detector.__file__).parent/name
        assert sha(p)==Q.FROZEN_HASHES[name],name
        dependencies.add(p)
    for p in sorted(set(files)|dependencies):
        rel=p.relative_to(ROOT) if p.is_relative_to(ROOT) else Path('external_frozen_scorer')/p.name
        text=p.read_text('utf-8-sig');dest=snap/rel
        write_once(dest,text);hashes[str(p)]=sha(p)
        if p in files:changes.extend(difflib.unified_diff([],text.splitlines(True),fromfile='/dev/null',tofile=str(rel).replace('\\','/')))
    write_once(OUT/'code.diff',''.join(changes));save(OUT/'source_hashes.json',hashes)
    protected=dict(protocol['protected_sources'])
    for p,h in hp['code_sha256'].items():protected[p]=h
    for p in dependencies:protected[str(p)]=sha(p)
    for p,h in protected.items():assert sha(p)==h,p
    resources=read(OUT/'production_contract/resources_after_run.json')
    event('CLOSEOUT',new_training_parents=0,head_forward_calls=1,generator_calls=0,D0_calls=0,
        CPU_geometry_workers=3,CPU_render_workers_per_dispatch=2,overlapping_render_workers_max=4,
        GPU_workers=0,display_fix='Earlier clipped parent previews retained; actual admission uses fullframe pages.',
        interpretation='1 low-N target passes, R1 production layout incompatible; R2 heldout encoding passes; 2 other parents blocked/held.')
    allfiles={str(p):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='COMPLETE.json' and p.suffix!='.zip'}
    # Child COMPLETE receipts are also evidence.
    for p in OUT.rglob('COMPLETE.json'):allfiles[str(p)]=sha(p)
    save(OUT/'COMPLETE.json',dict(status='COMPLETE_DATA_PILOT_NO_TRAINING_READY_PARENT',parents=3,conditions=6,
        geometry_and_N_pass_parents=1,production_encodable_conditions=1,new_training_parents=0,
        head_forward_calls=1,generator_calls=0,D0_calls=0,OT_calls=0,training_updates=0,
        files=allfiles,protected_sources=protected,reviewer='AI_PROXY_NONBLIND',resources_after=resources,
        resume='python -B -m r2.data_tools.bench_three_family_v1.run --resume'))
    return verify()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--resume',action='store_true');p.add_argument('--closeout',action='store_true')
    a=p.parse_args()
    if (OUT/'COMPLETE.json').exists():verify()
    elif a.closeout:closeout()
    else:print('Closure pending. Run --closeout after all frozen receipts and actual visual reviews are present; this entry never replays a head or dispatches generation.')
