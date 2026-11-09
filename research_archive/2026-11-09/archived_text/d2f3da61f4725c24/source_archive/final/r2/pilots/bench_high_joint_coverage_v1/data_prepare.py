"""CPU-only wider natural-N source data, preserving all historical geometry.

Existing accepted R1 conditions are consolidated without selecting new triangles,
repairing targets, running finite-pair detectors, or drawing evaluation noise.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse, collections, copy, csv, hashlib, io, json, sys, time
from pathlib import Path
import numpy as np
from r2.pilots.bench_natural_n_recipe_v1 import data_prepare as D

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'experiments/r2_bench_high_joint_coverage_v1'
DATA = OUT/'data'
PRIOR = ROOT/'experiments/r2_bench_natural_n_recipe_v1/data/manifest.json'
TASK = ROOT/'experiments/r2_bench_deployment_task_data_v1'
SOURCE_ROWS = TASK/'dataset/source_tasks_pending_adapter/manifest.json'
REGISTRY = ROOT/'experiments/r2_bench_exposure_allocation_v1/data_expansion/candidate_registry.csv'
SELECTED = ('9452149917cb2e27f4fb4dee5181bee','af264c02f69566cdf4fb4dee5181bee',
            'bcb4f576747e10bdd7f6e86d7d335698','77aa33479c2ec92f610c2a68437007d6')
VERSION = 'BENCH_HIGH_JOINT_COVERAGE_DATA_V1'
read, sha, arrays, write, save, protect = D.read, D.sha, D.arrays, D.write, D.save, D.protect


def csvwrite(path, rows):
    fields = sorted({k for r in rows for k in r})
    f=io.StringIO(newline='');w=csv.DictWriter(f, fieldnames=fields);w.writeheader()
    w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    value=f.getvalue().encode('utf-8')
    if p.exists():assert p.read_bytes()==value
    else:p.write_bytes(value)


def registry():
    with REGISTRY.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def inventory():
    if not (DATA/'resources_before.json').exists():write(DATA/'resources_before.json',D.resources())
    old=read(PRIOR);pool={r['parent_id'] for r in old['train']}
    held={r['parent_id'] for r in old['dev']+old['high_holdout']}
    held|={q['parent_id'] for q in read(PRIOR.parent/'eval_manifest.json')['requests'] if q['report_group'] in ('DEV','HOLDOUT','HIGH_HOLDOUT')}
    reg=registry();rows=[]
    for r in reg:
        p=r['parent_id'];n=int(r['source_N']);selected=p in SELECTED
        if p in pool:reason='EXISTING_TRAIN_RETAINED_NO_NEW_PARENT'
        elif p in held:reason='PROTECTED_ADAPTATION_HOLDOUT'
        elif r['source_complete']!='PLAUSIBLE':reason='HOLD_SOURCE_COMPLETENESS'
        elif not 257<=n<=799:reason='OUTSIDE_NATURAL_HIGH_CONTRACT'
        elif selected:reason='SELECTED_ACCEPTED_R1_ONE_REPRESENTATIVE_PER_FAMILY'
        elif r['visual_family_id']=='ORNAMENTAL_END_FRAME_SLATTED':reason='OLD_R1_C_NONMANIFOLD_OR_SEMANTIC_CONTAMINATION_NO_REBUILD_THIS_ROUND'
        else:reason='NOT_SELECTED_NEAR_VARIANT_OF_SELECTED_945'
        rows.append(dict(parent_id=p,N=n,face_bucket=n//20,visual_family=r['visual_family_id'],
            source_complete=r['source_complete'],source_structure=r['source_structure'],
            current_train=p in pool,protected_holdout=p in held,selected=selected,decision=reason,
            source_case_path=r['source_case_path'],source_case_sha256=r['source_case_sha256'],
            historical_scope=r['current_historical_exposure_scope'],official_split=r['official_split'],
            official_checkpoint_exposure='UNKNOWN',family_limit=r['family_independence_status']))
    assert set(SELECTED).isdisjoint(pool|held)
    write(DATA/'candidate_preregistration.json',dict(schema=VERSION,status='FROZEN_BEFORE_CONSOLIDATION',
        selected_parent_ids=list(SELECTED),rows=rows,selection_rule='Use four already accepted complete USER seat R1 conditions; one representative per recorded near-variant family; no target modification or new region search.',
        near_variant_groups=[['1d6150ae91a07cf6f4fb4dee5181bee','4eb3dc26aad9ace8f4fb4dee5181bee','d7d6a7fae6003dd6f4fb4dee5181bee'],
                             ['9452149917cb2e27f4fb4dee5181bee','ddd142d8dcd4e95bf4fb4dee5181bee','f540ad990dbbd600f4fb4dee5181bee']],
        retained_prior_HIGH=[r['parent_id'] for r in old['train'] if r['replay_group']=='HIGH'],
        blocked_bucket_goals={str(b):'NO_ADDITIONAL_TRUSTED_INDEPENDENT_SOURCE_IN_EXISTING_AUDIT' for b in (15,18,19)},
        excluded_family_note='bcb is an independent source UID but visually related to old7b58/c976 loop-frame family; not a brand-new family claim.',
        protected_eval_parents=sorted(held),prior_manifest=str(PRIOR),prior_manifest_sha256=sha(PRIOR),
        registry_path=str(REGISTRY),registry_sha256=sha(REGISTRY),no_simplification=True,no_new_region=True,
        no_generated_output_selection=True,model_calls=0))
    csvwrite(DATA/'source_inventory.csv',rows)
    return dict(status='PREREGISTERED',selected=4,prior_HIGH=2,total_planned_HIGH=6,total_planned_train=30,
                new_N=[int(next(r['source_N'] for r in reg if r['parent_id']==p)) for p in SELECTED],model_calls=0)


def consolidate(row, protected):
    from r2.data_tools.bench_parts_v1.construct import partition
    for key in ('source_case','source_finite_receipt','input','paired','quality','selection','review_source'):
        protect(protected,row[key+'_path'],row[key+'_sha256'])
    a=arrays(row['source_case_path']);oldpair=arrays(row['paired_path']);quality=read(row['quality_path'])
    receipt=read(row['source_finite_receipt_path'])
    assert not receipt['illegal_pairs'] and not receipt['exact_zero_area_faces']
    assert not receipt.get('unresolved_pairs',[])
    assert not quality['reasons'] and quality['mapping']=='PASS_BITWISE_C_PLUS_Y_EQUALS_SOURCE_M'
    assert row['semantic_decision']=='ACCEPT_SOURCE_TASK' and row['geometry_status']=='PASS_PENDING_SEMANTIC_REVIEW'
    pair=partition(a,oldpair['known_ids'])
    for key in pair:assert np.array_equal(pair[key],oldpair[key]),key
    case={**a,**pair};edges=D.boundary(pair['C_faces_original']);ne=len(edges)
    occ=np.zeros(ne,bool);supported=np.zeros(ne,bool);alias=np.zeros(ne,bool)
    owner=np.full(ne,-1,np.int64);triple=np.full((ne,3),-1,np.int64)
    for it in quality['source_interface_rows']:
        e=int(it['edge_id']);occ[e]=it['role']=='OWNER';alias[e]=it['role']=='ALIAS'
        supported[e]=it['role'] in ('OWNER','ALIAS');owner[e]=int(it['free_source_face_id']);triple[e]=it['source_triple']
    case.update(occupied_labels=occ,source_supported_labels=supported,source_alias_labels=alias,
                edge_owner_face=owner,edge_source_triple=triple,edge_records=edges,
                source_third_in_C=np.asarray([bool(ok and x[2] in set(pair['C_source_vertex_ids'])) for ok,x in zip(supported,triple)],bool))
    cid=row['parent_id']+'__'+row['proposed_region']+'__HIGH_JOINT_V1'
    dest=DATA/'conditions'/cid
    save(dest/'case.npz',case)
    core={k:case[k] for k in ('C_vertices','C_faces_original','C_faces_model','N','K')}
    save(dest/'input.npz',core)
    save(dest/'metadata.npz',{k:case[k] for k in ('C_source_vertex_ids','known_ids')})
    from meshflow_control.source_winding import capture_source,validate_source
    original,canonical=core['C_faces_original'],core['C_faces_model']
    perm=np.asarray([[np.flatnonzero(original[i]==v).item() for v in face] for i,face in enumerate(canonical)],np.int64)
    source=capture_source(core['C_vertices'][original],source_vertex_ids=case['C_source_vertex_ids'][original],
                          corner_permutation=perm,source_face_ids=case['known_ids'])
    validate_source(source,case['C']);save(dest/'source_condition.npz',source)
    reg=next(x for x in registry() if x['parent_id']==row['parent_id'])
    result=dict(parent_id=row['parent_id'],condition_id=cid,N=int(case['N']),K=int(case['K']),
        replay_group='HIGH',part_type='SEAT_SURFACE',split='HIGH_TRAIN_NEW',eval_group='HIGH_TRAIN_NEW',report_group='HIGH_TRAIN_NEW',
        case_path=str(dest/'case.npz'),case_sha256=sha(dest/'case.npz'),
        input_path=str(dest/'input.npz'),input_sha256=sha(dest/'input.npz'),
        paired_path=str(dest/'case.npz'),paired_sha256=sha(dest/'case.npz'),
        metadata_path=str(dest/'metadata.npz'),metadata_sha256=sha(dest/'metadata.npz'),
        source_condition_path=str(dest/'source_condition.npz'),source_condition_sha256=sha(dest/'source_condition.npz'),
        source_case_path=row['source_case_path'],source_case_sha256=row['source_case_sha256'],
        quality_file=row['quality_path'],quality_sha256=row['quality_sha256'],
        source_finite_receipt_path=row['source_finite_receipt_path'],source_finite_receipt_sha256=row['source_finite_receipt_sha256'],
        boundary_edges=ne,closed_C=ne==0,full_seat_review='PENDING_NEW_ACTUAL_VIEW',review_status='PENDING',training_ready=False,
        native_array_compatible=True,eligibility='EXACT_SOURCE_PENDING_ACTUAL_VIEW',admission_version=VERSION,N_contract=[128,799],
        source_geometry_modified=False,model_input_policy='C,N,known/free role only; no source-owner layout',
        source_labels_usage='DIAGNOSTIC_METADATA_NOT_GENERATOR_INPUT',extent_description=row['proposed_region'],
        history_scope=reg,visual_family_id=reg['visual_family_id'],prior_R1_condition_id=row['condition_id'],
        prior_R1_image_path=row['image_path'],prior_R1_image_sha256=row['image_sha256'],
        prior_R1_review_observation=row['review_observation'],prior_R1_semantic_decision=row['semantic_decision'],
        source_partition_unchanged=True)
    check=D.verify_case(result)
    check.update(full_target_unchanged=np.array_equal(case['full_target'],a['full_target']),
        source_R1_C_Y_unchanged=all(np.array_equal(case[k],oldpair[k]) for k in ('C','Y','known_ids','free_ids')),
        full_finite_receipt_reused=True,new_full_geometry_tests=0,new_query_or_noise_arrays=0)
    assert check['full_target_unchanged'] and check['source_R1_C_Y_unchanged']
    return result,check


def prepare():
    if (DATA/'candidates.json').exists():return dict(status='EXISTING_CONSOLIDATED_CANDIDATES_NO_OVERWRITE')
    inventory();prereg=read(DATA/'candidate_preregistration.json');assert prereg['selected_parent_ids']==list(SELECTED)
    protected={};protect(protected,PRIOR);protect(protected,SOURCE_ROWS);protect(protected,REGISTRY)
    protect(protected,TASK/'split_audit/new_data_split.json')
    source=read(SOURCE_ROWS)['rows'];rows=[];checks=[];start=time.perf_counter()
    for pid in SELECTED:
        r=next(x for x in source if x['parent_id']==pid and x['condition_id'].endswith('__R1'))
        row,check=consolidate(r,protected);rows.append(row);checks.append(check)
        print(json.dumps({k:row[k] for k in ('parent_id','N','K','boundary_edges','closed_C')},ensure_ascii=False),flush=True)
    old=read(PRIOR);old_ids={r['parent_id'] for r in old['train']}
    assert len({r['parent_id'] for r in rows})==4 and not old_ids&set(SELECTED)
    assert not set(SELECTED)&set(prereg['protected_eval_parents'])
    for p,h in protected.items():assert sha(p)==h
    write(DATA/'candidates.json',dict(schema=VERSION,status='CPU_CASES_READY_PENDING_ACTUAL_SOURCE_REVIEW',rows=rows,protected=protected))
    write(DATA/'cpu_checks.json',dict(status='PASS',rows=checks,original26_exactly_preserved=True,new_parent_ids=list(SELECTED),
        heldout_parent_overlap=[],whole_source_new_scores=0,source_mutations=0,noise_arrays=0,model_calls=0,seconds=time.perf_counter()-start))
    return dict(status='CPU_CASES_READY_PENDING_ACTUAL_VIEW',new_cases=4,model_calls=0)


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    rendered=[]
    for r in read(DATA/'candidates.json')['rows']:
        a=arrays(r['case_path']);whole=a['model_vertices'][a['source_faces']]
        c=whole[a['known_ids']];y=whole[a['free_ids']]
        dest=DATA/'gallery'/(r['parent_id']+'.png');dest.parent.mkdir(parents=True,exist_ok=True)
        assert not dest.exists(),'Do not overwrite source view'
        lo=whole.reshape(-1,3).min(0);hi=whole.reshape(-1,3).max(0);mid=(lo+hi)/2;half=(hi-lo).max()*.54
        fig=plt.figure(figsize=(16,8))
        for vi,(elev,azim) in enumerate(((28,-60),(12,50))):
            for j,title in enumerate(('M: complete source','C: unchanged complete USER seat','Y: exact source complement','C + Y: exact union')):
                ax=fig.add_subplot(2,4,vi*4+j+1,projection='3d')
                items=[(whole,'#b3b8bf')] if j==0 else ([(c,'#189ac9')] if j==1 else ([(y,'#b3b8bf')] if j==2 else [(y,'#b3b8bf'),(c,'#189ac9')]))
                tri=np.concatenate([x[0] for x in items]);colors=[color for faces,color in items for _ in faces]
                ax.add_collection3d(Poly3DCollection(tri[:,:,[0,2,1]],facecolor=colors,edgecolor='#4d545f',linewidth=.16))
                ax.set_xlim(mid[0]-half,mid[0]+half);ax.set_ylim(mid[2]-half,mid[2]+half);ax.set_zlim(mid[1]-half,mid[1]+half)
                ax.set_box_aspect((1,1,1));ax.view_init(elev=elev,azim=azim);ax.set_axis_off();ax.set_title(title,fontsize=10)
        fig.suptitle(f"{r['parent_id'][:8]} | natural N={r['N']} K={r['K']} | {r['extent_description']} | source only",fontsize=12)
        fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.92,wspace=0,hspace=-.15);fig.savefig(dest,dpi=150,facecolor='white');plt.close(fig)
        rendered.append(dict(parent_id=r['parent_id'],image_path=str(dest),image_sha256=sha(dest),case_sha256=r['case_sha256']))
    write(DATA/'gallery_manifest.json',dict(rows=rendered,views=2,columns=['M','C','Y','C+Y'],model_calls=0))
    return dict(status='RENDERED_SOURCE_ONLY',pages=len(rendered),model_calls=0)


def finalize():
    review=read(DATA/'source_visual_review.json');candidates=read(DATA/'candidates.json');old=read(PRIOR)
    assert len(review['rows'])==4 and all(x['actually_viewed'] for x in review['rows'])
    accepted=[];blocked=[]
    for r in candidates['rows']:
        v=next(x for x in review['rows'] if x['parent_id']==r['parent_id'])
        assert sha(v['image_path'])==v['image_sha256']
        if v['decision']!='ACCEPT':blocked.append(dict(parent_id=r['parent_id'],reason=v['notes']));continue
        row=copy.deepcopy(r);row.update(full_seat_review='ACCEPT',review_status='PASS',training_ready=True,
            eligibility='ACCEPTED_NATIVE_NATURAL_N_COMPLETE_SEAT',source_semantic_review=v)
        accepted.append(row)
    assert len(accepted)>=2 and len({r['visual_family_id'] for r in accepted})>=2,'No science gate: new data insufficient'
    train=copy.deepcopy(old['train'])+accepted
    assert train[:26]==old['train'] and len({r['parent_id'] for r in train})==len(train)
    high=[r for r in train if r['replay_group']=='HIGH'];prereg=read(DATA/'candidate_preregistration.json')
    assert not {r['parent_id'] for r in train}&set(prereg['protected_eval_parents'])
    protected=dict(candidates['protected']);protect(protected,DATA/'source_visual_review.json');protect(protected,DATA/'gallery_manifest.json')
    for r in train+old['high_holdout']:D.verify_case(r);protect(protected,r['case_path'],r['case_sha256'])
    contract=dict(status='FROZEN',parent_ids=sorted(r['parent_id'] for r in high),
        new_parent_ids=sorted(r['parent_id'] for r in accepted),prior_parent_ids=sorted(r['parent_id'] for r in old['train'] if r['replay_group']=='HIGH'),
        sampling='balanced_shuffled_parent_cycles',natural_N=True,lower_N=257,upper_N=799,
        same_15_18_19_bucket_expansion='BLOCKED_NO_ADDITIONAL_TRUSTED_SOURCE',
        comparison='Broader natural structures, count buckets, and per-parent repetition jointly differ; not isolated bucket18 causality.')
    manifest=dict(schema=VERSION,status='READY_NATIVE_DATA',train=train,dev=copy.deepcopy(old['dev']),high_holdout=copy.deepcopy(old['high_holdout']),
        train_old22=train[:22],train_focus2=[r for r in train if r['replay_group']=='FOCUS'],
        train_new4=accepted,train_high=high,high_pool_contract=contract,
        counts=dict(train_parents=len(train),conditions=len(train),old=22,focus=2,high=len(high),new=len(accepted),high_holdout=2),
        N_contract=[128,799],train_N_range=[min(r['N'] for r in train),max(r['N'] for r in train)],
        source_manifest=str(PRIOR),source_manifest_sha256=sha(PRIOR),protected=protected,blocked_rows=blocked,
        old26_rows_exactly_preserved=True,no_new_noise_or_queries=True,model_input_policy=old['model_input_policy'],
        family_limit='Four independent source IDs and four recorded visual groups; bcb overlaps old ring-frame family, 945 near-variant alternatives all remain excluded. No genealogy or pretraining-unseen certification.',model_calls=0)
    # Avoid relying on historical order for the OLD grouping helper.
    manifest['train_old22']=[r for r in train if r['replay_group']=='OLD']
    write(DATA/'manifest.json',manifest)
    bucket_rows=[]
    for bucket in sorted({r['N']//20 for r in train}):
        sub=[r for r in train if r['N']//20==bucket]
        bucket_rows.append(dict(bucket=bucket,parents=len(sub),conditions=len(sub),parent_ids=[r['parent_id'] for r in sub],
            N=[r['N'] for r in sub],K=[r['K'] for r in sub],free_faces=[r['N']-r['K'] for r in sub],
            free_coordinates=[9*(r['N']-r['K']) for r in sub],actual_training_exposure='NOT_STARTED'))
    csvwrite(DATA/'count_bucket_inventory.csv',bucket_rows)
    checks=[D.verify_case(r) for r in train+old['high_holdout']]
    for p,h in protected.items():assert sha(p)==h
    outputs=(DATA/'manifest.json',DATA/'cpu_checks.json',DATA/'source_visual_review.json',DATA/'gallery_manifest.json',DATA/'count_bucket_inventory.csv',DATA/'candidate_preregistration.json')
    receipt=dict(status='PASS',schema=VERSION,train_parents=len(train),new_parents=len(accepted),high_parents=len(high),
        original26_exactly_preserved=True,checked_cases=len(checks),same_source_geometry_and_partition=True,
        independent_parent_ids=True,family_split_scope=manifest['family_limit'],protected_eval_overlap=[],
        registered_images=len(review['rows']),actually_viewed_images=sum(v['actually_viewed'] for v in review['rows']),
        source_array_mutations=0,new_finite_geometry_scores=0,new_noise_arrays=0,new_query_arrays=0,model_calls=0,
        files={str(p):sha(p) for p in outputs},protected_hashes=protected,checks=checks,preparation_code_sha256=sha(__file__))
    write(DATA/'cpu_gate.json',receipt);write(DATA/'COMPLETE.json',dict(status='COMPLETE',train_parents=len(train),high_parents=len(high),new_parents=len(accepted),cpu_gate_sha256=sha(DATA/'cpu_gate.json'),model_calls=0))
    return {k:receipt[k] for k in ('status','train_parents','new_parents','high_parents','checked_cases','actually_viewed_images','model_calls')}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('inventory','prepare','render','finalize'));a=p.parse_args()
    print(json.dumps(globals()[a.action](),ensure_ascii=False))
