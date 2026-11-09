"""Seal actually reviewed Bench v2 data. No models or training dispatch."""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import copy
import json
from .build import OUT, ROOT, read, save, sha, csvout


def seal_review_notes():
    """Attach identity/hash evidence to explicitly written human-readable reviews."""
    notes=read(OUT/'visual_review_notes.json');m=read(OUT/'candidate_manifest.json')
    result=copy.deepcopy(notes)
    result['candidate_manifest_sha256']=sha(OUT/'candidate_manifest.json')
    result['manual_notes_sha256']=sha(OUT/'visual_review_notes.json')
    def evidence(path):return dict(path=str(path),sha256=sha(path))
    for row in result['parents']:
        row['actually_viewed']=True
        row['image_evidence']=[evidence(OUT/'gallery/review_v1/parents'/(row['parent_id']+'.png'))]
    for row in result['conditions']:
        found=[r for r in m['conditions'] if r['parent_id'].startswith(row['parent_prefix']) and r['condition_id']==r['parent_id']+'__'+row['suffix']+'__V2']
        assert len(found)==1,row
        src=found[0];row.update(condition_id=src['condition_id'],parent_id=src['parent_id'],
            actually_viewed=True,C_standalone_plausible=row['decision']=='ACCEPT')
        row['image_evidence']=[evidence(OUT/'gallery/review_v1/overview'/(src['parent_id']+'__p01.png'))]
        if row['parent_prefix']=='3e4f4c' and row['suffix']=='SEAT_SURFACE_1':
            row['image_evidence'].append(evidence(OUT/'gallery/review_v1/conditions'/(src['condition_id']+'.png')))
    save(OUT/'visual_review.json',result)


def finalize():
    candidates=read(OUT/'candidate_manifest.json')
    review=read(OUT/'visual_review.json')
    assert review['reviewer_type']=='AI_PROXY'
    assert review['candidate_manifest_sha256']==sha(OUT/'candidate_manifest.json')
    p_review={r['parent_id']:r for r in review['parents']}
    c_review={r['condition_id']:r for r in review['conditions']}
    assert len(p_review)==len(review['parents']) and len(c_review)==len(review['conditions'])
    for r in review['parents']+review['conditions']:
        assert r['actually_viewed'] and r['image_evidence']
        for evidence in r['image_evidence']:
            assert sha(evidence['path'])==evidence['sha256']
    result=copy.deepcopy(candidates)
    parents={r['parent_id']:r for r in result['parents']}
    for uid,r in parents.items():
        vr=p_review.get(uid)
        r['review_status']=vr['decision'] if vr else 'NOT_REVIEWED_TARGET_BLOCKED'
        r['review']=vr
        r['eligibility']='ACCEPTED_TARGET' if vr and vr['decision']=='ACCEPT' and not r['reasons'] else 'HELD'
    active=[]
    for r in result['conditions']:
        parent=parents[r['parent_id']]; vr=c_review.get(r['condition_id'])
        r['review']=vr
        r['review_status']=vr['decision'] if vr else 'NOT_REVIEWED_TARGET_BLOCKED'
        reasons=[]
        if parent['eligibility']!='ACCEPTED_TARGET':reasons.append('PARENT_NOT_ACCEPTED')
        if not r.get('adapter_compatible',False):reasons.append('GEOMETRY_OR_ADAPTER_NOT_PASS')
        if not vr or vr['decision']!='ACCEPT':reasons.append('VISUAL_NOT_ACCEPTED')
        r['admission_reasons']=reasons
        r['eligibility']='ACCEPTED_PENDING_CPU_PACKING' if not reasons else 'HELD'
        r['source_group_id']=r['parent_id']
        r['near_duplicate_group_status']='NOT_AUDITED; all existing TRAIN, no new heldout claim'
        if not reasons:
            assert vr['C_standalone_plausible'] is True
            r['extent_description']=vr['observed_extent']
            active.append(r)
    assert active and len(active)<=128 and len(parents)<=32
    result.update(status='VISUALLY_ACCEPTED_PENDING_CPU_PACKING',train=active,
        visual_review_path=str(OUT/'visual_review.json'),visual_review_sha256=sha(OUT/'visual_review.json'),
        candidate_manifest_path=str(OUT/'candidate_manifest.json'),candidate_manifest_sha256=sha(OUT/'candidate_manifest.json'),
        training_registered=False,training_schedule=None,
        task_scope='FIRST_BOUNDED_DATA_BATCH; not a replacement full training pool and no performance claim')
    save(OUT/'manifest.json',result)
    csvout(OUT/'accepted_conditions.csv',active)
    csvout(OUT/'held_conditions.csv',[r for r in result['conditions'] if r['admission_reasons']])
    csvout(OUT/'parents.csv',list(parents.values()))
    groups=defaultdict(lambda:defaultdict(list))
    for r in active:groups[r['parent_id']][r['part_type']].append(r['condition_id'])
    policy=dict(schema='BENCH_USER_V2_PARENT_FIRST_POLICY',training_authorized=False,
        update_budget=None,training_schedule=None,noise_arrays_created=0,
        sampling_order=['source group','parent','accepted semantic part','condition','joint transform and noise'],
        per_parent_probability={uid:1/len(groups) for uid in sorted(groups)},
        parent_parts={uid:dict(parts) for uid,parts in sorted(groups.items())},
        policy='Equal parent exposure; rotate its available parts, then variants. No global BACK oversampling. Do not reuse old321 schedule.',
        source_group_limits='Parent IDs retained; no cross-source near-duplicate certificate. All remain historical TRAIN.',
        augmentation_contract='Transform complete M, C and owner/bulk targets together with the existing R2 transform. No independent C normalization.',
        readiness='CPU packing validation required; a later training task must freeze budget, schedule, group audit, evaluation and initialization')
    save(OUT/'sampling_policy.json',policy)
    summary=dict(candidate_parents=len(parents),candidate_conditions=len(result['conditions']),
        accepted_parents=len(groups),accepted_conditions=len(active),
        accepted_parts=dict(Counter(r['part_type'] for r in active)),
        part_parent_counts={p:len({r['parent_id'] for r in active if r['part_type']==p}) for p in sorted({r['part_type'] for r in active})},
        rejected_or_held_conditions=len(result['conditions'])-len(active),
        unchanged_DEV_conditions=len(result['dev']),unchanged_DEV_parents=len({r['parent_id'] for r in result['dev']}),
        model_calls=0,training_updates=0)
    save(OUT/'data_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def verify():
    m=read(OUT/'manifest.json')
    for p,h in m['protected_source_hashes'].items():assert sha(p)==h,p
    assert sha(m['candidate_manifest_path'])==m['candidate_manifest_sha256']
    assert sha(m['visual_review_path'])==m['visual_review_sha256']
    for p in m['parents']:
        for path,h in p['artifact_hashes'].items():assert sha(path)==h,path
    for r in m['conditions']:
        for key in ('case','input','paired'):
            if key+'_path' in r:assert sha(r[key+'_path'])==r[key+'_sha256']
    print('SEALED_DATA_HASHES_VERIFIED; NO_GEOMETRY_OT_OR_MODEL_CALLS')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify',action='store_true');p.add_argument('--seal-reviewed-notes',action='store_true');a=p.parse_args()
    if a.seal_reviewed_notes:seal_review_notes()
    elif a.verify:verify()
    else:finalize()
