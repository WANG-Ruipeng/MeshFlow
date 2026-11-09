"""Independent array/split verification; save previously performed AI image review.

REVIEW_NOTES below are manually authored after the agent actually opened all ten
specified PNGs (two views of M|C|Y|C+Y). They are not inferred by this script.
"""
import collections
import hashlib
import json
import numpy as np
from . import layout_contract as A

ROOT=A.ROOT/'experiments/r2_bench_deployment_task_data_v1'
DATA=ROOT/'dataset/source_tasks_pending_adapter'
REVIEW_NOTES={
 '617fa8d6d65af35d71295682a9280b8b__R1':'C是有完整侧缘和厚度的长条坐垫，Y保留整张下承板及四腿；两视角组合位置吻合，输入用途清楚。',
 '617fa8d6d65af35d71295682a9280b8b__R2':'C是一根从板底延伸到地面的完整圆腿；虽然截面小但不是任意碎面，Y中其余三腿和座板明确保留。',
 '756108c9b72e95aa57d85583f4ad07cf__R1':'C覆盖完整长座面上表面，两端有来源构件交接的小轮廓转折；不是悬空散脸，放回后能明确定位座面。',
 '756108c9b72e95aa57d85583f4ad07cf__R2':'C是完整后框外侧表面，包含顶梁、竖条、下横梁和两端延长立柱；窗格留空有结构意义，不是缺失随机面。',
 '77aa33479c2ec92f610c2a68437007d6__R1':'C是整块带倒角厚度的座板，Y是两只完整造型墩座；部件边界清楚且组合恢复完整长凳。',
 '77aa33479c2ec92f610c2a68437007d6__R2':'C是整只弧形/收腰端部墩座，顶部承托面和底部均在，未见附带漂浮碎片；可作为用户输入完整支撑件。',
 '9452149917cb2e27f4fb4dee5181bee__R1':'C为等距平行座条的完整上表面组，长向范围整齐一致；多片分离来自真实条板间隙，不是随机裁剪。',
 '9452149917cb2e27f4fb4dee5181bee__R2':'C是从脚到扶手高度的完整端框，包含上下横杆和两根竖支撑，Y中的另一端框和座条清楚保留。',
 'abf250a01f526137a035c800c58595d9__R1':'C包含整组弧形包边座条，整体形成连续的弯曲座面轮廓，Y仅剩两墩座；条缝有规则且部位含义明确。',
 'abf250a01f526137a035c800c58595d9__R2':'C是完整的端部矩形支撑墩，顶部贴合座条底部；独立展示没有残缺切面或零散伴随片。',
}


def unordered_geometry(x):
    faces=[]
    for tri in x:
        faces.append(tuple(sorted(tuple(map(float,p)) for p in tri)))
    return hashlib.sha256(repr(sorted(faces)).encode()).hexdigest()


def main():
    manifest=A.read(DATA/'candidate_manifest.json');rows=manifest['rows'];assert len(rows)==30
    split_path=ROOT/'split_audit/new_data_split.json';split=A.read(split_path)
    contract_path=ROOT/'split_audit/staged_eval_contract.json';contract=A.read(contract_path)
    held=set(split['adaptation_unseen_parent_ids']);newtrain=set(split['new_training_parent_ids'])
    old=A.read(split['current_reused_training_manifest']);oldparents={r['parent_id'] for r in old['train']}
    assert not held & newtrain and not held & oldparents
    hashes={str(DATA/'candidate_manifest.json'):A.sha(DATA/'candidate_manifest.json'),
        str(split_path):A.sha(split_path),str(contract_path):A.sha(contract_path),
        str(A.resolve(split['current_reused_training_manifest'])):A.sha(A.resolve(split['current_reused_training_manifest'])),
        str(A.Path(__file__)):A.sha(__file__)}
    verification=[];geometries={};ids={};parent_source_identity={}
    allow={'C','C_vertices','C_faces_original','C_faces_model','C_source_vertex_ids','C_source_corner_permutations','N','K'}
    for r in rows:
        cid=r['condition_id'];parent=r['parent_id']
        assert r['parent_family_split_sha256']==A.sha(split_path)
        assert r['staged_eval_contract_sha256']==A.sha(contract_path)
        for key in ('input','paired','source_case','selection','quality','source_finite_receipt'):
            p=A.resolve(r[key+'_path']);hashes[str(p)]=A.sha(p);assert hashes[str(p)]==r[key+'_sha256']
        inp=A.arrays(r['input_path']);pair=A.arrays(r['paired_path']);src=A.arrays(r['source_case_path'])
        source_identity=(r['source_case_sha256'],str(src['transform_json'].item()))
        if parent in parent_source_identity:assert parent_source_identity[parent]==source_identity
        else:parent_source_identity[parent]=source_identity
        assert set(inp)==allow
        n,k=int(pair['N']),int(pair['K']);known=pair['known_ids'];free=pair['free_ids']
        assert n==len(src['source_faces'])==r['N'] and k==len(known)==r['K']
        assert len(np.unique(known))==k and len(np.unique(free))==n-k
        assert not set(known)&set(free) and np.array_equal(np.sort(np.r_[known,free]),np.arange(n))
        assert np.array_equal(known,r['source_face_ids'])
        assert np.array_equal(pair['C'],src['full_target'][known]) and np.array_equal(pair['Y'],src['full_target'][free])
        restored=np.empty_like(src['full_target']);restored[known]=pair['C'];restored[free]=pair['Y']
        assert np.array_equal(restored,src['full_target'])
        assert np.array_equal(pair['C_pre_ot'],src['pre_ot_full'][known]) and np.array_equal(pair['Y_pre_ot'],src['pre_ot_full'][free])
        assert np.array_equal(pair['C_vertices'],src['model_vertices'][pair['C_source_vertex_ids']])
        assert np.array_equal(pair['C_vertices'][pair['C_faces_original']],src['model_vertices'][src['source_faces'][known]])
        assert np.array_equal(pair['C_vertices'][pair['C_faces_model']],pair['C'])
        assert np.array_equal(np.take_along_axis(pair['C_faces_original'],pair['C_source_corner_permutations'],axis=1),pair['C_faces_model'])
        for key in allow:assert np.array_equal(inp[key],pair[key]),(cid,key)
        selected=A.read(r['selection_path']);sr=next(x for x in selected['rows'] if x['condition_id']==cid)
        assert sr['source_face_ids']==r['source_face_ids']
        assert sr['intended_eval_group']==r['intended_eval_group'] and sr['future_training_pool_membership']==r['future_training_pool_membership']
        if parent in held:
            assert r['intended_eval_group']=='ADAPTATION_UNSEEN_PARENT' and not r['future_training_pool_membership']
        else:
            assert parent in newtrain
            if r['recipe_id']=='R2':assert r['intended_eval_group']=='SAME_PARENT_HELDOUT_C' and not r['future_training_pool_membership']
            else:assert r['intended_eval_group']=='TRAIN_KNOWN_C' and r['future_training_pool_membership']
        assert not r['training_ready'] and not r['training_pool_membership'] and n>256
        geometries[cid]=unordered_geometry(inp['C']);ids[cid]=set(map(int,known))
        verification.append(dict(condition_id=cid,parent_id=parent,N=n,K=k,group=r['intended_eval_group'],
            exact_partition=True,C_same_source_transform=True,Y_same_source_transform=True,
            source_winding_reversible=True,input_only_whitelist=True,split_matches_frozen_selection=True,
            training_ready=False,model_budget_status=r['adapter_status']))
    future=[r for r in rows if r['future_training_pool_membership']]
    oldsignatures=set()
    for r in old['train']:
        p=A.resolve(r['input_path']);hashes[str(p)]=A.sha(p)
        oldsignatures.add(unordered_geometry(A.arrays(p)['C']))
    future_signatures=oldsignatures|{geometries[r['condition_id']] for r in future}
    reserve=[r for r in rows if r['intended_eval_group']=='SAME_PARENT_HELDOUT_C']
    for r in reserve:
        assert geometries[r['condition_id']] not in future_signatures
        for tr in future:
            if tr['parent_id']==r['parent_id']:
                assert not ids[r['condition_id']].issubset(ids[tr['condition_id']])
    dest=ROOT/'source_task_verification';dest.mkdir(exist_ok=True)
    A.write_csv(dest/'per_condition.csv',verification)
    result=dict(status='PASS_SOURCE_TASK_ARRAYS_SPLIT',conditions=30,parents=len({r['parent_id'] for r in rows}),
        group_counts=dict(collections.Counter(r['intended_eval_group'] for r in rows)),
        future_training_conditions=len(future),heldout_C=len(reserve),current_training_conditions_checked=len(old['train']),
        baseline_ready_conditions=0,source_targets_mutated=False,new_OT=0,new_model_forwards=0,
        no_current_or_staged_training_parent_overlap_with_new_holdout=True,
        heldout_C_order_invariant_geometry_not_equal_training=True,
        full_source_M_shared_for_same_parent_holdout=True,
        pretraining_exposure='UNKNOWN; all sources official train, not a pretraining-unseen claim',
        actual_training_schedule_exists=False)
    A.write_json(dest/'summary.json',result)
    gallery={r['condition_id']:r for r in A.read(DATA/'gallery_manifest.json')['rows']}
    review=[]
    for cid,note in REVIEW_NOTES.items():
        g=gallery[cid];p=A.resolve(g['image_path']);assert A.sha(p)==g['image_sha256']
        review.append(dict(condition_id=cid,decision='ACCEPT_SOURCE_TASK',observation=note,
            reviewer='AI_PROXY_layout_contract_agent',reviewer_type='AI_PROXY',actually_viewed=True,
            method='Opened this PNG directly with view_image; inspected both M/C/Y/C+Y rows before writing the observation. Nonblind semantic source-task review.',
            image_path=str(p),image_sha256=g['image_sha256'],
            baseline_ready=False,adapter_status='BLOCKED_N_OUTSIDE_128_256',
            decision_scope='Semantic coherence only; does not override independent geometry holds or model-budget blocker.'))
    A.write_json(DATA/'reviews/layout_agent.json',dict(status='COMPLETE_ACTUAL_IMAGE_REVIEW',reviewer='AI_PROXY',reviewer_identity='layout_contract_agent',rows=review,
        image_count=10,viewed_cells=80,training_ready=False,
        creation='Manual image observations recorded by script; no automatic semantic classification.'))
    for p,h in hashes.items():assert A.sha(p)==h
    A.write_json(dest/'COMPLETE.json',dict(status='COMPLETE',input_hashes=hashes,
        outputs={str(p):A.sha(p) for p in dest.iterdir() if p.is_file() and p.name!='COMPLETE.json'},
        review_path=str(DATA/'reviews/layout_agent.json'),review_sha256=A.sha(DATA/'reviews/layout_agent.json'),
        new_OT=0,new_model_forwards=0))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
