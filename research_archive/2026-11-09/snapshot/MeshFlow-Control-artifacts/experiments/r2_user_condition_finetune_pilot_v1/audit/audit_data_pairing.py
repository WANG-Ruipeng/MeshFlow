"""Independent read-only audit of materialized paired pilot streams.
No imports of training/data modules, no RNG, OT, model or GPU calls.
"""
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np

ROOT=Path('/mnt/e/MeshFlow-Control-artifacts/experiments/r2_user_condition_finetune_pilot_v1')
SOURCE=Path('/mnt/e/MeshFlow-Control-artifacts/experiments/chair_r2_user_conditions_v1')
OUT=ROOT/'audit'
ARMS=('R2_CONT_OLD','R2_CONT_USER')

def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def ah(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256(str((a.dtype,a.shape)).encode());h.update(a.tobytes());return h.hexdigest()
def bits(a,b): return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def require(ok,reason):
    if not ok:raise AssertionError(reason)
def write_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temp,path)
def write_csv(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    paths=[ROOT/'data/paired_plan.json',ROOT/'data/input_registration.json',ROOT/'data/precompute_receipt.json',
        ROOT/'data/actual_exposures.json',SOURCE/'active_conditions.json',SOURCE/'quality/summary.json',
        SOURCE/'quality/condition_summary.json',SOURCE/'dataset_handoff_active.json']
    if (ROOT/'code_manifest.json').exists():paths.append(ROOT/'code_manifest.json')
    paths += [ROOT/'data'/a/'paired_inputs.json' for a in ARMS]
    before={str(p):sha(p) for p in paths}
    plan,reg,receipt,active,quality,condquality=[load(p) for p in (
        ROOT/'data/paired_plan.json',ROOT/'data/input_registration.json',ROOT/'data/precompute_receipt.json',
        SOURCE/'active_conditions.json',SOURCE/'quality/summary.json',SOURCE/'quality/condition_summary.json')]
    require(sha(ROOT/'data/paired_plan.json')==receipt['plan_sha256'],'plan receipt hash')
    require(sha(ROOT/'data/input_registration.json')==receipt['input_registration_sha256'],'input receipt hash')
    require(sha(SOURCE/'active_conditions.json')==reg['source_files'][str(SOURCE/'active_conditions.json')],'active manifest hash')
    rows=plan['records'];parents=reg['parents'];tasks=reg['tasks']
    indexes={a:load(ROOT/'data'/a/'paired_inputs.json')['records'] for a in ARMS}
    for a in ARMS:
        require(sha(ROOT/'data'/a/'paired_inputs.json')==receipt['arm_cache_index_sha256'][a],'cache index hash')
        require(len(rows[a])==len(indexes[a])==8000,'exact 8000 rows per arm')
        require([x['sample_index'] for x in rows[a]]==list(range(8000)),'sample order')
    field_mismatches=Counter();different_K=Counter();different_free_sets=0;paired_face_permutation_same=0
    held=set(active['held_condition_ids'])
    active_tasks={r['condition_id'] for r in active['rows'] if r['split']=='train'}
    dev_parents={r['uid'] for r in active['rows'] if r['split']=='dev'}
    require(not dev_parents&set(parents),'train/DEV disjoint')
    for old,new in zip(rows[ARMS[0]],rows[ARMS[1]]):
        for key in ('sample_index','effective_batch_index','slot_index','uid','N','alpha','t','t_seed','permutation_seed','clean','global_step','local_step'):
            if old[key]!=new[key]:field_mismatches[key]+=1
        for arm,row in ((ARMS[0],old),(ARMS[1],new)):
            task=tasks[arm][row['task_id']];parent=parents[row['uid']]
            require(task['uid']==row['uid'] and task['N']==row['N']==parent['N'] and task['K']==row['K'],'task/parent count identity')
            require(sorted(task['source_face_ids']+task['free_source_ids'])==list(range(row['N'])),'source exact partition')
            require(len(task['source_face_ids'])==row['K'] and len(task['free_source_ids'])==row['N']-row['K'],'K/free count')
            require(indexes[arm][row['sample_index']]['free_coordinate_count']==9*(row['N']-row['K']),'indexed free denominator')
            if arm==ARMS[1]:
                require(row['task_id'] in active_tasks and row['task_id'] not in held,'active/HOLD membership')
                require(task['parent_sha256']==parent['sha256'],'same source M hash')
                require(task['parent_npz']==str(Path('/mnt/e/MeshFlow-Control-artifacts/data')/parent['npz']),'same source M path')
        different_K['USER_less' if new['K']<old['K'] else 'USER_greater' if new['K']>old['K'] else 'equal']+=1
        different_free_sets+=set(tasks[ARMS[0]][old['task_id']]['free_source_ids'])!=set(tasks[ARMS[1]][new['task_id']]['free_source_ids'])
    require(not field_mismatches,'paired slot field mismatch')
    full={}
    for uid,p in parents.items():
        path=Path('/mnt/e/MeshFlow-Control-artifacts/data')/p['npz']
        require(sha(path)==p['sha256'],'source parent hash')
        with np.load(path,allow_pickle=False) as z:full[uid]=z['full_target'].reshape(p['N'],3,3).copy()
        require(full[uid].dtype==np.float32,'parent fp32')
        before[str(path)]=p['sha256']
    all_samplechecks=0;batch_bytes=0
    def check_batch(job):
        arm,batch=job;ii=indexes[arm][batch*8:(batch+1)*8];rr=rows[arm][batch*8:(batch+1)*8]
        path=Path(ii[0]['sample_npz']);cache_sha=sha(path)
        require(all(Path(q['sample_npz'])==path and q['sample_npz_sha256']==cache_sha for q in ii),'batch cache hash')
        out=[]
        with np.load(path,allow_pickle=False) as z:
            for slot,(index,row) in enumerate(zip(ii,rr)):
                prefix=f's{slot}__';md=json.loads(str(z[prefix+'metadata'].item()))
                task=tasks[arm][row['task_id']];n,k=row['N'],row['K']
                require(md['object_id']==row['uid'] and md['task_id']==row['task_id'] and md['alpha']==row['alpha'],'cached sample identity')
                require(md['K']==k and md['free_coordinate_count']==9*(n-k),'cached free denominator')
                require(int(z[prefix+'y'])==n and bits(z[prefix+'t'],np.asarray(row['t'],np.float32)),'cached N/t')
                ids,known,corners,permutation=[z[prefix+key] for key in ('source_face_ids','known_mask','corner_permutations','face_permutation')]
                require(known.dtype==np.bool_ and int(known.sum())==k,'known mask')
                require(np.array_equal(np.sort(ids[known]),np.sort(task['source_face_ids'])),'cached known source IDs')
                require(np.array_equal(np.sort(ids[~known]),np.sort(task['free_source_ids'])),'cached free source IDs')
                require(np.array_equal(np.sort(permutation),np.arange(n)),'face permutation bijection')
                require(np.array_equal(np.sort(corners,axis=1),np.tile(np.arange(3),(n,1))),'source corner permutation bijection')
                g,e,fm,cm=[z[prefix+key] for key in ('free_gaussian_before_OT','free_epsilon','ot_face_map','ot_corner_map')]
                require(g.shape==e.shape==(n-k,3,3),'free Gaussian/epsilon shape')
                require(ah(g)==index['gaussian_sha256'] and ah(e)==index['epsilon_sha256'],'actual Gaussian/epsilon hash')
                require(fm.dtype==np.int64 and np.array_equal(np.sort(fm),np.arange(n-k)),'OT face bijection')
                require(cm.dtype==np.int64 and cm.shape==(n-k,3) and np.array_equal(np.sort(cm,axis=1),np.tile(np.arange(3),(n-k,1))),'OT corner bijection')
                target=full[row['uid']].copy();target.reshape(-1,3)[:,0]*=np.float32(row['alpha'])
                expected=target[ids][np.arange(n)[:,None],corners].reshape(n,9)
                context,xt,x1=[z[prefix+key] for key in ('context','xt','x1')]
                require(bits(expected[known],context[known]) and bits(context[known],xt[known]),'exact source C preservation')
                require(bits(expected[~known],x1[~known]),'free mapped target matches augmented source complement')
                require(md['shared_input_sha256']==index['shared_input_sha256'] and md['input_bytehash']==index['input_bytehash'] and md['label_bytehash']==index['label_bytehash'],'cached input/label identities')
                out.append(ah(permutation))
        return len(ii),path.stat().st_size,out
    actual_permutations={a:[] for a in ARMS}
    jobs=[(a,b) for b in range(1000) for a in ARMS]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for job,result in zip(jobs,pool.map(check_batch,jobs)):
            count,size,perm=result;all_samplechecks+=count;batch_bytes+=size;actual_permutations[job[0]].extend(perm)
    paired_face_permutation_same=sum(a==b for a,b in zip(*(actual_permutations[a] for a in ARMS)))
    counts={a:Counter(r['task_id'] for r in rows[a]) for a in ARMS}
    require(set(counts[ARMS[1]])==active_tasks and not set(counts[ARMS[1]])&held,'all117 exposed and HOLD0')
    qrows={r['uid']:r for r in quality['rows']}
    selected_quality=[qrows[u] for u in sorted(parents)]
    require(sum(q['confirmed_illegal_pair_count']>0 for q in selected_quality)==14,'14 defective source parents retained')
    parent_table=[];condition_table=[];breakdown=[]
    for arm in ARMS:
        for uid in sorted(parents):
            pr=[r for r in rows[arm] if r['uid']==uid];cat=Counter(r['paired_old_category'] for r in pr)
            require(len(pr)==320 and all(cat[i]==80 for i in range(4)),'per-parent balanced categories')
            parent_table.append(dict(arm=arm,uid=uid,N=parents[uid]['N'],exposures=len(pr),clean=sum(r['clean'] for r in pr),width=sum(not r['clean'] for r in pr),C20_clean_or_paired_slot=cat[0],C40_clean_or_paired_slot=cat[1],C20_width_or_paired_slot=cat[2],C40_width_or_paired_slot=cat[3],free_coordinate_count=sum(9*(r['N']-r['K']) for r in pr),source_M_sha256=parents[uid]['sha256'],source_confirmed_pairs=qrows[uid]['confirmed_illegal_pair_count'],source_unknown_pairs=qrows[uid]['unknown_pair_count']))
        for task_id,task in sorted(tasks[arm].items()):
            tr=[r for r in rows[arm] if r['task_id']==task_id]
            condition_table.append(dict(arm=arm,condition_id=task_id,uid=task['uid'],region_type=task.get('region_type','NA'),tier=task.get('tier','NA'),tier_aliases='|'.join(task.get('requested_tiers',[])),N=task['N'],K=task['K'],ratio=task['ratio'],exposures=len(tr),clean=sum(r['clean'] for r in tr),width=sum(not r['clean'] for r in tr),free_coordinate_count=len(tr)*9*(task['N']-task['K'])))
        for dim in ('region_type','tier','K','ratio'):
            buckets=defaultdict(list)
            for row in rows[arm]:
                task=tasks[arm][row['task_id']]
                key=task.get(dim,'NA') if dim!='ratio' else (row['ratio'] if arm==ARMS[0] else 'NA_USER_VARIABLE_K_OVER_N')
                buckets[str(key)].append(row)
            for value,group in sorted(buckets.items()):
                breakdown.append(dict(arm=arm,dimension=dim,value=value,exposures=len(group),unique_conditions=len({r['task_id'] for r in group}),unique_parents=len({r['uid'] for r in group}),free_coordinate_count=sum(9*(r['N']-r['K']) for r in group)))
    write_csv('parent_exposures.csv',parent_table);write_csv('condition_exposures.csv',condition_table);write_csv('exposure_breakdowns.csv',breakdown)
    cq={r['condition_id']:r for r in condquality['rows']}
    quality_counts={}
    for part in ('CC','CF','FF'):
        quality_counts[part]=dict(conditions_with_confirmed=sum(cq[t]['counts'][part]['confirmed_illegal_pair_count']>0 for t in active_tasks),conditions_with_unknown=sum(cq[t]['counts'][part]['unknown_pair_count']>0 for t in active_tasks),registered_exposures_with_confirmed=sum(counts[ARMS[1]][t] for t in active_tasks if cq[t]['counts'][part]['confirmed_illegal_pair_count']>0),registered_exposures_with_unknown=sum(counts[ARMS[1]][t] for t in active_tasks if cq[t]['counts'][part]['unknown_pair_count']>0))
    totals={a:sum(9*(r['N']-r['K']) for r in rows[a]) for a in ARMS}
    require(totals=={ARMS[0]:10048320,ARMS[1]:13784553},'free totals')
    after={p:sha(p) for p in before};require(after==before,'read-only input hashes unchanged')
    result=dict(schema='r2_user_condition_finetune_data_pairing_audit_v1',status='PASS',
        audited_exposure_stage='FROZEN_MATERIALIZED_TRAIN_STREAM; this audit does not assert optimizer consumption/completion',
        paired_slots=8000,actual_cached_samples_checked=all_samplechecks,source_parent_count=25,
        pairing_fields=['uid','N','source_M_npz_and_sha256','alpha','t','t_seed','permutation_seed','batch_and_slot','local_and_global_step'],pairing_mismatches=dict(field_mismatches),
        per_arm_samples=8000,per_parent_exposures=320,per_parent_clean=160,per_parent_width=160,per_parent_old_or_paired_four_category_count=80,
        old_conditions_exposed=len(counts[ARMS[0]]),user_conditions_exposed=len(counts[ARMS[1]]),held_exposures=0,dev_training_exposures=0,
        paired_K_comparison=dict(different_K),paired_free_set_different=different_free_sets,
        common_permutation_seed_does_not_imply_same_arrays=True,actual_face_permutation_arrays_equal=paired_face_permutation_same,
        free_coordinate_totals=totals,free_coordinate_USER_over_OLD=totals[ARMS[1]]/totals[ARMS[0]],
        user_region_exposures=dict(Counter(tasks[ARMS[1]][r['task_id']]['region_type'] for r in rows[ARMS[1]])),
        user_tier_exposures=dict(Counter(tasks[ARMS[1]][r['task_id']]['tier'] for r in rows[ARMS[1]])),
        user_K1_exposures=sum(r['K']==1 for r in rows[ARMS[1]]),
        batch_cache_files=2000,batch_cache_bytes=batch_bytes,protected_input_files=len(before),inputs_before_after_equal=True,input_hashes=before,
        source_quality=dict(fixed_scale=quality['fixed_scale'],detector_policy=quality['detector_policy'],implementation_sha256=quality['implementation_sha256'],confirmed_parent_count=sum(q['confirmed_illegal_pair_count']>0 for q in selected_quality),unknown_parent_count=sum(q['unknown_pair_count']>0 for q in selected_quality),confirmed_pair_count=sum(q['confirmed_illegal_pair_count'] for q in selected_quality),unknown_pair_count=sum(q['unknown_pair_count'] for q in selected_quality),parent_rows=selected_quality,active_user_condition_partition=quality_counts),
        interpretation=['K/free and supervision scalar totals differ; this is not a semantics-only intervention.','Same source defects remain in M; neither source repair nor generated-score subtraction is performed.','UNKNOWN remains separate from confirmed and clean; repeated conditions/exposures do not create new independent defect pairs.','AI_PROXY regions: full refers to declared R, not necessarily all anatomical seat/back surfaces.','Tier aliases are deduplicated conditions; alias names do not multiply exposure.','Six DEV parents are disjoint from25 training parents but repeatedly observed official-training-split development objects, not independent held-out test.'],
        execution_counts=dict(model_loads=0,model_forwards=0,model_backwards=0,optimizer_updates=0,OT_calls=0,Gaussian_draws=0,new_geometry_tests=0,GPU_calls=0),workers=2,native_threads_per_worker=1,wall_seconds=time.perf_counter()-start)
    write_json(OUT/'data_pairing_audit.json',result)
    md=f'''逐槽审计两臂各8,000条冻结、已物化训练输入后，8,000组对应槽位的父UID、N、来源M路径与哈希、宽度增广alpha、时间t、t种子及排列种子全部一致；每批8个不同父，每父320次（原宽160、增广160），旧比例与宽度四类各80次。旧395条条件和新117条条件全部有曝光，HOLD与DEV训练曝光均为0。这里的曝光指已物化训练流的登记次数；实际完成的optimizer消费另由训练账本核对。

新臂按父均匀、再父内唯一C均匀选择，没有座面/靠背或档位配额。实际座面/靠背为{result['user_region_exposures']['seat']:,}/{result['user_region_exposures']['back']:,}次；档位分布为{json.dumps(result['user_tier_exposures'],ensure_ascii=False)}，K=1共{result['user_K1_exposures']:,}次。去重后的档位别名不会重复计数。逐父、逐条件及区域/档位/K分布分别见 `parent_exposures.csv`、`condition_exposures.csv`、`exposure_breakdowns.csv`。

全部16,000个缓存样本均核对了NPZ身份、known/free原面集合、Gaussian/epsilon尺寸与哈希、OT面/角双射、增广后原C保持和free目标回溯。两臂共同的排列种子不代表不同K下的排列数组相同。OLD的free坐标监督总量为10,048,320，USER为13,784,553（约多{(result['free_coordinate_USER_over_OLD']-1)*100:.2f}%）；因此后续差异同时涉及条件选择、K、面积及free监督量，不能单独归因为语义标签。

25个共同父仍保留14个已知缺陷父，来源M累计310对确认问题和3对UNKNOWN，未修复或自动删去。117个活动新条件中，CC确认/UNKNOWN条件数为0/3，CF为2/0，FF为64/10；这些是固定检测器的来源分区诊断，重复条件与曝光不能相加当作独立问题面对，也不能从生成结果的冲突分数中扣掉。3ede座面保留局部共面重叠风险；7f82靠背保留未决面对192/195，UNKNOWN没有记成无交。区域含义仍为AI_PROXY，“full”仅指登记R。六个新DEV虽与训练25父不重叠，仍是反复观察的开发对象，不是独立泛化测试。

本次审计仅CPU读取已冻结缓存和既有质量结果，0模型、0OT、0新Gaussian、0新几何检测；{len(before)}个审计输入前后哈希一致。审计结果为PASS，详细身份、计数、来源诊断与局限见 `data_pairing_audit.json`。
'''
    (OUT/'DATA_PAIRING_AUDIT.md').write_text(md,encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','paired_slots','actual_cached_samples_checked','paired_K_comparison','actual_face_permutation_arrays_equal','free_coordinate_totals','user_tier_exposures','user_K1_exposures','protected_input_files','wall_seconds']},indent=2))

if __name__=='__main__':main()