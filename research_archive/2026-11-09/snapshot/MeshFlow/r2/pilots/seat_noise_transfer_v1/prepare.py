"""CPU-only frozen inputs for familiar-seat noise transfer and held-out seats.

Reads source assets and previously frozen input metadata only. No generated
geometry, model weights, score functions, training, or geometry repair.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import csv, hashlib, json
from pathlib import Path
import numpy as np
from . import common as C

OUT=C.OUT/'inputs'
QUICK=C.ROOT/'experiments/r2_seat_quick_learnability_v1'
DIAG=C.ROOT/'experiments/r2_seat_failure_diagnosis_v1'
SCALING=C.ROOT/'experiments/r2_user_data_scaling_32_67_v1'
NAMESPACE='R2_SEAT_NOISE_TRANSFER_V1/NEW_EVAL_FULL_N_MT19937'
KNOWN_NEAR_DUPLICATE_DEV='64871dc28a21843ad504e40666187f4e'

def seed(rid):
    return int.from_bytes(hashlib.sha256((NAMESPACE+'/'+rid).encode()).digest()[:4],'little')

def prepare():
    dest=OUT/'manifest.json'
    if dest.exists():
        old=C.read(dest)
        for p,h in old['source_hashes'].items():
            if C.sha(p)!=h:raise ValueError('Protected source changed: '+p)
        for q in old['requests']:
            for p,h in ((q['input_path'],q['input_sha256']),(q['full_noise_path'],q['full_noise_file_sha256'])):
                if C.sha(p)!=h:raise ValueError('Frozen input changed: '+p)
        for q in old['queries']:
            if C.sha(q['path'])!=q['sha256']:raise ValueError('Query changed')
        return old
    protected={}
    def protect(path,expected=None):
        p=C.resolve(path);h=C.sha(p)
        if expected is not None and h!=expected:raise ValueError('Source identity mismatch: '+str(p))
        if str(p) in protected and protected[str(p)]!=h:raise ValueError('Source changed during preparation')
        protected[str(p)]=h;return h
    def read(path):
        protect(path);return C.read(path)
    quick=read(QUICK/'inputs/manifest.json')
    train=read(SCALING/'data/train_manifest.json')
    panel=read(SCALING/'data/panel_manifest.json')
    roles=read(SCALING/'data/parent_roles.json')
    oldreg=read(SCALING/'evaluation/registration.json')
    supplement=DIAG/'coverage_audit/supplement_v1/selected_pair_best_fixed_reflection.csv'
    protect(supplement)
    near=list(csv.DictReader(supplement.open(encoding='utf-8-sig')))
    near04=[r for r in near if r['pair_id']=='PAIR04' and r['domain']=='M']
    assert len(near04)==1 and near04[0]['dev_parent']==KNOWN_NEAR_DUPLICATE_DEV
    train_ids=set(train['pools']['USER_EXPANDED_67'])
    train_groups={train['parents'][u]['group_uid'] for u in train_ids}
    train_exact={train['parents'][u]['exact_geometry_sha256'] for u in train_ids}
    devparents={p['uid']:p for p in panel['dev_parents']}
    rolemap={r['uid']:r for r in roles['rows']}
    qmap={q['condition_id']:q for q in oldreg['queries'] if q['panel']=='NEW32'}
    # Freeze the rule before constructing/selecting new requests. This is a
    # source-eligibility screen, not a new geometric quality metric.
    rule=dict(schema='seat_noise_transfer_input_rule_v1',namespace=NAMESPACE,
        fixed_order='SEAT_FULL then SEAT_HALF; parent_id then condition_id lexicographic; first two eligible of each type; four distinct parents',
        eligibility='Existing NEW32 reviewed upper sitting surface; AI_PROXY region review, valid source mapping, N128..256, 0<K<N, one C component; inherited C/C and C/Y illegal=UNKNOWN=0; parent/group/exact geometry outside EXP67',
        known_near_duplicate_exclusion=near04,
        unknown_near_duplicates='Not exhaustively screened. Use existing group/exact hash and published PAIR04 near-duplicate warning; no new pairwise geometry search.',
        FF_policy='Inherited Y/Y problems disclosed, not used to favor a generated outcome or to certify a globally clean source',
        seed='SHA256(namespace + slash + request_id), first four bytes little endian; collision fails preparation without redraw',
        noise='One RandomState(seed).randn(1,N,9).astype(float32); save original full-N array; inference z copies it and zeroes known slots for old.read_input; free is the original [K:] slice',
        familiar_noise_count=8,heldout_noise_count=4,conditions=8,requests=48,
        source_edits=0,query_policy='Reuse exact existing per-C points, weights and complete query array; rebind only metadata identity to unchanged C __base variant',
        result_driven_selection=False,official_generator_pretraining_seen='UNKNOWN',official_D0_pretraining_seen='UNKNOWN')
    C.write(OUT/'selection_rule.json',rule)
    candidates=[];selected=[];used=set();counts=Counter()
    seats=sorted([t for t in panel['conditions'] if t['condition_type'] in ('SEAT_FULL','SEAT_HALF')],key=lambda t:(t['condition_type'],t['parent_id'],t['condition_id']))
    for t in seats:
        p=devparents[t['parent_id']];uid=t['parent_id'];reasons=[]
        if uid not in roles['DEV'] or rolemap[uid]['role']!='DEV':reasons.append('NOT_FROZEN_DEV')
        if uid in train_ids:reasons.append('TRAIN_PARENT')
        if p['group_uid'] in train_groups or t['group_uid'] in train_groups:reasons.append('TRAIN_GROUP')
        if p['exact_geometry_sha256'] in train_exact:reasons.append('TRAIN_EXACT_GEOMETRY')
        if uid==KNOWN_NEAR_DUPLICATE_DEV:reasons.append('KNOWN_PAIR04_CROSS_GROUP_NEAR_DUPLICATE')
        if t['review_kind']!='AI_PROXY' or t['region_declared']!='upper sitting surface':reasons.append('UNCONFIRMED_REGION')
        if t['C_components']!=1 or not 128<=t['N']<=256 or not 0<t['K']<t['N']:reasons.append('INPUT_CONTRACT')
        for part in ('CC','CF'):
            if t['CC_CF_FF'][part]!={'illegal':0,'unknown':0}:reasons.append('SOURCE_'+part+'_PROBLEM')
        if t['condition_id'] not in qmap:reasons.append('MISSING_FROZEN_QUERY')
        if t.get('conditional_pretrain_seen') is not False:reasons.append('CONDITIONAL_TRAIN_EXPOSURE_NOT_FALSE')
        eligible=not reasons
        if eligible and uid in used:reasons.append('PARENT_ALREADY_SELECTED_OTHER_TYPE')
        elif eligible and counts[t['condition_type']]>=2:reasons.append('TYPE_QUOTA_COMPLETE')
        elif eligible:
            selected.append(t);used.add(uid);counts[t['condition_type']]+=1
        candidates.append(dict(parent_id=uid,group_uid=p['group_uid'],condition_id=t['condition_id'],condition_type=t['condition_type'],N=t['N'],K=t['K'],
            eligible_before_quota=eligible,selected=t in selected,exclusion_reasons=reasons,source_CC_CF_FF=t['CC_CF_FF'],review_kind=t['review_kind'],
            annotation_source=t['annotation_source'],annotation_sha256=t['annotation_sha256'],official_pretrain_seen='UNKNOWN',near_duplicate_audit='KNOWN_RECORDS_ONLY_NOT_EXHAUSTIVE'))
    assert len(selected)==4 and counts=={'SEAT_FULL':2,'SEAT_HALF':2} and len(used)==4
    C.csvwrite(OUT/'dev_selection.csv',candidates)
    C.write(OUT/'dev_selection.json',dict(rule_sha256=C.sha(OUT/'selection_rule.json'),rows=candidates,selected=[t['condition_id'] for t in selected]))

    # Seed checks cover both original 6000-step plans (including preflight), the
    # two 200-step continuation plans, and available inherited eval manifests.
    old_seeds=set();seed_sources=[];old_array_hashes=set()
    def harvest(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if 'seed' in k.lower() and isinstance(v,int):old_seeds.add(v)
                elif k in ('Gaussian_sha256','free_Gaussian_sha256','full_Gaussian_sha256') and isinstance(v,str):old_array_hashes.add(v)
                elif isinstance(v,(list,dict)):harvest(v)
        elif isinstance(value,list):
            for v in value:harvest(v)
    seed_paths=set((SCALING/'plans').rglob('*.json'))|set((QUICK/'plans').glob('*.json'))
    seed_paths|={SCALING/'evaluation/registration.json',DIAG/'registration.json',QUICK/'inputs/manifest.json',QUICK/'evaluation/registration.json'}
    width=C.ROOT/'experiments/r2_condition_width_response_v1/registration.json'
    if width.exists():seed_paths.add(width)
    for p in sorted(seed_paths):
        z=read(p);harvest(z);seed_sources.append(dict(path=str(p),sha256=protected[str(p)]))
    C.package()
    from meshflow_control.source_winding import validate_source
    protect(C.ROOT/'r2/src/meshflow_control/source_winding.py')
    protect(C.ROOT/'r2/src/meshflow_control/winding_solver.py')
    for p in [C.ROOT/'AGENTS.md',C.ROOT/'WORKSPACE.md',C.ROOT/'native_t1/docs/control_metrics.md',C.ROOT/'r2/pilots/user_condition_finetune_v1/evaluation.py']:
        protect(p)
    old_input_reader=C.old()
    conditions=[];requests=[];queries=[];source_checks=[];new_seeds=set();new_hashes=set()
    for index,item in enumerate([*quick['train_conditions'],*selected]):
        familiar=index<4
        if familiar:
            t=train['tasks'][item['task_id']];p=train['parents'][t['uid']]
            meta=dict(next(x for x in quick['conditions'] if x['base_condition_id']==item['task_id']))
            original_query=next(x for x in quick['queries'] if x['condition_id']==meta['condition_id'])
        else:
            t=item;p=devparents[t['parent_id']]
            meta=dict(t);meta.update(condition_id=t['condition_id']+'__base',base_condition_id=t['condition_id'])
            original_query=qmap[t['condition_id']]
        uid=t['uid'];cid=meta['condition_id'];basecid=meta['base_condition_id'];N=int(t['N']);K=int(t['K'])
        panelname='TRAIN_FAMILIAR' if familiar else 'DEV_HELDOUT';split='TRAIN' if familiar else 'DEV'
        protect(t['npz'],t['sha256']);protect(p['npz'],p['sha256']);protect(t['quality_path'],t['quality_sha256'])
        protect(t['annotation_source'],t['annotation_sha256'])
        src=C.arrays(t['npz']);parent=C.arrays(p['npz']);cond=src['C'];source=validate_source(src,cond)
        assert int(src['N'])==N and int(src['K'])==K and source['has_real_source_vertex_ids']
        faces=parent['source_faces'];vertices=parent['model_vertices'];ids=src['source_face_ids'];freeids=src['free_source_ids']
        assert C.same(src['C_original'],vertices[faces[ids]])
        assert len(set(map(int,ids))&set(map(int,freeids)))==0 and sorted([*map(int,ids),*map(int,freeids)])==list(range(N))
        edgeinc=defaultdict(list)
        for fi,tri in enumerate(faces):
            for j in range(3):edgeinc[tuple(sorted((int(tri[j]),int(tri[(j+1)%3]))))].append(fi)
        cset=set(map(int,ids));interface=sum(len(fs)==2 and sum(f in cset for f in fs)==1 for fs in edgeinc.values())
        assert interface>0
        protect(original_query['path'],original_query['sha256']);qa=C.arrays(original_query['path'])
        assert original_query['identity']['C_sha256']==C.array_sha(cond) and original_query['query_count']==2048
        reconstructed=np.einsum('ni,nij->nj',qa['C_barycentric'],cond.astype(float)[qa['C_point_faces']])
        assert np.max(np.abs(reconstructed-qa['C_points']))<1e-6*C.L
        query=dict(original_query);query.update(condition_id=cid,base_condition_id=basecid,parent_id=uid,panel=panelname,split=split,
            original_query_condition_id=original_query['condition_id'],query_reused_byte_exact=True,
            identity={**original_query['identity'],'condition_id':cid})
        queries.append(query)
        meta.update(parent_id=uid,group_uid=t['group_uid'],panel=panelname,split=split,variant='base',alpha=1.,
            N=N,K=K,ratio=K/N,condition_type=t['condition_type'],region_type='SEAT',tier=t['condition_type'].split('_')[1],
            condition_source=t['npz'],condition_source_sha256=t['sha256'],C_sha256=C.array_sha(cond),
            pair_id='NOISE_TRANSFER_C'+str(index).zfill(2),selection_order=index,official_pretrain_seen='UNKNOWN',
            conditional_train_parent_seen=familiar,condition_training_seen=familiar,source_interface_edges=interface,
            source_CC_CF_FF=t['CC_CF_FF'],source_quality=t['source_quality'])
        conditions.append(meta)
        source_checks.append(dict(condition_id=cid,parent_id=uid,panel=panelname,N=N,K=K,source_mapping_verified=True,
            source_ID_interface_edges=interface,source_CY_geometry_status='INHERITED_NO_CC_CF_ILLEGAL_OR_UNKNOWN',
            query_array_sha256={k:C.array_sha(v) for k,v in qa.items()},C_sha256=C.array_sha(cond),source_C_sha256=C.array_sha(source['C_original']),
            coordinate_transform=json.loads(str(parent['transform_json'].item())) if 'transform_json' in parent else 'NA_MISSING_TRANSFORM_JSON',
            source_CC_CF_FF=t['CC_CF_FF'],annotation_source=t['annotation_source'],annotation_sha256=t['annotation_sha256'],
            region_declared=t['region_declared'],review_kind=t['review_kind']))
        for candidate in range(8 if familiar else 4):
            rid='SEATNOISE__'+panelname+'__'+cid+'__n'+str(candidate)
            s=seed(rid)
            if s in old_seeds or s in new_seeds:raise ValueError('Seed collision, no silent redraw: '+rid)
            new_seeds.add(s)
            full=np.random.RandomState(s).randn(1,N,9).astype(np.float32)
            z=full.copy();z[:,:K]=0;free=z[:,K:].reshape(-1,3,3)
            hs=[C.array_sha(x) for x in (full,z,free)]
            if any(h in old_array_hashes or h in new_hashes for h in hs):raise ValueError('Noise array identity collision')
            new_hashes.update(hs)
            noisep=OUT/'noise'/(rid+'.npz');C.frozen_npz(noisep,full_Gaussian=full)
            ip=OUT/'requests'/(rid+'.npz');C.frozen_npz(ip,C=cond,z=z,N=np.asarray(N,np.int64),K=np.asarray(K,np.int64))
            q=dict(meta,request_id=rid,base_request_id=rid,candidate=candidate,action='NEW',input_path=str(ip),input_sha256=C.sha(ip),
                noise_seed=s,noise_namespace=NAMESPACE,noise_provenance='FULL_N_MT19937_FP32_THEN_ZERO_KNOWN_NO_REDRAW',
                full_noise_path=str(noisep),full_noise_file_sha256=C.sha(noisep),full_Gaussian_sha256=hs[0],Gaussian_sha256=hs[1],free_Gaussian_sha256=hs[2],
                shared_across_model_labels=True)
            old_input_reader.read_input(q)
            assert C.same(z[:,K:],full[:,K:]) and not np.any(z[:,:K])
            requests.append(q)
    assert len(requests)==48 and len(conditions)==len(queries)==8
    assert len({q['parent_id'] for q in requests})==8 and len(new_seeds)==48
    for p,h in protected.items():
        if C.sha(p)!=h:raise ValueError('Protected old asset changed: '+p)
    C.write(OUT/'source_verification.json',dict(status='PASS',conditions=source_checks,source_geometry_edits=0,old_hashes_unchanged=True))
    C.write(OUT/'seed_audit.json',dict(status='PASS',namespace=NAMESPACE,sources=seed_sources,old_unique_seeds=len(old_seeds),
        new_unique_seeds=len(new_seeds),seed_collision_count=0,array_hash_collision_count=0,
        scope='Inherited 32/67 formal and preflight plans; quick continuation plans; available eval registration/input manifests. No claim about unavailable internal official-pretraining RNG.',
        new_seed_records=[dict(request_id=q['request_id'],seed=q['noise_seed'],full_Gaussian_sha256=q['full_Gaussian_sha256'],Gaussian_sha256=q['Gaussian_sha256'],free_Gaussian_sha256=q['free_Gaussian_sha256']) for q in requests]))
    result=dict(schema='r2_seat_noise_transfer_inputs_v1',status='FROZEN',created_utc=C.utc(),L=C.L,
        conditions=conditions,requests=requests,queries=queries,focused_task_ids=quick['focused_task_ids'],focused_parent_ids=quick['focused_parent_ids'],
        heldout_task_ids=[t['condition_id'] for t in selected],heldout_parent_ids=sorted(used),all_pool_parent_ids=sorted(train_ids),
        train_conditions=quick['train_conditions'],training_data_json=str(SCALING/'data/train_manifest.json'),
        training_data_sha256=protected[str(SCALING/'data/train_manifest.json')],source_hashes=protected,
        baseline_D0_identity=quick['baseline_D0_identity'],baseline_checkpoint_identity=quick['baseline_checkpoint_identity'],
        model_labels=['EXP67_FROZEN','SMALL_C200','FULL_C200'],selection_rule=rule,selection_rule_sha256=C.sha(OUT/'selection_rule.json'),
        definition='Familiar C: 4x8 new noises; held-out C:4x4 new noises, fixed three endpoints. Panels separate. Existing DEV was evaluated before; new to conditional training, not untouched research holdout.',
        model_calls=0,backwards=0,training_updates=0,new_noise_draws=48,new_query_draws=0,new_geometry_scoring_calls=0,
        code_sha256=C.sha(__file__),coordinate_policy='Inherited whole-parent model space. No separate C center/scale, retessellation, source edit, or hidden GT ranking.')
    C.csvwrite(OUT/'requests.csv',requests);C.csvwrite(OUT/'conditions.csv',conditions)
    lines=['# R2_SEAT_NOISE_TRANSFER_V1 数据准备','',
        '冻结8个条件、48份新噪声输入。熟悉4C各8噪声（32请求），DEV新父4C各4噪声（16请求）；三模型共用同一请求数组，共144次RAW及144次单次D0 POST。此准备实际模型调用、训练、几何评分均为0。','',
        '|面板|父ID|类型|K/N|噪声数|来源C/Y共边|来源FF illegal/unknown|',
        '|---|---|---|---|---|---|---|']
    for t in conditions:
        ff=t['source_CC_CF_FF']['FF'];lines.append(f"|{t['panel']}|{t['parent_id']}|{t['condition_type']}|{t['K']}/{t['N']}|{8 if t['split']=='TRAIN' else 4}|{t['source_interface_edges']}|{ff['illegal']}/{ff['unknown']}|")
    lines+=['','熟悉4C直接沿用上一快速学习实验，过去是有意纳入成功/失败的训练内诊断，不能当作无偏测试。DEV先固定FULL再HALF、父ID词典序、每类前2条合格且父不重复；只按已有来源资格挑选，没有读取生成数组或评分择父。DEV虽未进入67父条件训练，曾作为开发数据被评估，不称全新未触碰测试集。','',
        'DEV排除了原67父ID、group及相同源几何hash，另显式排除PAIR04的64871…：它与训练85b169…跨group近重复。已有记录未穷尽全部近重复，因此独立形状身份仅在这些已核对范围内成立。官方生成器及D0预训练是否见过均UNKNOWN。','',
        '所有座面标签继承已有AI_PROXY审阅的upper sitting surface；C/C及C/Y未检出illegal或unknown，源ID跨C/Y双面接口存在。FF源风险完整记录，不等于全局无交，也不要求生成复制唯一来源椅子。','',
        '固定namespace为 `'+NAMESPACE+'`。每request只抽一次全N标准Gaussian float32并单独保存；旧推理合同要求known槽为0，因此z仅把[:K]置零，free保持全N抽样中的[K:]。三模型不再抽噪声。完整数组、实际z、free切片及文件SHA全部保存。种子与可用原32/67训练（含preflight）、快速续训和旧评估注册核对无碰撞；不可用的官方预训练内部随机流保持UNKNOWN。','',
        '8条C的2048点查询全部复用已有完整数组，坐标与权重字节不变；仅将metadata condition_id绑定到相同C的__base身份。保存原查询身份与文件hash，没有新增查询采样、坐标归一化、修复或重三角化。','',
        '兼容合同：split=TRAIN/DEV，panel=TRAIN_FAMILIAR/DEV_HELDOUT，variant=base，alpha=1；request保留原评分META全部字段、来源角序与顶点ID映射。`old.read_input`已逐条核验48份输入。','',
        '恢复入口：`python -B -m r2.pilots.seat_noise_transfer_v1.prepare`。已冻结manifest存在时仅核对hash，不重新抽噪声或覆盖。选择/排除见dev_selection.csv，种子核对见seed_audit.json，来源映射与完整查询hash见source_verification.json。']
    C._write_bytes(OUT/'PREPARATION.md',('\n'.join(lines)+'\n').encode('utf-8'))
    C.write(dest,result)
    return result

if __name__=='__main__':
    m=prepare();print(json.dumps(dict(status=m['status'],manifest=str(OUT/'manifest.json'),conditions=len(m['conditions']),requests=len(m['requests']),heldout_task_ids=m['heldout_task_ids'],model_calls=0),ensure_ascii=False))
