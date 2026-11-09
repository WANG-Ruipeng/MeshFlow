"""CPU-only Bench provenance inventory and future evaluation split validator.

Reads existing manifests and official split metadata, never a model, generated
noise, or an entire mesh archive. Cached geometry hashes are evidence only at
their documented exact-coordinate precision, not a near-duplicate certificate.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_deployment_task_data_v1/split_audit'
ARCHIVE = ROOT / 'downloaded_data/archives/shapenet-cls.tar.gz'
CURRENT = ROOT / 'experiments/r2_bench_relation_training_v1/data/manifest.json'
COMMON = ROOT / 'experiments/r2_bench_parts_convergence_v1/data/manifest.json'
PANEL = ROOT / 'experiments/r2_bench_relation_training_v1/eval_manifest.json'
INVENTORY = ROOT / 'experiments/r2_bench_data_chain_audit_20261021/geometry/archive_supply/per_parent.csv'
HISTORY = [
    ROOT / ('experiments/' + x + '/data/manifest.json') for x in (
        'r2_bench_control_transfer_v1', 'r2_bench_parts_convergence_v1',
        'r2_bench_clean_multipart_seam_v1', 'r2_bench_coverage_expansion_v1',
        'r2_bench_user_v2_training_v1', 'r2_bench_relation_training_v1',
        'r2_bench_joint_shift_v1')]


def resolve(p):
    s = str(p).replace('\\', '/')
    if os.name == 'nt': s = s.replace('/mnt/e/MeshFlow', 'E:/MeshFlow')
    else: s = s.replace('E:/MeshFlow', '/mnt/e/MeshFlow')
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def sha(p):
    h = hashlib.sha256()
    with resolve(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def read(p): return json.loads(resolve(p).read_text(encoding='utf-8-sig'))


def save(p, value):
    p = resolve(p); p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n'
    if p.exists():
        assert read(p) == value, 'Immutable artifact differs: ' + str(p)
    else: p.write_text(text, encoding='utf-8')


def csv_save(p, rows):
    if not rows: return
    buf = io.StringIO(newline=''); w = csv.DictWriter(buf, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    p = resolve(p); p.parent.mkdir(parents=True, exist_ok=True)
    text = buf.getvalue()
    if p.exists(): assert p.read_text(encoding='utf-8') == text.replace('\r\n', '\n')
    else: p.write_text(text, encoding='utf-8', newline='')


def parent_set(rows): return {r['parent_id'] for r in rows}


def build():
    import numpy as np
    current, common, panel = read(CURRENT), read(COMMON), read(PANEL)
    historical = {}; protected = {}
    for p in HISTORY:
        assert p.exists(), str(p)
        m = read(p); rows = m.get('train', [r for r in m.get('rows', []) if r.get('group') == 'TRAIN'])
        historical[p.parent.parent.name] = parent_set(rows)
        protected[str(p)] = sha(p)
    tp, cp, dev = parent_set(current['train']), parent_set(common['train']), parent_set(current['dev'])
    all_hist = set.union(*historical.values())
    assert len(tp) == 26 and len(cp) == 64 and len(dev) == 8 and not all_hist & dev
    old_panel_train = parent_set([r for r in panel['requests'] if r['group'] == 'TRAIN'])
    source = list(csv.DictReader(INVENTORY.open(encoding='utf-8-sig')))
    protected[str(INVENTORY)] = sha(INVENTORY)
    archive_summary_path = INVENTORY.parent / 'summary.json'
    archive_summary = read(archive_summary_path)
    assert sha(ARCHIVE) == archive_summary['archive_sha256']
    protected[str(ARCHIVE)] = archive_summary['archive_sha256']
    protected[str(archive_summary_path)] = sha(archive_summary_path)
    protected[str(PANEL)] = sha(PANEL)
    split = {}; split_records = []
    with tarfile.open(ARCHIVE) as tar:
        for role in ('train', 'test'):
            member = 'shapenet-02828884/split/' + role + '.npz'
            data = tar.extractfile(member).read()
            with np.load(io.BytesIO(data), allow_pickle=True) as z:
                for record in z['npz_list']:
                    uid = str(record['uid']); assert uid not in split
                    split[uid] = role
            split_records.append(dict(member=member, sha256=hashlib.sha256(data).hexdigest(), parent_count=sum(v == role for v in split.values())))
    assert len(source) == len(split) == 560 and {r['parent_id'] for r in source} == set(split)
    # Reuse all cached hashes we can authenticate from compact parent manifests.
    hash_sources = [ROOT/'experiments/r2_bench_control_transfer_v1/data/manifest.json',
                    ROOT/'experiments/r2_bench_user_data_v2_expansion_v1/supply/manifest.json',
                    ROOT/'experiments/r2_bench_coverage_expansion_v1/candidates/manifest.json']
    gh = collections.defaultdict(set); hash_evidence = collections.defaultdict(list)
    def visit(obj, path):
        if isinstance(obj, dict):
            uid = obj.get('parent_id')
            digest = obj.get('exact_geometry_sha256', obj.get('exact_source_geometry_hash'))
            if uid and digest:
                gh[uid].add(digest); hash_evidence[uid].append(str(path))
            for v in obj.values(): visit(v, path)
        elif isinstance(obj, list):
            for v in obj: visit(v, path)
    for p in hash_sources:
        if p.exists():
            protected[str(p)] = sha(p); visit(read(p), p)
    assert all(len(v) == 1 for v in gh.values()), 'Cached exact geometry hashes disagree'
    groups = collections.defaultdict(list)
    for uid, hashes in gh.items(): groups[next(iter(hashes))].append(uid)
    rows = []
    for r in sorted(source, key=lambda r: r['parent_id']):
        uid = r['parent_id']; digest = next(iter(gh.get(uid, ())), '')
        duplicates = sorted(x for x in groups[digest] if x != uid) if digest else []
        history_members = sorted(k for k, v in historical.items() if uid in v)
        rows.append(dict(parent_id=uid, official_split=split[uid], source_N=int(r['N']),
            source_archive_member=r['archive_member'], source_archive_member_sha256=r['member_sha256'],
            in_COMMON64=uid in cp, in_current_RELATION26=uid in tp,
            in_known_adaptation_manifest_union=uid in all_hist,
            adaptation_manifest_names=';'.join(history_members),
            old_DEV_REGRESSION=uid in dev, old_TRAIN_PANEL_REGRESSION=uid in old_panel_train,
            candidate_adaptation_unseen_parent=(uid not in all_hist and uid not in dev and uid not in old_panel_train),
            candidate_adaptation_unseen_scope='Known recorded adaptation manifests; checkpoint lineage must be selected and verified later',
            official_pretraining_partition=('OFFICIAL_TRAIN_PARTITION' if split[uid] == 'train' else 'OFFICIAL_TEST_PARTITION'),
            actual_official_checkpoint_parent_exposure='UNKNOWN_NO_PER_PARENT_OFFICIAL_TRAINING_LOG',
            cached_exact_geometry_sha256=digest or 'UNKNOWN',
            cached_exact_geometry_same_other_UIDs=';'.join(duplicates),
            exact_duplicate_with_known_adaptation=any(x in all_hist for x in duplicates) if digest else 'UNKNOWN',
            exact_duplicate_with_old_DEV=any(x in dev for x in duplicates) if digest else 'UNKNOWN',
            near_duplicate_family_status='UNKNOWN',
            source_two_exact_checks=r['passes_two_exact_checks'],
            training_or_evaluation_admission='NOT_ASSIGNED_REQUIRES_TARGET_CONDITION_REVIEW'))
    csv_save(OUT/'parent_inventory560.csv', rows)
    csv_save(OUT/'current_parent_membership.csv', [r for r in rows if r['in_current_RELATION26'] or r['old_DEV_REGRESSION'] or r['old_TRAIN_PANEL_REGRESSION']])
    save(OUT/'official_split_evidence.json', dict(archive_path=str(ARCHIVE), archive_sha256=protected[str(ARCHIVE)], members=split_records,
        source_code='datasets/mesh_dataset.py:123-130 chooses train.npz or test.npz',
        source_config='configs/snet/base-120m-ot-v-bench.yaml',
        actual_checkpoint_exposure='UNKNOWN; official split membership is evidence of intended partition, not a per-checkpoint exposure receipt'))
    for p in (ROOT/'datasets/mesh_dataset.py', ROOT/'configs/snet/base-120m-ot-v-bench.yaml'):
        protected[str(p)] = sha(p)
    summary = dict(status='INVENTORY_COMPLETE_NO_SPLIT_FROZEN', current_parents=len(tp), current_conditions=len(current['train']),
        COMMON_parents=len(cp), current_COMMON_overlap=len(tp&cp), current_not_COMMON=len(tp-cp),
        historical_panel_train_parents=len(old_panel_train), historical_panel_train_in_current=len(old_panel_train&tp),
        legacy_DEV_parents=len(dev), legacy_DEV_conditions=len(current['dev']),
        legacy_DEV_official_split=dict(collections.Counter(split[x] for x in dev)),
        official_split_counts=dict(collections.Counter(split.values())), known_historical_adaptation_union=len(all_hist),
        adaptation_history_parent_counts={k:len(v) for k,v in historical.items()},
        candidate_UIDs_unseen_known_adaptation=sum(r['candidate_adaptation_unseen_parent'] for r in rows),
        candidate_UIDs_unseen_by_official_split=dict(collections.Counter(r['official_split'] for r in rows if r['candidate_adaptation_unseen_parent'])),
        cached_exact_geometry_hash_UIDs=sum(bool(v) for v in gh.values()), cached_cross_UID_duplicate_groups=[sorted(v) for v in groups.values() if len(v)>1],
        new_parent_assignments=0, new_condition_assignments=0, new_noise_arrays=0, GPU_calls=0, model_calls=0, training_updates=0,
        important_boundary='Unseen UID is not certified independent shape family; new data quality/semantics are not inferred from inventory.',
        exact_hash_scope='Existing normalized unordered-triangle geometry hashes, reused only. Uncached UIDs remain UNKNOWN; no full mesh reread or near-duplicate inference.')
    save(OUT/'summary.json', summary)
    save(OUT/'protected_input_hashes.json', protected)
    save(OUT/'evaluation_contract.json', evaluation_contract())
    save(OUT/'manifest_template.json', dict(schema='BENCH_DEPLOYMENT_TASK_SPLIT_V1', status='DRAFT_NOT_FROZEN',
        initialization_checkpoint=None, initialization_adaptation_parent_ids=[], training_conditions=[],
        evaluation=dict(TRAIN_KNOWN_C=[], SAME_PARENT_HELDOUT_C=[], ADAPTATION_UNSEEN_PARENT=[], REGRESSION=[]),
        noise_registration=None, notes='No request counts, noise identities, checkpoint or training budget frozen by this inventory.'))
    text = f'''# Bench 父级来源与三组评价准备\n\n状态：只读盘点完成；没有冻结新的训练/评价数量，没有生成噪声或调用模型。\n\n- 当前 RELATION/SHIFT 为26父91条C；其中7父属于 COMMON12000的64父，19父为该次后续适配新增。\n- 旧48请求中的历史TRAIN4父与当前26父交集为0。旧面板不能说明本轮真实训练条件上的拟合。\n- 旧DEV8父39条件保持 REGRESSION；其中7父官方test、1父官方train。它们是反复使用的开发证据，不重新命名独立测试集。\n- 官方档案split为504 train /56 test；本轮只读两个split数组并核验档案hash。实际官方权重逐父曝光日志不可得，记UNKNOWN，不将test成员直接宣称为对所有预训练完全未见。\n- 已有实际适配清单父并集为{len(all_hist)}；剩余{summary['candidate_UIDs_unseen_known_adaptation']}个UID可列入“未出现在已核对适配清单”的候选，尚未通过目标几何、人类输入语义或近重复审核，不是已就绪数据。\n- 复用{len(gh)}父缓存精确几何hash。未缓存父的几何重复状态UNKNOWN；不同UID不等于独立款式。\n\n后续使用 evaluation_contract.json 与 manifest_template.json：先定父级/精确重复组边界，再造C；真实训练C、同父预留C、适配未见父分别报告。预留C不能只是换ID或角序的训练C；新父需同时避开所选初始化的适配祖先。旧DEV另列REGRESSION。来源完整目标只用于监督和检查，不能作唯一生成成功答案。\n\n冻结前运行 `python -B -m r2.data_tools.bench_deployment_tasks_v1.split_audit validate <manifest.json>`。该验证器只核对合同与身份，不能代替完整目标/条件图审。\n'''
    p=OUT/'REPORT.md'
    if p.exists(): assert p.read_text(encoding='utf-8')==text
    else: p.write_text(text,encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


def evaluation_contract():
    return dict(schema='BENCH_EVALUATION_THREE_GROUP_CONTRACT_V1', status='PREPARATION_ONLY',
        groups={
            'TRAIN_KNOWN_C':'Current training parent and exact trained C; fixed generated noise registered separately.',
            'SAME_PARENT_HELDOUT_C':'Same current training parent, reserved source-face region and C geometry; not an independent-parent generalization claim.',
            'ADAPTATION_UNSEEN_PARENT':'Parent absent from current training and every adaptation ancestor of chosen initialization. Exact geometry duplicate groups held out together; near-duplicate scope disclosed.',
            'REGRESSION':'Previously used DEV and historical TRAIN panels remain separately identified; never relabel as independent test.'},
        required_condition_fields=['condition_id','parent_id','source_group_id','target_version_id','target_case_path','target_case_sha256','input_path','input_sha256','C_geometry_sha256','C_topology_sha256','source_face_ids','N','K','part_type','extent_description','official_split','actual_pretraining_exposure','source_geometry_hash','near_duplicate_family_status','deployment_input_contract_version','deployment_layout_sha256','target_representability_status','review_receipt_path','review_receipt_sha256'],
        required_before_training=['initialization_checkpoint and hash','complete initialization adaptation ancestry','no parent/exact-geometry split leakage','parent/condition split frozen before model results','noise files and hashes registered without worker-dependent RNG','RAW/BASE/FINAL contracts frozen','C adoption + body + support judged on the same output; CF/FF/I/fragments/deletion costs retained'],
        independent_group_not_unique_GT_reconstruction=True, no_training_or_noise_budget_implied=True,
        source_target_usage='Supervision and task validity only; hidden completion is not an inference input.',
        unresolved_states=['UNKNOWN','NA','BLOCKED'])


def validate(path):
    m=read(path); errors=[]; train=m.get('training_conditions',[]); ev=m.get('evaluation',{})
    required=evaluation_contract()['required_condition_fields']; tp=parent_set(train)
    tc={r['condition_id'] for r in train}; cg={r.get('C_geometry_sha256') for r in train}; sg={r.get('source_geometry_hash') for r in train}
    ancestry=set(m.get('initialization_adaptation_parent_ids', []))
    ancestors_hash=set(m.get('initialization_adaptation_geometry_hashes', []))
    historical_dev=parent_set(read(CURRENT)['dev'])
    historical_panel=parent_set(read(PANEL)['requests'])
    if m.get('status')!='FROZEN': errors.append('MANIFEST_NOT_FROZEN')
    if not m.get('initialization_checkpoint'): errors.append('INITIALIZATION_NOT_REGISTERED')
    if m.get('initialization_ancestry_status')!='VERIFIED': errors.append('INITIALIZATION_ANCESTRY_NOT_VERIFIED')
    if not train: errors.append('NO_TRAINING_CONDITIONS')
    if not m.get('noise_registration'): errors.append('NO_NOISE_REGISTRATION')
    for group in evaluation_contract()['groups']:
        if group not in ev: errors.append('MISSING_GROUP:'+group)
    seen=set()
    for group,rows in [('TRAINING',train)]+list(ev.items()):
        for r in rows:
            cid=r.get('condition_id','UNKNOWN'); pid=r.get('parent_id'); prefix=group+':'+cid+':'
            for key in required:
                if key not in r: errors.append(prefix+'MISSING_'+key)
            if group=='TRAINING' and pid in historical_dev: errors.append(prefix+'OLD_DEV_MUST_REMAIN_REGRESSION')
            for key in ('input','target_case','review_receipt'):
                p=r.get(key+'_path'); digest=r.get(key+'_sha256')
                if p and digest:
                    if not resolve(p).is_file(): errors.append(prefix+'MISSING_FILE_'+key)
                    elif sha(p)!=digest: errors.append(prefix+'HASH_MISMATCH_'+key)
            if r.get('K') is not None and r.get('N') is not None:
                if not 2<=r['K']<r['N']: errors.append(prefix+'INVALID_K_N')
            if 'source_face_ids' in r and len(set(r['source_face_ids']))!=len(r['source_face_ids']): errors.append(prefix+'DUPLICATE_C_FACE_ID')
            if group=='TRAIN_KNOWN_C' and (pid not in tp or cid not in tc): errors.append(prefix+'NOT_TRAINED_C')
            if group=='SAME_PARENT_HELDOUT_C' and (pid not in tp or cid in tc or r.get('C_geometry_sha256') in cg): errors.append(prefix+'HELDOUT_C_IDENTITY_OVERLAP')
            if group=='ADAPTATION_UNSEEN_PARENT':
                if pid in tp|ancestry: errors.append(prefix+'ADAPTATION_PARENT_EXPOSED')
                if pid in historical_panel: errors.append(prefix+'OLD_PANEL_MUST_REMAIN_REGRESSION')
                if r.get('source_geometry_hash') in sg|ancestors_hash: errors.append(prefix+'EXACT_SOURCE_DUPLICATE_WITH_TRAIN_OR_ANCESTOR')
                if r.get('source_geometry_hash') in (None,'','UNKNOWN'): errors.append(prefix+'EXACT_SOURCE_HASH_REQUIRED')
            if r.get('target_representability_status')!='PASS': errors.append(prefix+'TARGET_REPRESENTABILITY_NOT_PASS')
            if group!='TRAINING':
                rid=r.get('request_id',cid)
                if (group,rid) in seen: errors.append(prefix+'DUPLICATE_REQUEST')
                seen.add((group,rid))
    result=dict(status='PASS' if not errors else 'BLOCKED', manifest_path=str(resolve(path)),manifest_sha256=sha(path),errors=errors,model_calls=0,new_noise=0)
    print(json.dumps(result,ensure_ascii=False,indent=2)); return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build','verify','validate']);parser.add_argument('manifest',nargs='?');args=parser.parse_args()
    if args.action=='build': build()
    elif args.action=='verify':
        p=read(OUT/'protected_input_hashes.json')
        for name,digest in p.items(): assert sha(name)==digest,name
        rows=list(csv.DictReader((OUT/'parent_inventory560.csv').open(encoding='utf-8')))
        summary=read(OUT/'summary.json')
        assert len(rows)==560
        assert sum(r['cached_exact_geometry_sha256']!='UNKNOWN' for r in rows)==summary['cached_exact_geometry_hash_UIDs']==170
        assert sum(r['candidate_adaptation_unseen_parent']=='True' for r in rows)==summary['candidate_UIDs_unseen_known_adaptation']==456
        result=dict(status='PASS',protected_files=len(p),parent_rows=560,cached_hash_rows=170,unassigned_adaptation_unseen_UIDs=456,
                    source_hashes_unchanged=True,models=0,GPU_calls=0,training_updates=0,new_noise_arrays=0,
                    code_sha256=sha(Path(__file__)),summary_sha256=sha(OUT/'summary.json'),
                    inventory_sha256=sha(OUT/'parent_inventory560.csv'))
        save(OUT/'verification.json',result)
        print(json.dumps(result))
    else:
        if not args.manifest: parser.error('validate requires manifest')
        result=validate(args.manifest)
        raise SystemExit(0 if result['status']=='PASS' else 2)
