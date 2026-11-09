"""Publish a future-only selection from existing audits; never edit source assets.

CPU/stdlib only. Default publishes once; --verify checks the existing publication.
The output is an eligible candidate pool, not a registered training experiment.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / 'future_pool_v1'
POINTER = ROOT / 'r2/data_tools/bench_future_pool.json'
SOURCE = ROOT / 'experiments/r2_bench_coverage_expansion_v1/data/manifest.json'
PROPOSAL = HERE / 'risk_proposal.json'
PASS = 'ELIGIBLE_TWO_EXACT_CHECKS_ONLY'
HOLD = 'QUARANTINE_REBUILD_BEFORE_REUSE'
UNKNOWN = 'PENDING_TARGET_AUDIT'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path):
    return path.relative_to(ROOT).as_posix()


def write(path, obj):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def resolve_case(raw):
    raw = raw.replace('\\', '/')
    for prefix in ('/mnt/e/MeshFlow/', 'E:/MeshFlow/'):
        if raw.startswith(prefix):
            p = (ROOT / raw[len(prefix):]).resolve()
            assert p.is_relative_to(ROOT), raw
            return p
    raise ValueError(f'Unexpected source path: {raw}')


def prepare():
    p, m = read(PROPOSAL), read(SOURCE)
    protected = dict(p['source_hashes'])
    protected[rel(PROPOSAL)] = sha(PROPOSAL)
    for path, expected in protected.items():
        assert sha(ROOT / path) == expected, path
    rows = m['train'] + m['dev']
    assert len(m['train']) == 321 and len(m['dev']) == 39
    assert len({r['condition_id'] for r in rows}) == len(rows)
    assert {r['condition_id']: r for r in rows} == {r['condition_id']: r for r in m['rows']}
    assert not ({r['parent_id'] for r in m['train']} & {r['parent_id'] for r in m['dev']})
    for row in rows:
        path = resolve_case(row['case_path'])
        assert sha(path) == row['case_sha256'], row['condition_id']
        protected[rel(path)] = row['case_sha256']
    risks = {r['condition_id']: r for r in p['conditions']}
    assert set(risks) == {r['condition_id'] for r in m['train']}
    selection = []
    for row in m['train']:
        risk = risks[row['condition_id']]
        assert (risk['parent_id'], risk['case_sha256']) == (row['parent_id'], row['case_sha256'])
        overlap, zero = risk['C_Y_exact_geometry_overlap_pairs'], risk['Y_exact_zero_area_model64_faces']
        if overlap is None or zero is None:
            decision = UNKNOWN
            assert overlap is None and zero is None
        elif overlap > 0 or zero > 0:
            decision = HOLD
        else:
            assert overlap == zero == 0
            decision = PASS
        selection.append(dict(risk, future_pool_status=decision,
                              included_in_future_candidate_train=decision == PASS))
    assert Counter(r['future_pool_status'] for r in selection) == {PASS: 114, HOLD: 163, UNKNOWN: 44}
    return p, m, selection, protected


def stats(rows):
    return dict(conditions=len(rows), parents=len({r['parent_id'] for r in rows}),
                by_part=dict(sorted(Counter(r['part_type'] for r in rows).items())),
                known_Y_internal_duplicate_conditions=sum((r['Y_source_ID_excess_faces'] or 0) > 0 for r in rows),
                known_Y_internal_duplicate_parents=len({r['parent_id'] for r in rows if (r['Y_source_ID_excess_faces'] or 0) > 0}))


def verify():
    _, source, selection, protected = prepare()
    receipt = read(OUT / 'execution_receipt.json')
    assert receipt['protected_source_hashes'] == protected
    for name, expected in receipt['published_hashes'].items():
        assert sha(ROOT / name) == expected, name
    pointer = read(POINTER)
    assert pointer['manifest_sha256'] == sha(OUT / 'manifest.json')
    manifest = read(OUT / 'manifest.json')
    selected = {r['condition_id'] for r in selection if r['future_pool_status'] == PASS}
    assert manifest['train'] == [r for r in source['train'] if r['condition_id'] in selected]
    assert manifest['dev'] == source['dev']
    seen = set(selected)
    for filename, status in [('quarantine_manifest.json', HOLD), ('pending_audit_manifest.json', UNKNOWN)]:
        held = read(OUT / filename)
        ids = {r['condition_id'] for r in selection if r['future_pool_status'] == status}
        assert held['rows'] == [r for r in source['train'] if r['condition_id'] in ids]
        assert not seen & ids
        seen |= ids
    assert seen == {r['condition_id'] for r in source['train']}
    assert manifest['schedule'] is None and manifest['training_registered'] is False
    print(json.dumps(dict(status='PASS', groups=manifest['summary'],
                          original_case_files_verified=360, protected_files_verified=len(protected),
                          new_model_calls=0, new_geometry_score_calls=0, deleted_files=0), ensure_ascii=False))


def publish():
    assert not OUT.exists(), 'Publication exists; use --verify. Never overwrite an existing selection.'
    assert not POINTER.exists(), 'Existing future pool pointer needs an explicit version transition.'
    p, m, selection, protected = prepare()
    groups = {s: [r for r in selection if r['future_pool_status'] == s] for s in (PASS, HOLD, UNKNOWN)}
    ids = {s: {r['condition_id'] for r in rs} for s, rs in groups.items()}
    summary = {s: stats(rs) for s, rs in groups.items()}
    stamp = datetime.now(timezone.utc).isoformat()
    policies = [
        'Future Bench candidate-pool selection only; historical frozen manifests/results remain unchanged.',
        'No original file, face, C, Y, N, source identity, label or review is deleted or repaired.',
        'Quarantined conditions cannot enter a new training pool until rebuilt and re-audited in a new version.',
        'Missing matching exact-target audit means pending, not zero defects and not demonstrated invalidity.',
        'Eligible means only zero whole-face exact C/Y overlap and zero exactly zero-area valid Y faces in the cited audit.',
        'Y-only opposite-winding duplicates remain separately flagged; no full selfintersection or visual-clean certification.',
        'No filtering by generated good/bad outcomes, model score, or DEV results. DEV rows preserved verbatim.',
        'New human-input construction must favor complete coherent recognizable parts (prefer K>=2); this filter is not a fresh semantic review.',
        'No old exposure schedule is inherited; a future authorized training requires a new frozen schedule and registration.',
        'Original row fields are historical metadata; outer selection status governs future eligibility.',
    ]
    train = [r for r in m['train'] if r['condition_id'] in ids[PASS]]
    manifest = dict(schema='BENCH_FUTURE_CANDIDATE_POOL_EXACT_GATES_V1',
                    status='ELIGIBLE_CANDIDATES_NOT_CERTIFIED_CLEAN', created_utc=stamp,
                    train=train, dev=m['dev'],
                    rows=[r for r in m['rows'] if r['group'] == 'DEV' or r['condition_id'] in ids[PASS]],
                    summary=summary, source_manifest=rel(SOURCE), source_manifest_sha256=sha(SOURCE),
                    audit_proposal=rel(PROPOSAL), audit_proposal_sha256=sha(PROPOSAL),
                    schedule=None, training_registered=False, source_labels_for_training_only=True,
                    part_labels_not_model_inputs=True, data_policy=policies)
    OUT.mkdir()
    write(OUT / 'manifest.json', manifest)
    for filename, status in [('quarantine_manifest.json', HOLD), ('pending_audit_manifest.json', UNKNOWN)]:
        write(OUT / filename, dict(schema='BENCH_FUTURE_POOL_HOLD_V1', status=status,
                                  rows=[r for r in m['train'] if r['condition_id'] in ids[status]],
                                  decisions=groups[status], summary=summary[status], source_manifest_sha256=sha(SOURCE)))
    write(OUT / 'already_excluded.json', p['already_excluded'])
    write(OUT / 'summary.json', dict(groups=summary, dev_conditions=39, dev_parents=8,
                                    exclusions_are_per_condition_not_whole_parent=True,
                                    previous_exclusions=p['already_excluded_summary'], policies=policies))
    with (OUT / 'selection.csv').open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(selection[0]))
        w.writeheader()
        w.writerows(selection)
    kept = summary[PASS]
    report = f'''# Bench 后续候选池筛选（2026-10-21）

已按用户选择执行“从后续训练池剔除不合适样本，保留原件和实验记录”。只建立独立清单；没有物理删除、目标修复、模型调用或训练。

| 后续用途 | 条件数 | 涉及父对象数 |
|---|---:|---:|
| 通过两项精确检查，保留为候选 | 114 | {kept['parents']} |
| 已知 C/Y 整面重合或有效 Y 精确零面积，暂时隔离 | 163 | {summary[HOLD]['parents']} |
| 缺少相同口径目标审计，待核验 | 44 | {summary[UNKNOWN]['parents']} |

三组条件互斥且覆盖原321条；父对象数不可相加，同一父可有不同状态的条件。163条中79条有C/Y整面重合、136条有有效Y零面积，两者重叠52条。隔离是保守的未来使用决定，不证明所有双面薄壁的作者意图错误。

旧39条DEV / 8父完整保留，旧训练清单、曝光表、网格、权重和所有实验记录均未变。此前已排除的26条旧候选和3条新候选不重新加入。

## 如何使用

- [候选训练清单](manifest.json)：114条原始TRAIN记录，附原样DEV。不是已经登记的训练任务。
- [隔离清单](quarantine_manifest.json)：163条原始记录及逐条原因，重建并复核后才能在新版本恢复。
- [待核验清单](pending_audit_manifest.json)：44条；不能用NA冒充无缺陷。
- [全部321条决策](selection.csv)、[执行回执](execution_receipt.json)。

机器可读入口为 `r2/data_tools/bench_future_pool.json`，维护约定见 `r2/docs/bench_data_pool_policy.md`。此入口用于后续新Bench任务登记，不改历史实验加载器或维护模型默认。未来要训练须新建采样计划，不能沿用旧321条的曝光计划；本次没有启动训练。

## 尚未解决的质量问题

保留的114条中仍有{kept['known_Y_internal_duplicate_conditions']}条 / {kept['known_Y_internal_duplicate_parents']}父含已知Y内部重复面。这次没有仅凭反向重合就删除整父。全量Y-Y自交、部分共面重叠、窄缝、细片质量和真实用户输入适用性未获重新完整认证。

筛选使可用候选父数从79降至30；BACK_SURFACE仅3条 / 3父（另有3条SEAT_BACK组合）。因此不能马上把这个小池当成覆盖完整的新训练集，也不能把以后模型变化单独归因为目标更干净。待核验44条和隔离条件应在独立新版本补齐审计或重建，禁止为凑数量静默放回。

完整腿、闭合部件或多个有意义的独立部件不因“不是单张连通薄片”被自动排除。继续优先完整、规整、可辨认的人类部件输入；旧AI_PROXY语义审阅原样保存，不冒充本轮新图审。

本轮没有释放磁盘空间；它减少的是后续训练误用。恢复本轮检查：从仓库根目录使用CPU Python执行 `experiments/r2_bench_data_retirement_20261021/execute_future_filter.py --verify`。不传`--verify`的发布入口拒绝覆盖已有版本。
'''
    (OUT / 'README.md').write_text(report, encoding='utf-8')
    pointer = dict(schema='BENCH_FUTURE_POOL_POINTER_V1', status='CANDIDATE_ONLY_REQUIRES_NEW_TRAINING_REGISTRATION',
                   manifest=rel(OUT / 'manifest.json'), manifest_sha256=sha(OUT / 'manifest.json'),
                   quarantine_manifest=rel(OUT / 'quarantine_manifest.json'),
                   pending_audit_manifest=rel(OUT / 'pending_audit_manifest.json'),
                   selection=rel(OUT / 'selection.csv'), historical_replay_unchanged=True,
                   documentation='r2/docs/bench_data_pool_policy.md')
    write(POINTER, pointer)
    for path, expected in protected.items():
        assert sha(ROOT / path) == expected, path
    outputs = sorted(OUT.iterdir()) + [POINTER, Path(__file__).resolve()]
    write(OUT / 'execution_receipt.json', dict(schema='BENCH_FUTURE_POOL_FILTER_RECEIPT_V1',
            status='COMPLETE', created_utc=stamp, protected_source_hashes=protected,
            published_hashes={rel(path): sha(path) for path in outputs},
            ledger=dict(deleted_files=0, moved_files=0, source_modifications=0,
                        new_geometry_score_calls=0, new_reviews=0, model_calls=0,
                        GPU_calls=0, training_updates=0, reused_exact_audit_conditions=277,
                        pending_exact_audit_conditions=44), groups=summary))
    verify()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    verify() if args.verify else publish()
