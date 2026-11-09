"""Write cleanup closeout documents from the verified receipts; never delete assets.

Root executes this only after cleanup.ps1 -Stage Verify completed. Standard library
only. Existing documents are reused only when byte-identical; no silent overwrite.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DATE = '20261025'
NOTICE_NAME = f'WEIGHTS_RETIRED_{DATE}.md'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def windows(path):
    value = str(path).replace('\\', '/')
    if value.startswith('/mnt/e/'):
        value = 'E:/' + value[7:]
    return value


def host(path):
    value = windows(path)
    if os.name != 'nt' and value.lower().startswith('e:/'):
        value = '/mnt/e/' + value[3:]
    return Path(value)


def under(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def link(label, target, document):
    relative = os.path.relpath(host(target), Path(document).parent).replace('\\', '/')
    return f'[{label}](<{relative}>)'


def gib(n):
    return f'{int(n) / (1 << 30):.3f}'


def receipt_rows(plan, key):
    value = plan.get(key, [])
    return value if isinstance(value, list) else [value]


def locate_groups(removals):
    """Only new notices in ordinary local experiment/run roots; mirror everything else."""
    groups = {}
    for row in removals:
        value = windows(row['Path'])
        match = re.match(r'^(E:/MeshFlow(?:-Control-artifacts)?/experiments/[^/]+)/', value, re.I)
        kind = 'experiment'
        if not match:
            match = re.match(r'^(E:/MeshFlow/native_t1/runs/[^/]+)/', value, re.I)
            kind = 'native_run'
        if match:
            identity = match.group(1)
        else:
            # Legacy profile, abandoned worktree, or independently stored artifact.
            identity = str(Path(value).parent).replace('\\', '/')
            kind = 'other_weight_location'
        key = identity.lower()
        if key not in groups:
            local = host(identity)
            is_external = value.lower().startswith('e:/meshflow-control-artifacts/')
            if kind in ('experiment', 'native_run') and not is_external and under(local, ROOT):
                destination = local / NOTICE_NAME
            else:
                scope = 'External' if is_external else 'Other'
                root_tail = identity.split(':/', 1)[-1]
                destination = OUT / 'retirement_notices' / scope / root_tail / NOTICE_NAME
            assert under(destination, ROOT), f'Notice escapes workspace: {destination}'
            groups[key] = dict(identity=identity, kind=kind, destination=destination,
                               mirrored=is_external or kind == 'other_weight_location', rows=[])
        groups[key]['rows'].append(row)
    return sorted(groups.values(), key=lambda item: item['identity'].lower())


def model_label(path):
    value = windows(path)
    if 'official' in value.lower() or '/output/120m-ot-v-bench/' in value:
        return '官方资产（保留）'
    if '/r2_seam_pool47_couple_v1/checkpoints/PLAIN3000_step2000.pt' in value:
        return 'Chair PLAIN_S2000 完整恢复'
    if '/r2_seam_quality_wave1_wave2_v1/inference/PLAIN_S2000_generator.pt' in value:
        return 'Chair PLAIN_S2000 推理'
    if '/r2_seam_quality_four_v1/inference/LONG_occupancy.pt' in value:
        return 'Chair S2000 配套 LONG C-only 头'
    if '/r2_chair_multipart_seam_v1/' in value:
        return 'Chair MULTIPART_S1500：BACK 有限正结果'
    if '/r2_full_seat_learning_curve_v1/' in value:
        return 'Chair FULL_S6000 方法比较参考'
    if 'chair_hybrid_start' in value or 'OT_HYBRID_pilot1000_total5000' in value:
        return 'Chair START 维护默认'
    if '/r2_postprocess_avmr_pmr_v1/assets/chair_denoiser_' in value:
        return '官方 Chair D0'
    if '/r2_bench_parts_convergence_v1/training/COMMON/' in value:
        return 'Bench COMMON12000 历史起点'
    if '/r2_bench_full_seat_curriculum_v1/training/NATIVE/' in value:
        return 'Bench Native6000'
    if '/r2_bench_full_seat_curriculum_v1/training/SEAM/' in value:
        return 'Bench SEAM3000 当前研究参考'
    if '/r2_bench_seat_coverage_v1/training/NATIVE/' in value:
        return 'Bench Coverage1500：RAW 有限正结果'
    if '/r2_bench_backend_shape_preservation_v1/' in value:
        return 'Bench SHAPE500：保形候选，未晋级'
    if '/r2_bench_three_family_v1/production_contract/' in value:
        return 'Bench 冻结 C-only 布局头'
    return '其他保留的官方/公开模型或原依赖'


def make_notice(group, keep, report_path, receipt_path):
    path = group['destination']
    weights = [r for r in group['rows'] if r['Kind'] == 'RETIRED_WEIGHT']
    copies = [r for r in group['rows'] if r['Kind'] == 'ARCHIVED_DUPLICATE']
    retained = [r for r in keep if windows(r['Path']).lower().startswith(group['identity'].lower() + '/')]
    retained_weights = [r for r in retained if Path(r['Path']).suffix.lower() in ('.pt', '.pth', '.ckpt', '.safetensors')]
    lines = [f'# 2026-10-25 资产退役补记', '',
             f'原实验/资产目录：`{group["identity"]}`。', '',
             '本文件记录用户本轮授权清理之后的本地可用性，不修改该实验在运行当时的事实、指标、结论或历史完成回执。', '',
             f'实际移除 {len(weights)} 份权重及 {len(copies)} 份已有校验 ZIP 的展开审阅副本，逻辑大小 {gib(sum(int(r["Bytes"]) for r in group["rows"]))} GiB。', '']
    if group['mirrored']:
        lines += ['这是中央镜像告示；未向上述外部目录或旧工作树写入新文件。', '']
    if weights:
        lines += ['以下权重已退役，依赖它们的历史完整训练重放、对应旧检查点续训或重新模型采样在本地不再可用。代码或旧报告仍可能列出其原路径；不得自动下载、恢复、重训或将缓存输出冒充重新生成。', '',
                  '若本目录仍有保留终点，保留终点的推理/恢复能力与已退役中间状态分开；清理没有启动任何续训。', '']
    if copies:
        lines += ['审阅副本仅从逐文件白名单删除；已验证的原 ZIP、`PACKAGE_MANIFEST.csv`、原始实验数据及证据继续保留。展开目录的旧本地浏览路径可能失效，但 ZIP 中仍保存逐字节相同的成员。', '']
    lines += ['原 RAW、POST、固定噪声、查询、逐例表、图审、源代码、原报告、原 `COMPLETE/completion` 和历史 hash 不被本次权重退役覆盖。离线复算仍可使用保存的输出；要求已删权重存在的历史全链校验入口不再作为当前可执行入口。', '',
              '| 移除路径 | 类型 | Bytes | SHA256 | 原因 |', '| --- | --- | ---: | --- | --- |']
    for r in sorted(group['rows'], key=lambda x: x['Path'].lower()):
        lines.append(f'| `{cell(windows(r["Path"]))}` | {r["Kind"]} | {r["Bytes"]} | `{r["SHA256"]}` | {cell(r["Reason"])} |')
    if retained_weights:
        lines += ['', '本目录继续保留的权重：', '']
        for r in retained_weights:
            lines.append(f'- {link(Path(windows(r["Path"])).name, r["Path"], path)}：{gib(r["Bytes"])} GiB，SHA256 `{r["SHA256"]}`。')
    lines += ['', f'完整清理事实见 {link("清理报告", report_path, path)} 与 {link("实际完成回执", receipt_path, path)}。', '',
              '本次不切换默认模型，不训练、不生成，不 commit/push。', '']
    return '\n'.join(lines)


def run():
    plan_path = OUT / 'prepared_manifest.json'
    receipt_path = OUT / 'cleanup_receipt.json'
    archive_path = OUT / 'local_repair_archive_receipt.json'
    plan, receipt, archive = read(plan_path), read(receipt_path), read(archive_path)
    assert plan['Schema'] == 'EXACT_FILE_RETIREMENT_V1'
    assert receipt['Status'] == 'COMPLETE_VERIFIED', 'Root must finish deletion and verification first'
    assert archive['status'] == 'ARCHIVED_VERIFIED'
    assert receipt['PreparedManifestSHA256'] == sha(plan_path)
    assert receipt['LedgerSHA256'] == sha(OUT / 'deletion_ledger.jsonl')
    for key in ('NoTraining', 'NoGeneration', 'NoOriginalDatasetDeletion', 'NoDefaultChange', 'NoGitCommitPush'):
        assert receipt[key] is True, key
    removals, keep = receipt_rows(plan, 'Remove'), receipt_rows(plan, 'Keep')
    assert len(removals) == int(receipt['DeletedFiles'])
    assert sum(int(r['Bytes']) for r in removals) == int(receipt['LogicalBytes'])
    assert len({windows(r['Path']).lower() for r in removals}) == len(removals)
    assert not ({windows(r['Path']).lower() for r in removals} & {windows(r['Path']).lower() for r in keep})
    for row in removals:
        assert not host(row['Path']).exists(), f'Removed file exists: {row["Path"]}'
    for row in keep:
        assert host(row['Path']).is_file(), f'Protected file missing: {row["Path"]}'
        assert host(row['Path']).stat().st_size == int(row['Bytes'])
    # Small archive/metadata hashes only; the parent Verify already checked protected large tensors.
    assert host(archive['path']).stat().st_size == archive['bytes']
    assert sha(host(archive['path'])) == archive['sha256']
    local_experiment = ROOT / 'experiments/r2_bench_raw_local_repair_v1'
    assert sha(local_experiment / 'COMPLETE.json') == archive['source_seal_sha256']
    groups = locate_groups(removals)
    kinds = Counter(r['Kind'] for r in removals)
    kind_bytes = {k: sum(int(r['Bytes']) for r in removals if r['Kind'] == k) for k in kinds}
    weights = [r for r in keep if Path(r['Path']).suffix.lower() in ('.pt', '.pth', '.ckpt', '.safetensors')]
    report_path, retained_path = OUT/'CLEANUP_REPORT.md', OUT/'RETAINED_MODELS.md'

    report = ['# 2026-10-25 实验归档与存储清理', '',
        f'本轮已按逐文件白名单完成清理，并复核 {receipt["ProtectedHashChecks"]} 项保护资产的 hash。清理状态：`{receipt["Status"]}`；完成时间：`{receipt["UTC"]}`。', '',
        f'共移除 **{receipt["DeletedFiles"]} 个文件，逻辑大小 {gib(receipt["LogicalBytes"])} GiB**；E 盘可用空间实测增加 **{gib(receipt["ObservedFreeDeltaBytes"])} GiB**。', '',
        '逻辑大小是被移除路径的文件长度之和，包含硬链接；它不等于独立物理分配。卷可用空间变化也可能含其他进程同期活动，两个数分开报告。', '',
        '| 清理类型 | 文件数 | 逻辑 GiB |', '| --- | ---: | ---: |']
    for k in sorted(kinds):
        name = {'RETIRED_WEIGHT': '旧候选、非保留端点与中间/滚动权重', 'ARCHIVED_DUPLICATE': '已在保留 ZIP 中逐字节验证的展开审阅副本'}.get(k, k)
        report.append(f'| {name} | {kinds[k]} | {gib(kind_bytes[k])} |')
    report += ['', '## 本轮实验收口', '',
        'RAW 局部修复试点保持 `NO_TRAINING_DATA_GATE_FAILED`：12 请求中 6 份有非空 EDIT、4 份满足当前保守数值资格；新增最终局部修复集中在91e两份，并伴随未获得对照后处理新增的5条合法 F/F 共边。局部几何正证据保留，不升级为通用后端，也没有因此新增训练或权重。', '',
        f'已归档 {archive["scientific_members"]} 个科学文件，包含全部12份目标数组/掩码、逐例评分与配对表、实际图审、实际代码及12张同链 FINAL 对照图。审阅包 {archive["bytes"]:,} bytes（{archive["bytes"]/1_000_000:.3f} MB）：{link("R2_BENCH_RAW_LOCAL_REPAIR_V1_REVIEW_SMALL.zip", archive["path"], report_path)}。', '',
        f'ZIP SHA256：`{archive["sha256"]}`。完整原实验及跨阶段 RAW/D0/POST 缓存仍保留；小包不是可独立执行整个训练/模型流程的重放包。原实验记录见 {link("REPORT.md", local_experiment/"REPORT.md", report_path)}。', '',
        '## 保留与删除边界', '',
        f'继续保留的模型和使用身份见 {link("RETAINED_MODELS.md", retained_path, report_path)}。Chair S2000及其配套头、官方D0与CPU后处理保持；MULTIPART的BACK正例分支、FULL_S6000、START保持；Bench COMMON12000、Native6000、SEAM3000、Coverage1500和SHAPE500的推理/恢复资产保持。它们各有证据范围，没有合并称为单一全面最优模型。', '',
        '官方/公开模型和原始数据不删；实验数据集、来源网格、合法条件、面/角点身份、固定噪声/查询、RAW/POST、评分、图审、代码、训练日志与历史失败记录继续保留。数据侧本轮只移除已证明与保留 ZIP 成员逐字节相同的 `review_package` 展开副本，不把旧结果或唯一调试证据当缓存删掉。', '',
        '保留资产的旧报告和 `COMPLETE/completion` 不改写，原 hash 仍表示当时完成状态。删除权重的事实以本次清单、账本及旁置退役告示为准。已退役端点的完整历史重放/续训不可用；保存结果的离线分析仍可继续。不得为了通过旧校验而自动下载、恢复或重训已退役权重。', '',
        '## 逐实验退役告示', '',
        '外部 `MeshFlow-Control-artifacts` 目录的告示写入本中央目录的镜像树；没有向外部目录新增文档。', '',
        '| 实验/资产目录 | 移除文件数 | 退役说明 |', '| --- | ---: | --- |']
    for group in groups:
        report.append(f'| `{cell(group["identity"])}` | {len(group["rows"])} | {link("退役告示", group["destination"], report_path)} |')
    report += ['', '## 核验与边界', '',
        f'- {link("准备白名单与原文件 hash", plan_path, report_path)}',
        f'- {link("逐文件实际删除账本", OUT/"deletion_ledger.jsonl", report_path)}',
        f'- {link("最终清理回执", receipt_path, report_path)}',
        f'- {link("本轮审阅归档回执", archive_path, report_path)}', '',
        '本轮清理不训练、不运行生成模型、不重新评分、不改变模型默认、不 commit/push。保留恢复权重只意味着资产仍可用，不授权继续训练。', '']

    retained = ['# 当前保留模型与流程身份', '',
        '依据2026-10-25实际清理完成回执；以下是资产可用性与证据范围，不是统一排行榜，也不改变现有CLI默认。大权重的 hash 由最终清理核验验证，本文不重新加载模型。', '',
        '| 角色 | 保留内容与用途 | 证据边界 |', '| --- | --- | --- |',
        '| Chair 主研究流程 | PLAIN_S2000完整恢复与推理权重 + LONG C-only头 + 官方Chair D0；GEO_C/来源角序 → HINGE_OPP015，ALPHA/HOLE按归档合同使用 | 保存已验证的有限几何/视觉正结果；CPU后处理无额外神经权重 |',
        '| Chair BACK 分支 | MULTIPART_S1500完整恢复、推理与配套占用头 | 独立的靠背正结果，不自动替代S2000 |',
        '| Chair 比较与维护 | FULL_S6000完整状态/推理；START维护默认与原硬链接名称 | FULL为充分训练比较参考，START仍是维护CLI默认 |',
        '| Bench 起点 | COMMON12000完整状态/推理，官方Bench生成器及D0 | 保护历史来源与公共模型 |',
        '| Bench 当前研究参考 | Native6000、SEAM3000完整状态与推理，固定C-only小头 | 用户已保留的完整座面路线；不宣称DEV整体泛化已解决 |',
        '| Bench Coverage1500 | 最终完整状态与推理，现成RAW/评测数据 | 新输入上的RAW组织有有限正证据，不能取代SEAM3000整体参考 |',
        '| Bench SHAPE500后端 | candidate_denoiser与step0500完整恢复 | 保形/视觉局部收益保留，但未通过整体几何门槛，不替换官方D0 |', '',
        'RELDEF500权重退役：相对SHAPE没有新增严格视觉正例，DEV冲突代价仍在；原预测提议、网格、指标、图审和报告保留。最近RAW局部修复试点未通过数据准入，未产生新权重。', '',
        '## 精确保留权重', '',
        '| 角色 | 文件 | GiB | SHA256 |', '| --- | --- | ---: | --- |']
    for r in sorted(weights, key=lambda x: x['Path'].lower()):
        retained.append(f'| {cell(model_label(r["Path"]))} | {link(cell(windows(r["Path"])), r["Path"], retained_path)} | {gib(r["Bytes"])} | `{r["SHA256"]}` |')
    retained += ['', '同名历史中间/滚动文件不被视为终点副本；只保留本表列出的实际资产。START的两个保留路径为同一硬链接分配，不能按两个独立物理大文件计算空间。', '',
        f'全部保护文件（含数据清单、证据和归档）见 {link("prepared_manifest.json", plan_path, retained_path)}；实际移除路径见 {link("清理报告", report_path, retained_path)}。保留终点仍需其既有配置、元数据和数据合同；不从缺失历史检查点静默恢复，不自动启动任何新任务。', '']

    documents = {report_path: '\n'.join(report), retained_path: '\n'.join(retained)}
    for group in groups:
        documents[group['destination']] = make_notice(group, keep, report_path, receipt_path)
    # Preflight all destinations before writing any, and never rewrite historical files.
    for path, content in documents.items():
        assert under(path, ROOT)
        if path.exists():
            assert path.read_bytes() == content.encode('utf-8'), f'Existing different document: {path}'
    for path, content in documents.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x', encoding='utf-8', newline='\n') as stream:
                stream.write(content)
    result = dict(status='DOCS_COMPLETE', prepared_manifest_sha256=sha(plan_path),
                  cleanup_receipt_sha256=sha(receipt_path), archive_receipt_sha256=sha(archive_path),
                  notice_count=len(groups), documents={str(p): sha(p) for p in documents},
                  no_deletion=True, no_model_load=True, no_old_evidence_modified=True,
                  external_notices_central_only=True)
    destination = OUT/'closeout_docs_receipt.json'
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if destination.exists():
        assert destination.read_bytes() == encoded.encode('utf-8')
    else:
        with destination.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(encoded)
    print(json.dumps({k: v for k, v in result.items() if k != 'documents'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
