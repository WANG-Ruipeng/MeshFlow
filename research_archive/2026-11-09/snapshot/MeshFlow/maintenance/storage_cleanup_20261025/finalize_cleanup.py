"""Consolidate both exact-file retirement passes and verify effective retained assets."""
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path('/mnt/e/MeshFlow')
OUT = ROOT / 'maintenance/storage_cleanup_20261025'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def norm(p):
    return str(p).replace('\\', '/')

def host(p):
    p = norm(p)
    return Path('/mnt/e') / p[3:] if p.lower().startswith('e:/') else Path(p)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def write(p, s):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(s, encoding='utf-8')

def dump(p, v):
    write(p, json.dumps(v, ensure_ascii=False, indent=2))

def main():
    if (OUT / 'FINAL_CLEANUP_RECEIPT.json').exists():
        raise RuntimeError('Already finalized')
    a, b = read(OUT/'cleanup_receipt.json'), read(OUT/'legacy_small_receipt.json')
    p, q = read(OUT/'prepared_manifest.json'), read(OUT/'legacy_small_plan.json')
    assert a['Status'] == b['Status'] == 'COMPLETE_VERIFIED'
    assert a['PreparedManifestSHA256'] == sha(OUT/'prepared_manifest.json')
    assert b['PlanSHA256'] == sha(OUT/'legacy_small_plan.json')
    assert a['LedgerSHA256'] == sha(OUT/'deletion_ledger.jsonl')
    assert b['LedgerSHA256'] == sha(OUT/'legacy_small_deletion_ledger.jsonl')
    removed = {norm(x['Path']).lower() for x in q['Remove']}
    all_removed = p['Remove'] + q['Remove']
    for r in all_removed:
        assert not host(r['Path']).exists(), r['Path']
    keep = [r for r in p['Keep'] if norm(r['Path']).lower() not in removed]
    for i, r in enumerate(keep, 1):
        assert sha(host(r['Path'])) == r['SHA256'], r['Path']
        if i % 100 == 0:
            print(f'Final retained SHA256 checks {i}/{len(keep)}', flush=True)
    refs = read(OUT/'dataset_active_dependencies.json')['nonweight_references']
    all_removed_paths = {host(r['Path']) for r in all_removed}
    assert all(host(r['path']) not in all_removed_paths for r in refs)
    for r in refs:
        path = host(r['path'])
        assert path.exists(), str(path)
        if r['kind'] == 'file' and r.get('bytes') is not None:
            assert path.stat().st_size == r['bytes'], str(path)
    free_now = shutil.disk_usage(ROOT).free
    count = a['DeletedFiles'] + b['DeletedFiles']
    total = a['LogicalBytes'] + b['LogicalBytes']
    observed = free_now - a['FreeBytesBefore']
    weights = sum(r.get('Kind', 'RETIRED_WEIGHT') == 'RETIRED_WEIGHT' for r in all_removed)
    receipt = dict(status='COMPLETE_VERIFIED', removed_files=count, removed_weight_files=weights,
                   removed_archived_duplicates=count-weights, logical_bytes=total,
                   logical_gib=total/(1<<30), observed_free_delta_bytes=observed,
                   observed_free_delta_gib=observed/(1<<30), free_bytes=free_now,
                   effective_protected_sha256_checks=len(keep), active_references=len(refs),
                   pass1_receipt_sha256=sha(OUT/'cleanup_receipt.json'),
                   pass2_receipt_sha256=sha(OUT/'legacy_small_receipt.json'),
                   no_models_run=True, no_original_datasets_deleted=True,
                   no_default_change=True, no_commit_push=True,
                   scope='Supplement overrides only 237 generic first-pass KEEP rows after explicit legacy audit; current models and evidence unchanged.')
    dump(OUT/'effective_retained_assets.json', keep)
    dump(OUT/'FINAL_CLEANUP_RECEIPT.json', receipt)
    for name in ('CLEANUP_REPORT.md', 'RETAINED_MODELS.md'):
        shutil.copyfile(OUT/name, OUT/name.replace('.md', '_FIRST_PASS.md'))
    report = (OUT/'CLEANUP_REPORT.md').read_text(encoding='utf-8')
    report = report.replace('# 2026-10-25 实验归档与存储清理\n',
        '# 2026-10-25 实验归档与存储清理\n\n'
        f'**两阶段最终总计：删除{weights}份旧权重和{count-weights}份重复展开副本，共{count}文件；逻辑大小{total/(1<<30):.3f} GiB，E盘可用空间实测增加{observed/(1<<30):.3f} GiB。**\n\n'
        f'最终重新验证{len(keep)}项实际保留资产SHA256与{len(refs):,}条当前非权重引用；均通过。最终依据为[合并回执](FINAL_CLEANUP_RECEIPT.json)及[有效保留资产](effective_retained_assets.json)。\n\n'
        '补充阶段清理9月旧工作树AE/FM/PONQ的237份小权重（2.892GiB）。它们在第一遍仅因小文件暂留，随后按17组旧报告确认已封存、无当前维护依赖；不是官方权重。补充清单只覆盖这237项第一遍泛化KEEP判断，核心保留模型不变。见[补充审计](legacy_small_weights_audit.json)、[补充删除账本](legacy_small_deletion_ledger.jsonl)。\n\n'
        '下面保留第一阶段的分项事实；其删除量不得与上述最终总计再次相加。\n')
    write(OUT/'CLEANUP_REPORT.md', report)
    model_text = (OUT/'RETAINED_MODELS.md').read_text(encoding='utf-8')
    model_text = '\n'.join(line for line in model_text.splitlines()
                           if not any(norm(r['Path']).lower() in line.lower() for r in q['Remove'])) + '\n'
    model_text = model_text.replace('## 精确保留权重',
        '补充清理已移除第一阶段暂留的237份旧AE/FM/PONQ小权重。以下表格为补充后实际仍在的资产；最终清单与hash见[effective_retained_assets.json](effective_retained_assets.json)。\n\n## 精确保留权重')
    model_text = model_text.replace('其他保留的官方/公开模型或原依赖', '公开教师或保留的小型测试资产（按路径区分）')
    write(OUT/'RETAINED_MODELS.md', model_text)
    notice = ['# 旧工作树小权重补充退役（2026-10-25）', '',
              '用户授权清理旧权重；以下237份9月AE/FM/PONQ权重已移除。当前Chair/Bench模型无依赖。',
              '旧源码、训练日志、评分、数据和图仍保留；依赖这些权重的历史完整重放退役，不自动恢复。', '',
              '| 路径 | Bytes | SHA256 |', '| --- | ---: | --- |']
    for r in q['Remove']:
        notice.append(f'| `{norm(r["Path"])}` | {r["Bytes"]} | `{r["SHA256"]}` |')
    write(OUT/'retirement_notices/LEGACY_SMALL_WEIGHTS_RETIRED_20261025.md', '\n'.join(notice)+'\n')
    # Change only maintenance memory created in this task, never a historical experiment report.
    for filename, marker in [('WORKSPACE.md', '## 2026-10-25 当前轮次归档与旧权重退役'),
                             ('AGENTS.md', '## 2026-10-25 user-authorized storage retirement')]:
        file = ROOT/filename
        old = file.read_text(encoding='utf-8')
        before, after = old.split(marker, 1)
        after = after.replace('删除153份旧权重', f'删除{weights}份旧权重')
        after = after.replace('removed 153 obsolete weight files', f'removed {weights} obsolete weight files')
        after = after.replace('逻辑大小134.206GiB', f'逻辑大小{total/(1<<30):.3f}GiB')
        after = after.replace('实测增加132.748GiB', f'实测增加{observed/(1<<30):.3f}GiB')
        after = after.replace('842项保护资产SHA256复核通过', f'第一阶段842项、补充清理后实际保留{len(keep)}项资产SHA256复核通过')
        after = after.replace('maintenance/storage_cleanup_20261025/cleanup_receipt.json', 'maintenance/storage_cleanup_20261025/FINAL_CLEANUP_RECEIPT.json')
        if filename == 'AGENTS.md':
            after += '\nSupplemental legacy audit retired 237 small AE/FM/PONQ checkpoints from old worktrees; final totals and effective retained hashes are in `maintenance/storage_cleanup_20261025/FINAL_CLEANUP_RECEIPT.json` and `effective_retained_assets.json`. These override only first-pass generic KEEP entries; current retained model identities did not change.\n'
        write(file, before+marker+after)
    with (OUT/'RUNBOOK.md').open('a', encoding='utf-8') as f:
        f.write('\n\n## Final supplement\n\n`cleanup_legacy_small.ps1` completed the separately audited 237-file old-worktree retirement. The first-pass receipt is preserved as a historical stage, and its generic KEEP list is superseded for those exact files. Do not rerun the first-pass full verifier expecting these retired files to exist. The authoritative final state is `FINAL_CLEANUP_RECEIPT.json` and `effective_retained_assets.json`; `finalize_cleanup.py` verified the effective retained hashes and both deletion ledgers. Final documentation supersedes the `_FIRST_PASS.md` snapshots.\n')
    dump(OUT/'final_documentation_hashes.json', {name:sha(ROOT/name) for name in [
        'AGENTS.md','WORKSPACE.md','maintenance/storage_cleanup_20261025/CLEANUP_REPORT.md',
        'maintenance/storage_cleanup_20261025/RETAINED_MODELS.md']})
    print(json.dumps(receipt, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
