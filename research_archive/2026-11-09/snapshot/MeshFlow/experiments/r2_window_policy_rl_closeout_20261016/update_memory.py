"""CPU-only maintenance: preserve original scientific results, update route memory."""
from pathlib import Path
import difflib
import hashlib
import json

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
RUN = ROOT / 'experiments/r2_window_policy_rl_v1'
receipt = json.loads((OUT / 'cleanup_receipt.json').read_text(encoding='utf-8-sig'))
assert receipt['status'] == 'COMPLETE' and receipt['deleted_count'] == 63
assert receipt['deleted_logical_bytes'] == 605818757
targets = ['WORKSPACE.md', 'AGENTS.md', 'r2/docs/research_route_memory.md',
           'r2/docs/research_route_registry.json', 'r2/docs/research_history_sources.md']
backup = OUT / 'memory_before'
backup.mkdir(exist_ok=True)
originals = {}
for name in targets:
    p = ROOT / name
    data = p.read_bytes()
    dest = backup / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise RuntimeError('Memory backup already exists; inspect maintenance state before rerun.')
    dest.write_bytes(data)
    originals[name] = data

def replace_once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new, 1)

prefix = 'experiments/r2_window_policy_rl_closeout_20261016'
section = '''## R30：窗口随机策略 RL 收口（2026-10-16）

区别于 R29 的奖励端点 NFT，本轮采用窗口内实际高斯动作的 score-function/RLOO 更新。标定256输出通过事前门后，执行64批机会/512轨迹，54次实际AdamW更新、10次全零跳过；预定batch_063终点，不挑best。该配置的真实梯度、参数更新及旧logp重算检查通过，但这些不等于生成改进。[S86](../../experiments/r2_window_policy_rl_v1/REPORT.md)

固定新噪声44请求：TRAIN为8条已奖励C×4，DEV为3条历史开发C×4。FULL_DET/FULL_STOCH/RL_STOCH的合法接口与AI_PROXY严格联合均TRAIN9/32、DEV0/12，成功身份完全相同。主对照RL−FULL_STOCH新增/丢失均0；共边71→69条、总长度61.70846→56.79985，减少4.90861全部来自85b169/n0。TRAIN的CF/FF/I/小角变化为−0.02521/+0.65345/−0.21515/+0.23776pp；I沿用ConflictCov_C，非全网格自交率。保留局部收益及尾部代价，不把二值接口或AI图审当完整可用性认证。

**决定：当前配置无稳定升级，停止扩大，不自动延长、调奖励/窗口或切默认。** 不是所有RL被否定，也没有证明奖励投机；未来须提出超出R29/R30的实质变化及可检验预测。本轮TRAIN是跨噪声检查，DEV仅三条历史C，不夸大泛化。

用户授权清理63份batch_000..062中间权重，605,818,757字节（577.75MiB）；保留batch_063、FULL_S6000/官方模型及全部buffer、RAW/POST、评分、图审和旧报告。中间模型重放和旧控制器全链恢复退役，不自动补训。当前清理零模型调用。[收口报告](../../experiments/r2_window_policy_rl_closeout_20261016/NEGATIVE_RESULT_REPORT.md) · [清理回执](../../experiments/r2_window_policy_rl_closeout_20261016/cleanup_receipt.json)。

'''
memory = originals['r2/docs/research_route_memory.md'].decode('utf-8-sig')
memory = replace_once(memory, '更新：2026-10-15；原FULL_CONTOUR历史盘点补入NFT训练与冻结新噪声检查的收口。',
                      '更新：2026-10-16；在FULL_CONTOUR与NFT收口记录上补入窗口随机策略RL的实际结果及清理状态。')
memory = replace_once(memory, '## R29：', section + '## R29：')
memory = replace_once(memory, '本次补入两份NFT报告，来源索引共 **85份**',
                      '随后补入两份NFT报告与一份窗口RL报告，来源索引共 **86份**')
memory = replace_once(memory, '登记 **29条已运行但未形成稳定升级的候选路线（R01—R29）**',
                      '登记 **30条已运行但未形成稳定升级的候选路线（R01—R30）**')
memory = replace_once(memory, '“31种方法已被证明永远无效”', '“这些方法已被证明永远无效”')
memory = replace_once(memory, '## 28条已运行候选：不要原样重做', '## 原盘点28条候选（R01—R28）：不要原样重做；R29/R30见上方收口')
(ROOT / 'r2/docs/research_route_memory.md').write_text(memory, encoding='utf-8', newline='')

workspace_add = '''## 2026-10-16 窗口轨迹RL收口与权重清理

R2_WINDOW_POLICY_RL_V1完成标定及64批机会（54次实际AdamW更新/10次跳过）。固定44请求上，FULL_DET/FULL_STOCH/RL_STOCH的合法接口与严格联合均TRAIN9/32、DEV0/12，成功身份完全相同；RL相对同随机采样器的冻结FULL没有新增成功，共边总长度减少4.90861，TRAIN FF +0.65345pp，CF/I有局部下降。当前配置记为负结果，不扩大，不否定所有RL，不把标定通过/非零梯度当终态收益。

按用户授权已删除batch_000..062共63份中间权重，605,818,757字节（577.75MiB）；保留batch_063最终适配器、FULL_S6000、官方模型及全部RAW/POST/buffer/图审/评分/原报告。历史逐批模型重放及旧训练控制器全链恢复退役，不自动补训。原报告/ZIP/最终权重哈希未变。当前维护零训练/模型调用/重评分，不切默认，不commit/push。

[负结果收口](experiments/r2_window_policy_rl_closeout_20261016/NEGATIVE_RESULT_REPORT.md) · [清理回执](experiments/r2_window_policy_rl_closeout_20261016/cleanup_receipt.json) · [生命周期说明](experiments/r2_window_policy_rl_v1/CHECKPOINT_RETIREMENT_20261016.md)。研究登记新增R30，来源报告86份；FULL_S6000仍为方法比较参照。

'''
(ROOT / 'WORKSPACE.md').write_text(workspace_add + originals['WORKSPACE.md'].decode('utf-8-sig'), encoding='utf-8', newline='')

register = json.loads(originals['r2/docs/research_route_registry.json'])
assert not any(r['id'] == 'R30' for r in register['routes'])
register['version'] = 'research_route_memory_20261016_window_policy_rl_closeout_v1'
register['counts']['not_stable_upgrade'] = 30
register['counts']['primary_reports'] = 86
register['routes'].append({
    'id': 'R30', 'name': '窗口随机策略LoRA：实际动作REINFORCE/RLOO',
    'status': 'NOT_STABLE_UPGRADE', 'lifecycle': 'CLOSED_NEGATIVE_CURRENT_RECIPE',
    'evidence': '256标定通过；64批/512轨迹，54实际AdamW更新、10全零跳过。固定44请求三模型合法接口/J4均TRAIN9/32、DEV0/12且身份相同。RL相对FULL_STOCH无新增/丢失；共边71→69、长度61.70846→56.79985。TRAIN FF+0.65345pp、I−0.21515pp；实际AI_PROXY图审，保留UNKNOWN。',
    'nearest_prior_route': 'R29',
    'substantive_change': '窗口内实际随机动作的score-function梯度；非NFT端点回归。',
    'retry_boundary': '不盲目扩大当前配置；新机制须说明相对R29/R30的实质变化和可检验预测。负结果不否定全部RL，不证明奖励投机。',
    'source_ids': [86], 'closeout_report': prefix + '/NEGATIVE_RESULT_REPORT.md',
    'cleanup_receipt': prefix + '/cleanup_receipt.json',
    'retained': 'batch_063 final LoRA/AdamW; FULL_S6000 and official models; all buffers, RAW/POST, metrics, reviews, source and logs',
    'retired': 'batch_000..062 checkpoint.pt (63 files); historical per-batch model replay and original controller full-chain verification/resume unavailable'
})
(ROOT / 'r2/docs/research_route_registry.json').write_text(json.dumps(register, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

report_data = (RUN / 'REPORT.md').read_bytes()
report_sha = hashlib.sha256(report_data).hexdigest()
assert report_sha == 'fa6777bf12b1a32f4c87dbee8cfb3c469c7199b2d0e0979d63620c39362caedd'
sources = originals['r2/docs/research_history_sources.md'].decode('utf-8-sig')
sources = replace_once(sources,
    '2026-10-15. Primary reports: 85 (83 from the 2026-10-14 audit plus two NFT reports). This index records existing reports, not 85 independent methods or training replications.',
    '2026-10-16. Primary reports: 86 (83 from the 2026-10-14 audit, two NFT reports, and one window-policy RL report). This index records existing reports, not 86 independent methods or training replications.')
sources = sources.rstrip() + f'\n| S86 | [R2_WINDOW_POLICY_RL_V1](../../experiments/r2_window_policy_rl_v1/REPORT.md) | {len(report_data.splitlines())} | {len(report_data)} | `{report_sha}` |\n'
(ROOT / 'r2/docs/research_history_sources.md').write_text(sources, encoding='utf-8', newline='')

agents = originals['AGENTS.md'].decode('utf-8-sig').rstrip() + '''

## 2026-10-16 Window-policy RL negative-result closure

R30 records the actual windowed stochastic-policy LoRA/RLOO experiment, distinct from R29 endpoint NFT. Calibration passed; 64 opportunities produced 54 optimizer updates and 10 skips. Frozen final evaluation has identical FULL_DET/FULL_STOCH/RL_STOCH legal-interface and strict-joint identities: TRAIN9/32, DEV0/12. RL lost 4.90861 total legal interface length versus FULL_STOCH; local CF/I gains coexist with FF costs. Do not automatically extend this configuration or infer all RL is invalid. At user request, batch_000..062 checkpoint.pt (63 files) were deleted; batch_063, FULL_S6000, official models and all evidence remain. Historical intermediate replay and the old controller full-chain resume are retired. Current route register has 30 tested candidates and two pre-training gate blocks. See experiments/r2_window_policy_rl_closeout_20261016/NEGATIVE_RESULT_REPORT.md and its cleanup_receipt.json. This memory does not authorize new training, cleanup, commit or push.
'''
(ROOT / 'AGENTS.md').write_text(agents, encoding='utf-8', newline='')

lifecycle = '''# 当前检查点生命周期（2026-10-16）

R2_WINDOW_POLICY_RL_V1 已完成并收口为当前配置的负结果。用户随后授权清理中间权重。

- 已删除：training/batch_000..062/update/checkpoint.pt，共63份、605,818,757字节。
- 保留：batch_063终点（SHA256 bbda0ccaa1c86160b0574a6071a95d5eacc4e38d5db1051b9a26cae85ca26603）、FULL_S6000基座、官方模型及所有输入、buffer、RAW/POST、评分/图审、日志、代码与回执。
- 可复核：保留结果与逐例配对；终点推理依赖完整，但本次维护不授权新生成。
- 已退役：中间模型逐批重放/优化器恢复，以及原训练控制器全链校验/恢复。不要运行 `resume.ps1 -Phase train` 期待恢复；不要因旧路径缺文件自动补训。
- 原报告、README_RESUME.md、final.json与审阅ZIP未改，描述的是清理前的历史完成状态；本文件补充当前存储事实。固定64批合同结束，终点含优化器状态不授权续训。

[收口报告](../r2_window_policy_rl_closeout_20261016/NEGATIVE_RESULT_REPORT.md) · [逐文件清理回执](../r2_window_policy_rl_closeout_20261016/cleanup_receipt.json)
'''
(RUN / 'CHECKPOINT_RETIREMENT_20261016.md').write_text(lifecycle, encoding='utf-8')

diff = []
manifest = []
for name, before in originals.items():
    after = (ROOT / name).read_bytes()
    diff.extend(difflib.unified_diff(before.decode('utf-8-sig').splitlines(True), after.decode('utf-8').splitlines(True), fromfile='before/' + name, tofile='after/' + name))
    manifest.append({'path': name, 'before_sha256': hashlib.sha256(before).hexdigest(), 'after_sha256': hashlib.sha256(after).hexdigest()})
(OUT / 'maintenance.diff').write_text(''.join(diff), encoding='utf-8')
(OUT / 'maintenance_manifest.json').write_text(json.dumps({'status': 'COMPLETE', 'scientific_files_modified': 0, 'source_report_sha256': report_sha, 'memory_changes': manifest, 'original_report_and_zip_unchanged': True}, indent=2), encoding='utf-8')
print(json.dumps({'status': 'COMPLETE', 'route': 'R30', 'sources': 86, 'deleted_count': 63, 'bytes': 605818757}))
