"""Human report and portable OBJ meshes; all attempt records remain authoritative."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from ..io_utils import file_hash,write_json

LABEL={'fixed':'Fixed','a':'原 A','opt_fixed':'Fixed→优化','opt_a':'A→优化','vtk_qem':'VTK QEM','cgal_constrained':'受约束 CGAL QEM'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run)
    summary=json.loads((run/'SUMMARY.json').read_text());rows=json.loads((run/'results.json').read_text())['rows']
    lookup={(r['uid'],r['arm']):r for r in rows};ss={r['arm']:r for r in summary['summary']}
    hot=json.loads((run/'HOT_TIMING.json').read_text()) if (run/'HOT_TIMING.json').exists() else {'rows':[]}
    med=lambda v:float(np.median(v)) if v else None
    warm={arm:med([r['median_hot_seconds'] for r in hot['rows'] if r['arm']==arm]) for arm in ['fixed','a','vtk_qem']}
    both=[]
    for uid in sorted({r['uid'] for r in rows}):
        aa=lookup.get((uid,'opt_a'));ff=lookup.get((uid,'opt_fixed'))
        if aa and ff and aa.get('qualified') and ff.get('qualified'):
            both.append((uid,aa['metrics']['rms_bbox_percent']/ff['metrics']['rms_bbox_percent']))
    lines=[
      '本轮结论：布局有优化空间，原A尚未在预定质量约束下证明稳定收益。',
      '',
      f"已记录 {summary['recorded_attempts']}/96 次布局尝试；缺项 {len(summary['missing'])}；程序异常 {len(summary['execution_failures'])}。16 个对象均为已见开发对象，所有方法使用同一固定主场。本轮没有训练 AE、FM 或新 A，也没有执行 MeshFlow 长训练。",
      '',
      '预定合格条件：恰好512面、公共有效性审计通过，且实际三维三角形的最小角和最低角q05均不低于该对象Fixed（1e-6度数值容差）。该条件是严格诊断合同，不是通用工业标准。',
      '',
      '|方法|有效输出|512面|质量达标|全部条件达标|RMS中位数（bbox %）|q05中位数（度）|原始运行秒中位数|',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in LABEL:
        s=ss[arm];rms=f"{s['median_rms_bbox_percent']:.4f}" if s['median_rms_bbox_percent'] is not None else '—'
        if arm=='cgal_constrained':rms='—（面数不齐）'
        q=f"{s['median_q05_deg']:.2f}" if s['median_q05_deg'] is not None else '—'
        t=f"{s['median_algorithm_seconds']:.4f}" if s['median_algorithm_seconds'] is not None else '—'
        lines.append(f"|{LABEL[arm]}|{s['valid_count']}/16|{s['budget_match_count']}/16|{s['quality_pass_count']}/16|{s['qualified_count']}/16|{rms}|{q}|{t}|")
    lines += ['', '上表RMS和角度保留全部可读输出，包含不满足质量条件的结果，不能当作合格成功子集的性能。CGAL输出实际面数范围为'
        +f"{ss['cgal_constrained']['actual_faces_min']}–{ss['cgal_constrained']['actual_faces_max']}；完整误差仍保存在逐样本账本中。",
        '', '关键配对读数', '',
        f"- 原A在 {ss['a']['rms_improved_count']}/16 个对象上比Fixed误差低；逐对象误差比的中位数为 {ss['a']['median_rms_over_fixed_all_readable_NOT_equal_budget_ranking']:.3f}。其质量达标为 {ss['a']['quality_pass_count']}/16，两项必须一起解释。",
        f"- Fixed→优化在 {ss['opt_fixed']['target_count_both_seeds']}/16 个对象上，满足全部约束且两组独立评测采样均达到至少10%的RMS改善。逐对象误差改善中位数为 {(1-ss['opt_fixed']['median_rms_over_fixed_all_readable_NOT_equal_budget_ranking'])*100:.2f}%。",
        f"- A→优化在 {ss['opt_a']['target_count_both_seeds']}/16 个对象上通过上述两次采样复核（第一评测种子为 {ss['opt_a']['target_count']}/16，存在一个临界对象）；满足全部质量和预算条件为 {ss['opt_a']['qualified_count']}/16。",
        f"- 两种优化均满足约束的对象共 {len(both)} 个，其中A初始化最终误差更低的有 {sum(x[1]<1 for x in both)} 个。这是条件比较，不能替代全16对象的成功率。",
        '', '计时和数值复核', '',
        f"- 单独热测三次中位数：Fixed {warm['fixed']:.6f}s，原A {warm['a']:.6f}s，VTK QEM {warm['vtk_qem']:.6f}s。重复输出网格哈希全部一致：{all(r['same_mesh_hash_all_repeats'] for r in hot['rows'])}。",
        '- 原始计时保留冷调用开销；优化只运行一条既定轨迹，包含CPU最近面搜索、CUDA梯度、传输、保护与可行化。独立最终审计与评测另外记账。',
        '- 质量恢复失败的优化可能更快，但没有产生合格结果；不能把包含这些失败的耗时中位数解释为加速。速度比较必须同时满足误差和质量目标。',
        '- 第二组独立8192点评测在rechecks中；SUMMARY包含采样波动和相对Fixed改善方向的一致性。这是同一输出的采样复核，不是独立训练或优化复现，也不是统计置信区间；10%是本轮预定的探索阈值。']
    for arm in LABEL:
        s=ss[arm]
        lines.append(f"  - {LABEL[arm]}：两组RMS绝对差中位数 {s['recheck_rms_absolute_difference_median']:.6f} 个bbox百分点；改善方向一致 {s['recheck_improvement_sign_consistent_count']}/16；两组采样都达到10%改善且满足约束的对象 {s['target_count_both_seeds']}/16。")
    lines += ['', '解释边界', '',
      '- 有限预算局部优化的停滞不能证明表示没有更好解。Fixed→优化的最佳迭代位于1–27步（中位11.5），约99.83%的提案被拒绝；这是本优化器找到的收益，不是充分优化上界。角度min/q05是非光滑约束，恢复与投影可能停滞。优化目标采用近似三点积分，独立面积RMS优先。',
      '- 原A未经本轮目标训练；若优化证明有空间而A不能稳定取得收益，下一步应评估任务对齐的A训练，不应直接扩大FM训练。',
      '- CGAL6.1.1使用明确命名的classic-plane QEM和自定义保护；逐步最低角保护只检查改变的候选面，global q05只在端点审计，原有低质量面可能残留。因此它与方向优化器并非全程处于同一双角度可行域，未达预算或质量不能解释成传统方法普遍无法解决该问题。',
      '- 本轮不支持未见形状/跨预算泛化、完整生成质量或MeshFlow优劣结论。',
      '', '查看结果', '',
      '- [配对误差图](figures/paired_rms.png)',
      '- [误差与时间图](figures/rms_time.png)',
      '- [误差与面质量图](figures/rms_quality.png)',
      '- 网格对照图：figures/meshes_*.png（四页，全部16对象）。',
      '- [逐样本CSV](PER_SAMPLE.csv)、[完整汇总JSON](SUMMARY.json)、[方法与复现](METHOD_AND_REPRODUCE_zh.md)、[实验协议](RUN_PROTOCOL.json)。',
      '- 所有网格原始数组：attempts/<UID>/<arm>/mesh.npz；OBJ：meshes_obj/<UID>/<arm>.obj。',
      '- 每次优化保留optimization.json、完整接受/拒绝账本及固定迭代快照。']
    (run/'REPORT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    for row in rows:
        if row.get('status')!='success':continue
        target=run/'meshes_obj'/row['uid']/(row['arm']+'.obj');target.parent.mkdir(parents=True,exist_ok=True)
        with np.load(run/row['mesh_file'],allow_pickle=False) as z:v=z['vertices'];f=z['faces']
        with target.open('x',encoding='ascii') as out:
            out.write('# Frozen-field layout diagnostic; no repair or vertex change\n')
            for xyz in v:out.write('v '+' '.join(format(x,'.17g') for x in xyz)+'\n')
            for ijk in f:out.write('f '+' '.join(str(int(i)+1) for i in ijk)+'\n')
    files=[p for p in run.rglob('*') if p.is_file() and p.name!='MANIFEST_SHA256.json']
    write_json(run/'MANIFEST_SHA256.json',{'files':{str(p.relative_to(run)):file_hash(p) for p in sorted(files)}})
    print(json.dumps({'report':str(run/'REPORT_zh.md'),'manifest_files':len(files),'obj_exports':sum(r.get('status')=='success' for r in rows)}))
if __name__=='__main__':main()




