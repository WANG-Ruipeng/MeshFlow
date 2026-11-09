"""Summarize completed repair and independent references; no scientific execution."""
import csv
import difflib
from pathlib import Path
import shutil
import statistics
import time
from .common import *
from .ledger import Budget
from .registration import verify

def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+
        ['|'+'|'.join(str(v) for v in row)+'|' for row in rows])

def csvfile(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r));path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def finalize(out=OUT):
    out=Path(out);pre=read_json(out/'precheck/result.json');ref=read_json(out/'reference/reference_result.json')
    synthetic=read_json(out/'reference/synthetic_result.json');monitor=read_json(out/'tests/monitor/result.json')
    identity=verify(out);atomic_json(out/'registration/source_verification_after.json',identity)
    account=Budget(out,'finalize').snapshot();s=account['summary']
    assert (s['precheck_forwards'],s['precheck_zero_update_backwards'])==(76,20)
    assert (s['cumulative_precheck_forwards'],s['cumulative_precheck_backwards'])==(86,21)
    assert s['failed_actual_calls']==0 and not s['pending']
    assert all(v['all_state_restored'] and v['telemetry']['background_healthy'] for v in pre['arms'].values())
    zero=[r for r in pre['comparisons'] if r['kind']=='zero_identity']
    assert len(zero)==16 and all(r['passed'] and r['max_abs']==0 for r in zero)
    assert ref['status']=='PASS_REFERENCE_SELF_CONSISTENCY' and len(ref['pairs'])==8
    costs=[]
    for task in pre['cost']['per_fixture']:
        for arm in ARMS:
            m=task['medians'][arm]
            calls=[r for r in pre['rows'] if r['stage']=='cost_measured' and r['arm']==arm and r['fixture_id']==task['fixture_id']]
            costs.append(dict(fixture_id=task['fixture_id'],arm=arm,measured_pairs=m['pairs'],
                forward_ms=1000*m['forward_seconds'],forward_backward_ms=1000*m['forward_backward_seconds'],
                relation_cuda_ms=statistics.median(r['profile']['relation_milliseconds'] for r in calls),
                correction_cuda_ms=statistics.median(r['profile']['attention_correction_milliseconds'] for r in calls),
                full_forward_cuda_ms=statistics.median(r['forward_cuda_milliseconds'] for r in calls),
                peak_allocated_bytes=max(r['torch_peak_allocated_bytes'] for r in calls),
                peak_reserved_bytes=max(r['torch_peak_reserved_bytes'] for r in calls)))
    csvfile(out/'results/cost_per_fixture_arm.csv',costs)
    numeric=[]
    for pair in ref['pairs']:
        spec=pair['pair'];comp=next(r for r in ref['comparisons'] if r['pair_id']==spec['pair_id'] and r['rule_n']==8)
        numeric.append(dict(pair_id=spec['pair_id'],fixture_id=spec['fixture_id'],t=spec['t'],
            query_index=spec['query_index'],known_index=spec['known_index'],
            mu_L2=comp['mu_L2'],covariance_Frobenius=comp['covariance_Frobenius'],compressed_mass_abs=comp['compressed_mass_abs'],
            reference_self_consistent=pair['reference_self_consistent'],production_192_accuracy_pass=comp['original_accuracy_pass']))
    csvfile(out/'results/independent_reference_summary.csv',numeric)
    native_failures=[r for r in pre['comparisons'] if r['kind']=='nonzero_face_corner_permutation' and r['arm']=='H_CONT' and not r['passed']]
    padding_failures=[r for r in pre['comparisons'] if r['kind']=='padding_valid_outputs' and not r['passed']]
    result=dict(status='COMPLETED_FINITE_REPAIR_AUDIT',implementation_execution='DONE',monitor_repair='PASS',
        actual_counts=s,CPU_reference_calls=ref['actual_reference_calls']+synthetic['actual_reference_calls'],
        CPU_kernel_evaluations=ref['actual_kernel_points']+synthetic['actual_kernel_points'],old_rule_CPU_calls=ref['actual_old_relation_cpu_calls'],
        zero_initialization='PASS_BITWISE_FP32_AND_BF16',old_baseline_continuity='PASS_BITWISE',
        independent_reference='PASS_SELF_CONSISTENCY_NOT_RIGOROUS_CERTIFICATE',
        candidate_192_accuracy='FAIL_ON_REGISTERED_HARD_PAIRS',full_model_cost=pre['cost']['status'],
        nonzero_permutation='FAIL_FROZEN_TOLERANCE_INCLUDING_H_CONT',padding='FAIL_FROZEN_TOLERANCE_IN_RELATION_ARMS_AT_MAXIMUM_N',
        former_monitor_failure_resolved=True,scientific_gates_pass=False,formal_training='NOT_RUN',
        method_quality='NA_NOT_EVALUATED',default_model_changed=False,protected_identity=identity,
        budget_amendment='Explicit implementation continuation; retains original cumulative96F/24B hard caps, reuses8 old H baselines and measuresM once vs H/S twice per fixture')
    atomic_json(out/'results/summary.json',result)
    maximum_self={k:max(p['two_tolerance_errors'][k] for p in ref['pairs']) for k in ('mu_L2','covariance_Frobenius','compressed_mass_abs')}
    costtable=[]
    for t in pre['cost']['per_fixture']:
        costtable.append([t['fixture_id'],f"{t['surface_to_H_CONT_forward_median_ratio']:.4f}",
            f"{t['surface_to_H_CONT_forward_backward_median_ratio']:.4f}",'PASS' if t['passed'] else 'FAIL'])
    zerotable=[]
    for arm in ARMS[1:]:
        for precision in ('FP32','BF16'):
            values=[r for r in zero if r['arm']==arm and r['precision']==precision]
            zerotable.append([arm,precision,max(r['max_abs'] for r in values),'逐位一致'])
    permrows=[]
    for arm in ARMS:
        for kind in ('nonzero_face_corner_permutation','padding_valid_outputs'):
            v=[r for r in pre['comparisons'] if r['arm']==arm and r['kind']==kind]
            permrows.append([arm,kind,sum(r['passed'] for r in v),len(v),f"{max(r['max_abs'] for r in v):.9g}",sum(r['outside_tolerance'] for r in v)])
    text=f'''# Chair Finite Surface Attention：修复后续跑与独立参考

本次已修复实现缺陷并实际完成全部计划检查。新增 **76次真实模型前向、20次零更新反向**；连同旧轮累计 **86/96前向、21/24反向**。新调用异常0、日志异常0、未决调用0。正式optimizer更新0、新生成0，三臂模型、AdamW与RNG均恢复。独立CPU完成34次积分、1472625个核点评价和24次既有求积规则调用。

**当前问题已从缺少工程检查推进到明确的数值与成本结果：新增关系分支真实零初始化通过；固定192点候选的精度不足，且两个40%上下文任务的前向成本略超3倍门槛。**总体状态为`COMPLETED_FINITE_REPAIR_AUDIT`；正式训练和模型质量比较仍为NOT_RUN/NA。

## 实现已修复、真实模型已执行

旧错误来自显存监测器重复传`time_unix/pid`。修订监测器在追加日志前剔除保留键，保存原采样时间，并为后台异常提供独立错误文件和同步保护。模拟成功、N/A、超时、后台异常、并发采样，以及真实nvidia-smi只读测试通过；本轮三臂遥测均健康。

旧代码、报告、账本和raw保持只读；修订代码位于`experiments/mf_h1_local0/chair_surface_attention_v1r1/`。这次续跑依据用户明确的继续授权，单独登记协议，不回写原失败为成功。

两个真实普通任务的零初始化检查如下。旧H_CONT八份重复输出直接复用，并通过本来就要执行的四次H_CONT前向再次核对当前运行路径，差异均0，不为核验增加调用。

{table(['关系臂','主干精度','最大绝对差','相对原H_CONT'],zerotable)}

两关系臂新增104496个参数，初始化及临时非零参数逐位相同；真实旧AdamW逐参数恢复step5000，新参数step0及零矩；所有正式预检均零optimizer更新。每个实际关系forward只构造一次R，四层各使用自己的消息MLP；12次关系臂完整F+B均已核对R一次与第3/6/9/12层事件，未重复四次积分。

## 独立参考回答了什么

固定8例包含3个原普通查询最坏点、1个原密集参考未收敛点、4个t≤0.5对照。采用三角形局部坐标/Duffy映射后的FP64自适应Gauss–Kronrod积分，与原均匀三角格不同；在三角形最近点移位指数、中心化求矩，避免远点下溢和大坐标二阶矩相减。两档相对精度1e-8/1e-10，几何分片点按输入预登记。

八例两档精度的最大特征差异为：均值L2 **{maximum_self['mu_L2']:.3g}**，协方差Frobenius **{maximum_self['covariance_Frobenius']:.3g}**，压缩质量 **{maximum_self['compressed_mass_abs']:.3g}**，均低于冻结自洽要求1e-5。SciPy误差估计和归一化传播估计另存；它们不是严格数学认证，也不是对全部7504对的重新积分。

固定192点与独立参考的实际差异：

{table(['固定例','t','mu L2','Sigma Frobenius','压缩质量','三项各<0.01'],[
    [r['pair_id'],r['t'],f"{r['mu_L2']:.9g}",f"{r['covariance_Frobenius']:.9g}",f"{r['compressed_mass_abs']:.9g}",'PASS' if r['production_192_accuracy_pass'] else 'FAIL'] for r in numeric])}

结果支持“晚期窄核相对三角格欠分辨”的假说：3个最坏点的显著误差依然存在；四个t≤0.5对照的192点最大特征差异仅约2.05e-5。原12288点参考在worst_mu仍有约0.003286均值误差，在原未收敛点约0.001291；49152点在这8例三项均低于0.001。不能把后者当作全任务收敛证明，更没有替换生产节点后重新训练。

完整方法、每次积分误差估计与账本见[独立参考报告](reference/REFERENCE_REPORT.md)，实际数据见[逐例比较CSV](reference/reference_comparisons.csv)。

## 真实成本已补齐

同一真实任务、同一模型起点、同一BF16/FP32合同、单GPU串行执行。H_CONT与SURFACE每任务各2次完整F+B、MOMENT各1次，均有前向预热。MOMENT次数减少是为了保留原总反向24次硬上限，已在修订协议中预登记；它的成本只能作单次描述。旧缺失计时的第一次F+B只计预算，未伪造为有效测量。

{table(['任务','SURFACE/H完整forward','SURFACE/H完整F+B','两项均≤3'],costtable)}

两个40%任务的完整前向为 **3.0508倍、3.1355倍**，越过3倍止损线；所有F+B倍率在1.91–2.65之间。当前候选成本门禁为COST_BLOCKED。次数有限，且两个前向结果接近门槛，不能外推为稳定性能排行榜；未因接近阈值补跑挑结果。

{table(['任务','臂','测量次数','完整F ms','完整F+B ms','关系CUDA ms','四层修正CUDA ms'],[
    [r['fixture_id'],r['arm'],r['measured_pairs'],f"{r['forward_ms']:.3f}",f"{r['forward_backward_ms']:.3f}",
     f"{r['relation_cuda_ms']:.3f}",f"{r['correction_cuda_ms']:.3f}"] for r in costs])}

关系与修正CUDA事件处于完整forward内部，不能与完整时间相加。完整F+B结束于backward同步后，不包含之后的梯度范数统计；包含原运行时检查和loss构造。峰值Torch allocated/reserved及进程/全局显存原始观测均保存。

本轮全局显存最小空闲至少9589 MiB，高于登记余量2445.45 MiB。WSL进程显存N/A仍保留N/A，使用采样全局占用作保守上界证据并注明可能漏掉短峰；不把Torch allocated等同进程峰值。没有OOM，没有改变batch、精度或自然N/K。

## 非零置换与padding的限制

{table(['臂','检查','通过数','总任务','最大绝对差','阈外坐标数'],permrows)}

冻结的FP32容差仍为atol=1e-6、rtol=1e-5，没有放宽。原H_CONT本身的四项面/角置换均未过该严格数值容差，最大差约1.17e-5，因此不能把新增臂的全部置换差异归因于关系算子。它们在数学上应等变，有限精度的实际数值合同仍未全部通过；原模型失败也不能自动豁免新模块。

padding方面H_CONT四例逐位保持；MOMENT/SURFACE分别在两个最大N任务有约5e-6–7e-6差异，普通任务逐位保持。与原H_CONT不同的这部分必须单列，不能写成所有mask检查通过。小模块FP64与非零分支的既有测试保留，但不代替实际模型结果。逐元素差异及独立归因审计见`registration/`和`precheck/real_comparisons.csv`。

## 下一步判断

实现现在已经能在真实Chair模型完成检查，继续训练所需的工程接口不再缺失。科学证据仍不支持直接启动固定192点候选的三臂训练：独立参考确认其精度不足，40%前向成本略越界，严格非零置换/padding数值合同也有未解决项。

优先下一步是**在这批固定失败输入上实现并验证能解析边/角附近窄核的有限三角积分算法**，以本次独立参考作裁判，同时保留真实模型成本约束。检验假说是误差取决于三角子格相对核集中区域的尺度，单纯保留同一均匀192点不能稳定控制该误差。工程实现应优先利用三角形局部坐标或条件高斯积分结构；本轮只完成参考和现有候选测量，没有换新生产算法后启动训练。

SURFACE对H_CONT、SURFACE对MOMENT、MOMENT对H_CONT及三臂对START的质量比较均为NA。当前证据不够选择MOMENT为有效模型，也不否定有限表面关系本身；确认的是当前192点实现尚不满足注册合同。旧START的138个raw和完整基线表仍在上一轮目录，本轮没有重新生成。

所有源码、原始输出、成本CSV、独立参考、累计账本、修订协议及hash均保留。固定修复与下一步参考核验已完成，未改变默认模型，未追加训练或采样。
'''
    (out/'CHAIR_FINITE_SURFACE_ATTENTION_R1_REPORT.md').write_text(text,encoding='utf8')
    # Static scientific export from stored measurements; no new model call.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    tasks=pre['cost']['per_fixture'];fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    x=np.arange(len(tasks));axes[0].bar(x-.18,[t['surface_to_H_CONT_forward_median_ratio'] for t in tasks],.36,label='Full forward')
    axes[0].bar(x+.18,[t['surface_to_H_CONT_forward_backward_median_ratio'] for t in tasks],.36,label='Full F+B')
    axes[0].axhline(3,color='black',linestyle='--');axes[0].set_xticks(x,[t['fixture_id'] for t in tasks],rotation=20)
    axes[0].set_ylabel('SURFACE / H_CONT (same task)');axes[0].legend();axes[0].set_title('Measured whole-model cost')
    for n in (8,64,128):
        vals=[next(c['mu_L2'] for c in ref['comparisons'] if c['pair_id']==p['pair']['pair_id'] and c['rule_n']==n) for p in ref['pairs'][:4]]
        axes[1].plot(range(4),vals,marker='o',label=str(3*n*n)+' nodes')
    axes[1].axhline(.01,color='black',linestyle='--');axes[1].set_yscale('log')
    axes[1].set_xticks(range(4),[p['pair']['pair_id'] for p in ref['pairs'][:4]],rotation=20)
    axes[1].set_ylabel('Mean-vector L2 difference');axes[1].legend();axes[1].set_title('Against independent adaptive reference')
    (out/'figures').mkdir(exist_ok=True);fig.savefig(out/'figures/repair_evidence.png',dpi=160);plt.close(fig)
    with (out/'CHAIR_FINITE_SURFACE_ATTENTION_R1_REPORT.md').open('a',encoding='utf8') as f:
        f.write('\n![实测成本和独立参考差异](figures/repair_evidence.png)\n')
    (out/'RUN_AND_RESUME.md').write_text('''# Finite repair entry

From WSL:

```bash
cd /mnt/e/MeshFlow
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.mf_h1_local0.chair_surface_attention_v1r1.run_all --resume
```

The completed GPU phase is reused, never rerun. It verifies the protected original run and frozen repair hashes, then regenerates the report from stored results. The independently frozen CPU reference and its ledgers are already complete and never silently replayed. An interrupted dispatch requires inspecting its durable reservations before an explicit repair; no uncounted restart is allowed.

Executed schedule: 76 new forwards / 20 zero-update backwards, cumulative 86 / 21 of the original 96 / 24 hard limits. H_CONT and SURFACE cost have two measurements per task; MOMENT one, disclosed in the amendment. Unused allowances do not authorize new scientific jobs. Formal training and new generation have no entry in this repair.

The old failed experiment remains read-only. Its known baseline raw is referenced, not copied or regenerated. Source diffs, hashes, resources, CPU reference protocols and all measurement CSV/NPZ evidence are retained here.
''',encoding='utf8')
    code_files=sorted(Path(__file__).parent.glob('*.py'));source_hashes={}
    diff=[]
    for p in code_files:
        rel=p.relative_to(ROOT).as_posix();h=digest(p);target=out/'source_code'/p.name;target.parent.mkdir(exist_ok=True)
        if not target.exists():shutil.copyfile(p,target)
        assert digest(target)==h;source_hashes[rel]=h
        old=ROOT/'experiments/mf_h1_local0/chair_surface_attention_v1'/p.name
        diff.extend(difflib.unified_diff(old.read_text(encoding='utf8').splitlines(True) if old.exists() else [],
            p.read_text(encoding='utf8').splitlines(True),fromfile=old.relative_to(ROOT).as_posix() if old.exists() else '/dev/null',tofile=rel))
    atomic_json(out/'source_code/source_hashes.json',source_hashes)
    (out/'code.diff').write_text(''.join(diff),encoding='utf8')
    atomic_json(out/'RUN_COMPLETE.json',dict(status='DONE_FINITE_REPAIR_AUDIT',completed_unix=time.time(),
        planned_actual_calls_complete=True,scientific_gates_pass=False,training_started=False,default_changed=False))
    atomic_json(out/'delivery_hashes.json',{p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*'))
        if p.is_file() and p.name!='delivery_hashes.json' and not p.name.endswith('.lock')})
    print('REPAIR_COMPLETE: 76F/20B new; 86F/21B cumulative; all states restored; measured gates not passed.',flush=True)
    return result

if __name__=='__main__':finalize()
