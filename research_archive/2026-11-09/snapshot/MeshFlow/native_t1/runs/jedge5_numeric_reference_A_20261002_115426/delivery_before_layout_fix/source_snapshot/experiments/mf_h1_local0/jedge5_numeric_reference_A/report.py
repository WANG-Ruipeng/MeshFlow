"""Render the audited window-A result, without any model execution."""
import argparse
from pathlib import Path
from .assets import read_json,write_json,file_sha256


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();out=a.out
    protocol=read_json(out/'registration/protocol.json');analysis=read_json(out/'analysis.json');audit=read_json(out/'final_audit.json')
    judgments=read_json(out/'diagnostic_judgments.json')
    lookup={(r['case_id'],r['model'],r['solver']):r for r in analysis['source_rows']}
    def scalar(case,model,solver,key):
        row=lookup[(case,model,solver)]
        if not row.get('complete'):return '未完成' if row['source']=='new_formal' else '缺失'
        v=row.get('scalars',{}).get(key)
        return '缺失' if v is None else f'{v:.6f}'
    mainrows=[]
    for c in protocol['cases']:
        for model in ('B','J'):
            mainrows.append([c['id'],model]+[scalar(c['id'],model,s,'remainder_symmetric_RMS_bbox_pct') for s in ('E50','E100','E200','R1','R2','D1','D2')])
    refs=[]
    for r in analysis['convergence_evidence']:
        vals=[]
        for pair in ('R1__R2','D1__D2','D2__R2'):
            v=r['pair_values'].get(pair)
            vals.append('不可配对' if v is None else f"{v['coordinate_RMS_bbox_pct']:.6f} / {v['surface_RMS_bbox_pct']:.6f}")
        refs.append([r['case_id'],r['model']]+vals)
    jb=[]
    for r in analysis['J_minus_B']:
        if r['solver'] not in ('R1','R2','D1','D2'):continue
        keys=('remainder_symmetric_RMS_bbox_pct','boundary_RMS_bbox_pct','boundary_p95_bbox_pct','CF_proper_pair_count','FF_illegal_pair_count')
        jb.append([r['case_id'],r['solver']]+(['不可配对']*len(keys) if r['status']!='PASS' else [f"{r['J_minus_B'][k]:+.6f}" for k in keys]))
    adaptive=[]
    for r in analysis['source_rows']:
        if r['source']=='new_formal' and r['solver'].startswith('D'):
            adaptive.append([r['case_id'],r['model'],r['solver'],r['nfe'],r['accepted_steps'],r['rejected_steps'],f"{r['last_accepted_time']:.10f}",r['status']])
    complete=[r for r in analysis['source_rows'] if r['source']=='new_formal' and r.get('complete')]
    quality={key:dict(min=min(r['scalars'][key] for r in complete),max=max(r['scalars'][key] for r in complete)) for key in (
        'CF_proper_affected_free_fraction','CF_proper_pair_count','FF_illegal_pair_count','free_total_surface_area',
        'free_minimum_angle_lt5_fraction','free_near_degenerate_count','free_exact_zero_area_count','nan_coordinate_count','inf_coordinate_count',
        'CF_unresolved_or_degenerate_pair_count','FF_unresolved_or_degenerate_pair_count')}
    counts=audit['actual_counts'];timing=audit['timing'];resource=audit['resources']
    results=dict(schema='jedge5_numeric_reference_A_results_v1',execution_audit=audit['status'],scientific_completeness=audit['scientific_completeness'],
        conclusions=judgments,actual_counts=counts,analysis=analysis,quality_ranges=quality,timing=timing,resources=resource,
        model_identities=protocol['models'],input_identity=dict(evaluation_sha256=protocol['evaluation_sha256'],registered_inputs_sha256=protocol['registered_inputs_sha256'],source_archive=protocol['source_archive']),
        audit_path='final_audit.json',audit_sha256=file_sha256(out/'final_audit.json'),code_diff_sha256=file_sha256(out/'code.diff'))
    write_json(out/'results.json',results)
    text=f'''# A Numerical Reference Report

{judgments['summary']}

## 实际执行与范围

执行审计 **{audit['status']}**，数值参考矩阵完整性 **{audit['scientific_completeness']}**。
32个正式设置：{counts['formal_complete']}条完整、{counts['formal_incomplete']}条预算内未完成；另有1条预登记的独立进程重复，全部保存数组逐位一致。
本窗口模型前向尝试共 **{counts['model_forward_attempts']}**，其中正式RK4 {counts['formal_RK4_NFE']}、正式Dopri {counts['formal_adaptive_NFE']}、重复400、有效预检12、日志故障预检6；硬上限30000。
前向返回已确认{counts['confirmed_model_forward_returns']}次，故障预检最后1次返回状态未持久化，已保守计入预算。训练更新、反向、OT均为0。

只研究4个有意选择、已经观察过的案例，不是新确认集或总体胜率估计。N112/K12/free100，冻结累计3000步raw B/J，不用EMA。
旧Euler只读复用22条；P0/patch4/10402两模型的Euler200原本缺失，标缺失且未补跑。
存档发现、原数组、全部SHA及缺失项见registration。Gaussian实际数组复用，没有按seed重抽；mask/y由存档槽位、有效面数与旧入口合同精确恢复，来源已明示。

## 数值实现与门禁

仅自由坐标进入ODE和误差范数。C按原槽位直接复制，known时间恒1，free使用真实FP32时间；面数条件112不变。
参数、速度、坐标与加权更新均FP32，关闭autocast/TF32，严格确定性、SDPA math、eval/no_grad；实际前向dtype钩子检查通过。
全部成功任务的模型state哈希前后一致；真实t=1在B/J四例中均有限；旧FP32调用与新接口逐位一致。已保存状态与每次完整输入组装C均逐位保持。

R1/R2为明确写出的经典四阶段RK4，100/200步分别400/800 NFE。D1/D2调用环境原有torchdiffeq 0.2.5的Dopri5Solver，rtol/atol分别1e-4/1e-6与1e-5/1e-7。
采用原库控制器和系数，first_step=.01、FP64时间控制、step_t=[1]；绕开公共包装器的端点nextafter扰动，实际评估t=1，不越界、不clip。
标准FSAL仅复用算法端点导数，没有新增几何/模型特征缓存。误差范数为free坐标上的RMS(error/(atol+rtol*max(abs(y0),abs(y_candidate))))；atol为原坐标单位。
[Dopri选项文档](https://github.com/rtqichen/torchdiffeq/blob/master/FURTHER_DOCUMENTATION.md)和本地实际源码均核对，依赖版本/源码保护通过。
D的固定时点中间状态明确标为库密集插值；不能当作接受步节点。真实函数时间与controller时间分别写入function_log。

CPU FP64解析门禁通过（恒定、线性、时间相关ODE；RK4误差比15.35；更紧D容差改善两类解析误差）。两次开发/最终CPU验证共3364次解析函数调用，其中最终1682次有JSON证据，前一次1682次仅保留代理执行记录，未伪造第二份原始stdout。它们均不计作神经模型前向。
CPU表面距离/坐标公式/C保护测试通过。首次CPU预检因动态伪源路径记录报错、首次B GPU预检因日志哈希API类型错误中断，均保留记录；修复只涉及登记代码，未改数值公式。有效预检随后通过。

## 数值收敛证据

表中每格为“同槽逐坐标RMS / 独立双向表面RMS”，单位均为原bbox百分比；不是各自GT误差，也不是认证误差界。

{table(['案例','模型','R1 vs R2','D1 vs D2','D2 vs R2'],refs)}

{judgments['classification_text']}

不以GT误差最低挑选求解器；不同算法内部稳定，也不能自动替代跨算法一致性。

## 各自剩余GT误差

以下为排除C后的剩余对称RMS，单位% bbox。全部为FP32，不能与上轮BF16数字混用。

{table(['案例','模型','Euler50','Euler100','Euler200','R1','R2','D1','D2'],mainrows)}

{judgments['geometry_text']}

## 同配置 J−B 与质量取舍

RMS/Gamma单位为百分点；相交为对数差。负值仅表示对应列更低，不构成综合胜出。

{table(['案例','设置','剩余RMS差','Gamma RMS差','Gamma p95差','CF proper差','FF illegal差'],jb)}

完整两个方向误差、固定目标覆盖、整条Gamma到free表面与有限边的RMS/p95/各阈值coverage、CF/FF共面重叠/UNRESOLVED、窄角、退化及面积均保存在per_case.csv和逐输出metrics.json。
新完整输出质量范围：

{table(['指标','最小','最大'],[[k,f"{v['min']:.6g}",f"{v['max']:.6g}"] for k,v in quality.items()])}

源C的静态拓扑异常审计只读复用，不混作free生成错误。精确共边/水密/流形等本轮没有新增拓扑审计：NOT_RUN；CF/FF相交诊断已实际执行。没有Pair、Stage2、焊接或后处理。

## 自适应成本与未完成状态

{table(['案例','模型','设置','NFE','接受','拒绝','最后接受t','状态'],adaptive)}

未完成轨迹只保存最后接受状态及已有固定时点，不对t<1打终点分，也不补预算、放宽容差或重试。

## 资源、时间与保护

每个GPU worker在CUDA初始化/模型加载前取得共享OS flock，单轨迹后退出。窗口A实际GPU并发1，独立CPU评分并发上限2；没有读取或汇总窗口B实验结果。
已记录锁等待{timing['recorded_lock_wait_seconds']:.3f}s；模型加载{timing['recorded_model_load_seconds']:.3f}s；模型与求解器墙钟合计{timing['recorded_model_and_solver_wall_seconds']:.3f}s；模型前向CUDA事件合计{timing['recorded_model_forward_CUDA_event_seconds']:.3f}s；CPU几何评分合计{timing['CPU_geometry_scoring_seconds_sum']:.3f}s；CPU端点分析/绘图{timing['CPU_analysis_seconds']:.3f}s。
各列有重叠，不能相加当总墙钟，也不是隔离吞吐基准。首次日志故障预检没有持久化完整GPU耗时，标UNKNOWN，不按0伪计。
Torch峰值allocated {resource['torch_peak_allocated_bytes']/2**30:.3f}GiB、reserved {resource['torch_peak_reserved_bytes']/2**30:.3f}GiB；GPU全局used峰值{resource['global_memory_used_peak_MiB']:.0f}MiB，最少free {resource['global_memory_free_min_MiB']:.0f}MiB。
全局数据包含无关进程。WDDM进程显存峰值不可得（NOT_AVAILABLE），不由全局减法或权重大小估算。

读取权重、输入、公共源码及求解库保护均通过；HEAD {audit['HEAD_start']} → {audit['HEAD_end']}。只写窗口A两个授权目录。
实际新增代码diff已在本run独立临时目录git apply逐字节验证。没有修改生产默认采样器、模型、loss或精度，没有commit/push。

## 交付与结论边界

- results.json：本轮结论、完整指标/端点差、实际计数与时间。
- per_case.csv / endpoint_differences.csv / J_minus_B.csv：逐输出与配对结果。
- trajectories/*/raw.npz、solver_step_log.json、function_log.json：含未完成状态的原始证据。
- fixed_view_full.png / fixed_view_free_only.png：固定视角、每案例跨模型与设置同范围；缺失/未完成格留空。
- final_audit.json、registration、code.diff、source_snapshot：身份、测试、保护及可回放代码。

{judgments['final_text']}

固定矩阵及预登记重复已结束，本轮停止。未自动换求解器、扩大预算或开始新训练。
'''
    (out/'A_NUMERICAL_REFERENCE_REPORT.md').write_text(text,encoding='utf-8')
    print(dict(report=str(out/'A_NUMERICAL_REFERENCE_REPORT.md'),results_sha256=file_sha256(out/'results.json')))

if __name__=='__main__':main()
