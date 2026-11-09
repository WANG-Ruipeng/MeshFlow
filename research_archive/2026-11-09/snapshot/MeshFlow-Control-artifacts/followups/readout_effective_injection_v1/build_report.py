from pathlib import Path
import csv,json,math,hashlib,statistics,datetime
r=Path('/mnt/e/MeshFlow-Control-artifacts/followups/readout_effective_injection_v1')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def avg(rs,k):return statistics.mean(float(x[k]) for x in rs)
layers=rows(r/'analysis/layer_metrics.csv');pairs=rows(r/'analysis/prediction_pairs.csv')
for x in layers:
    x['residual_to_hidden_l2']=float(x['residual_l2'])/float(x['before_l2'])
    x['effective_to_hidden_l2']=float(x['effective_l2'])/float(x['before_l2'])
    x['projection_onto_residual']=float(x['effective_over_residual_l2'])*float(x['effective_residual_cosine'])
fields=['label','request_id','panel','step','t','layer','residual_to_hidden_l2','effective_to_hidden_l2','projection_onto_residual']
with (r/'derived_layer_ratios.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(layers)
status=read(r/'status.json');plan=read(r/'plan.json');summary=read(r/'analysis/summary.json')
occ=[json.loads(x) for x in (r/'budget/gpu_occupancy.jsonl').read_text().splitlines()]
resource=[json.loads(x) for x in (r/'resource_monitor.jsonl').read_text().splitlines()]
used=[];free=[];util=[];process=[]
for x in resource:
    if 'gpu' in x:
        row=x['gpu']['stdout'].strip().splitlines()[1].split(',')
        used.append(int(row[2].strip().split()[0]));free.append(int(row[3].strip().split()[0]));util.append(int(row[4].strip().split()[0]))
        for line in x['gpu_processes']['stdout'].splitlines()[1:]:
            try:process.append(int(line.split(',')[-1].strip().split()[0]))
            except ValueError:pass
resource_facts=dict(samples=len(used),device_used_mib_max=max(used),device_free_mib_min=min(free),device_utilization_percent_max=max(util),observed_process_used_mib_max=max(process) if process else None,gpu_owner_seconds=sum(x['seconds'] for x in occ if x['event']=='END'),wall_seconds=status['wall_seconds'],cpu_analysis_seconds=summary['wall_seconds'],cpu_threads_forward=4,cpu_threads_analysis=2)
(r/'resource_summary.json').write_text(json.dumps(resource_facts,indent=2)+'\n')
s=['# READOUT 有效注入与同模型开关诊断','', '日期：2026-10-07。状态：48/48 前向完成，96/96 层记录、24/24 ON/OFF 配对有效；无训练、无新噪声、无完整生成、无几何重新评分。','',
'## 结论','',
'反馈提出的数值疑点得到部分证实：残差没有在转 BF16 时变成零，但加到 BF16 hidden 后，大量非零元素没有留下可表示的变化，深层尤其明显。与此同时，全部 24 个同状态开关对照都改变了 free 预测。因此，“分支完全不起作用”已被排除；“BF16 是效果不佳的主要原因”仍未建立。','',
'当前证据更准确的表述是：通路已经产生可测的即时作用，作用幅度较小，且实际加回受到明显的量化影响；尚未证明它承担了有益的条件推理。前一轮 B/C 未显示一致优势的结论保持。','',
'## 固定面板与执行定义','',
'- B1000、C1000：各自原有 ON 轨迹。每个权重取下列 4 个请求的 step 0/25/45，即 t=0/0.5/0.9；每状态各运行 ON、OFF 一次，共 48 次。',
'- P1 取父 ID 排序首尾各一个，使用既有 n4（原评价中的 NEW 组，不是新抽样）；P2 取两面座板、四面座板加背板的既有 n0。选样在前向前冻结，不按本轮结果挑选。',
'- ON 调用原 apply_readout，仅用只读 hook 记录真实输入、FP32 residual、实际 cast 和实际返回 hidden。OFF 在相同权重、状态下同时跳过 3/6/9/12 四个残差加回。',
'- OFF 不等于 A，也不是完整 OFF 轨迹；B/C 在 t>0 使用各自状态，不能当作同状态的跨模型因果对照。',
'- 隔离的一步严格执行原 FP32 Euler 表达式 x + velocity.float() * (1/50)，并原样写回 C。没有继续积分生成网格。','',
'| 请求 | N | K | 备注 |','|---|---:|---:|---|']
for q in plan['requests']:s.append(f"| {q['request_id']} | {q['N']} | {q['K']} | {q['panel']} |")
s+=['','## 实际残差加回','',
'定义 r 为分支 FP32 输出，c 为实际转成 BF16 后的残差，e=float(h_after)−float(h_before)。吞掉比例指 r≠0 但 e=0 的标量元素占比，**不是信息丢失率，也不是整个角/面完全不变的比例**。下表每格是 4 请求 × 3 时刻、共 12 个记录的等权均值。','',
'| 层 | B 吞掉非零元素 | C 吞掉非零元素 | B ‖e‖/‖r‖ | C ‖e‖/‖r‖ | B cos(e,r) | C cos(e,r) |','|---|---:|---:|---:|---:|---:|---:|']
for layer in (3,6,9,12):
    b=[x for x in layers if x['label']=='B1000' and int(x['layer'])==layer];c=[x for x in layers if x['label']=='C1000' and int(x['layer'])==layer]
    s.append(f"| {layer} | {avg(b,'residual_nonzero_swallowed_fraction'):.2%} | {avg(c,'residual_nonzero_swallowed_fraction'):.2%} | {avg(b,'effective_over_residual_l2'):.3f} | {avg(c,'effective_over_residual_l2'):.3f} | {avg(b,'effective_residual_cosine'):.3f} | {avg(c,'effective_residual_cosine'):.3f} |")
s+=['',
'96 个层记录中，非零 r 转成 BF16 后归零的比例全部为 0；cast 相对 L2 误差均值约 0.166%。主要量化差异出现在加法保存处。96 个层记录的 e 范数全部非零。',
'',
'范数比不能称为“保留百分比”：个别第 3 层记录超过 1，因为 BF16 舍入跨档可能放大部分增量。第 12 层虽然约 97%–98% 的标量元素不变，B 的每个 free corner 仍有至少一个 hidden 通道改变；C 的该比例均值为 99.97%。这与“97% 的面完全没收到信息”完全不同。',
'',
'第 3 层保留的范数相对较高，第 12 层明显衰减且方向偏离更大；不能把最后一层的数值概括为整条通路失效。这里未做逐层关闭，不能分摊每一层对最终输出的贡献。',
'',
'本轮是 checkpoint1000 的既有 Euler 状态，原报告的 residual/hidden 是训练批次的记录。二者的输入/时刻不同，不能直接把数值差解释成权重发生变化。逐记录 residual/hidden、effective/hidden 和残差方向投影见 derived_layer_ratios.csv。','',
'## 同模型 ON/OFF 预测差','',
'仅统计 free 槽。相对预测差 = ‖v_ON−v_OFF‖₂/‖v_ON‖₂；每格为 4 请求等权均值。下一步 RMS 的单位是原模型坐标单位，不是毫米，也不是相容性或最终质量指标。','',
'| t | B 相对预测差 | C 相对预测差 | B 单步坐标差 RMS | C 单步坐标差 RMS |','|---|---:|---:|---:|---:|']
for step in (0,25,45):
    b=[x for x in pairs if x['label']=='B1000' and int(x['step'])==step];c=[x for x in pairs if x['label']=='C1000' and int(x['step'])==step]
    s.append(f"| {step/50:g} | {avg(b,'velocity_relative_l2_on'):.4%} | {avg(c,'velocity_relative_l2_on'):.4%} | {avg(b,'next_state_delta_rms'):.6g} | {avg(c,'next_state_delta_rms'):.6g} |")
v=[float(x['velocity_relative_l2_on']) for x in pairs];coord=[float(x['velocity_changed_coordinate_fraction']) for x in pairs];faces=[float(x['velocity_changed_face_fraction']) for x in pairs]
s+=['',f'24 对的相对预测差范围为 {min(v):.4%}–{max(v):.4%}。发生任意数值变化的 free 标量坐标占 {min(coord):.2%}–{max(coord):.2%}，free 面占 {min(faces):.2%}–{max(faces):.2%}；数量多并不等于改变量大。24 对预测和隔离单步坐标差均非零。', '',
'这些结果说明当前 readout 对最终数值预测有直接影响，排除了“实际上完全被吞掉”的解释。约百分之一以内的速度变化不能直接换算成最终网格变化；没有完整 OFF rollout、质量评分或无语义扰动对照，无法区分有益控制、无益扰动和多步累积作用。也不能据此宣布新通路已被充分利用。','',
'## 优先下一步与可检验假说','',
'优先做一次固定权重、固定状态的精度路径对照，先验证“加法量化是否真是功能瓶颈”，不直接放大残差或重训。待检验假说是：若低精度加回限制了通路作用，那么在能把残差传递至后续运算的精度路径中，ON/OFF 的分支增量应更稳定地保留，并在最终 free 预测中显现；若只增加 hidden 改变量、最终预测增量仍很小，则数值吞掉并非唯一或主要瓶颈。',
'',
'这个后续对照必须同时包含原精度 ON/OFF 与候选精度 ON/OFF，区分精度本身对主干的改变与 readout 增量；零残差时应保留明确的精度基线。不应把 h.float()+r 后立即转回 BF16 当作已修复。任何候选改动都需独立目录和显式预算。本轮没有执行该后续，也未修改精度、loss、默认模型或训练组织。','',
'## 执行、验证与限制','',
'- 24/24 ON 隔离下一步与已有轨迹字节一致；ON/OFF 的输入状态逐字节一致；C 下一步字节保持。原 ON 路径未被探针改变。',
'- 48 个调用分别有 RESERVE/RETURN/COMMIT；没有重试、反向、优化器更新、新高斯、完整采样或重新评分。B/C 权重状态 hash 与调用期间 RNG hash 前后一致。',
'- 原源码/检查点/输入/既有结果及主报告 hash 保持。原 E:\\MeshFlow Git HEAD 为 ae91eaa6f882a7d311233fa32053b7cd5de7ebe0，原有三个未跟踪项未变；无 commit/push。',
'- RTX 5080，单 GPU worker，前向 CPU 4 线程、统计 CPU 2 线程。启动前检查显存/进程/CPU/RAM，CPU 代码复核与实现并行。两模型顺序加载，共享锁避免加载尖峰；未停止其他进程。',
f"- GPU owner {resource_facts['gpu_owner_seconds']:.3f} 秒；诊断主体总计 {status['wall_seconds']:.3f} 秒；CPU 汇总 {summary['wall_seconds']:.3f} 秒。监控 {resource_facts['samples']} 次，设备总 used 峰值 {resource_facts['device_used_mib_max']} MiB、free 最低 {resource_facts['device_free_mib_min']} MiB。设备总量包括其他应用，不等于本进程分配。",
'- PyTorch 两阶段峰值分别见 B1000_integrity.json/C1000_integrity.json；计数器在加载后重置，不能当作训练显存估计。探针包含 CPU 拷贝/压缩，计时不是推理性能基准。',
f"- 捕获 NPZ 合计 {status['compressed_arrays_bytes']/1024**2:.2f} MiB，保留用于完全离线复算；小表与代码可单独查看。",
'- 只有 4 个确定性选择请求、3 个时刻；不是整个 8 父分布的代表性评估。没有独立训练重复，没有逐层消融，没有语义/质量判定。',
'- 预检发现账本函数形参重名，已在 0 次模型调用时修正并保留 plan.initial.json/preflight_revision.json。运行中未更改冻结脚本。完成后的恢复只使用纯 CPU 验证/统计入口，不重新启动模型。','',
'## 证据入口','',
'- [96 条逐层结果](analysis/layer_metrics.csv)、[24 条逐状态预测配对](analysis/prediction_pairs.csv)、[48 条调用检查](analysis/cell_audit.csv)。',
'- [分层汇总](analysis/layer_label_layer.csv)、[分时预测汇总](analysis/prediction_label_step.csv)、[精确指标定义](analysis/metric_definitions.json)、[CPU 汇总及输入 hash](analysis/summary.json)。',
'- [额外范数比与方向投影](derived_layer_ratios.csv)、[冻结计划及全部依赖 hash](plan.json)、[调用账本](budget/calls.jsonl)、[GPU owner 账本](budget/gpu_occupancy.jsonl)。',
'- [实际执行脚本](run_check.py)、[纯 CPU 分析代码](analyze.py)、[只读完成验证代码](verify_completed.py)、[完成验证结果](final_verification.json)。',
'- cells/ 内含 48 对 JSON+NPZ；开启分支的 NPZ 包含四层真实 before/residual/cast/after。可以只从这些文件复算，不加载模型。','',
'离线复核命令（既有 WSL venv；两者均零模型调用）：','',
'```text',
'/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B /mnt/e/MeshFlow-Control-artifacts/followups/readout_effective_injection_v1/verify_completed.py',
'/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B /mnt/e/MeshFlow-Control-artifacts/followups/readout_effective_injection_v1/analyze.py --root /mnt/e/MeshFlow-Control-artifacts/followups/readout_effective_injection_v1',
'```','']
(r/'READOUT_EFFECTIVE_INJECTION_REPORT.md').write_text('\n'.join(s),encoding='utf-8')
print(json.dumps(dict(report=str(r/'READOUT_EFFECTIVE_INJECTION_REPORT.md'),resources=resource_facts,velocity_relative_range=v and [min(v),max(v)],layers=len(layers),pairs=len(pairs))))