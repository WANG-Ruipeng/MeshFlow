"""Summarize existing diagnosis evidence, no torch/model imports or execution."""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
ROOT=Path("/mnt/e/MeshFlow")
RUN=ROOT/"native_t1/runs/geometry_gate_code_audit_20260930_150400"
BASE="2502d5f89c54706cc4700d3cdea4a12fd6792529"
result=json.loads((RUN/"results.json").read_text())
model=(ROOT/"native_t1/model.py").read_text(encoding="utf-8-sig")
old=subprocess.check_output(["git","show",BASE+":native_t1/model.py"],cwd=ROOT).decode("utf-8-sig").replace("\r\n","\n")
added_init='        # Plain None adds no state_dict key and leaves the original arithmetic untouched.\n        self.context_encoder = None\n        self.context_observer = None\n'
added_forward='        if self.context_encoder is not None:\n            from .context_geometry import inject_context\n            delta = self.context_encoder(x, known_mask, valid_mask, observer=self.context_observer)\n            hidden = inject_context(hidden, delta, known_mask, observer=self.context_observer)\n'
if model.count(added_init)!=1 or model.count(added_forward)!=1:raise RuntimeError("Unexpected model delta")
recovered=model.replace(added_init,"").replace(added_forward,"")
if recovered!=old:raise RuntimeError("Disabled path differs beyond documented additions")
normalized={}
for name in ("models/equidit.py","models/attention.py","models/utils.py","native_t1/data.py","native_t1/runtime.py"):
    working=(ROOT/name).read_text(encoding="utf-8-sig")
    stored=subprocess.check_output(["git","show",BASE+":"+name],cwd=ROOT).decode("utf-8-sig").replace("\r\n","\n")
    normalized[name]=working==stored
if not all(normalized.values()):raise RuntimeError("Protected source differs")
review=dict(status="PASS",default_path_after_removing_only_seven_added_lines_matches_base=True,
    canonical_original_model_sha256=hashlib.sha256(old.encode()).hexdigest(),normalized_protected_source_matches=normalized,
    properties=dict(encoder_disabled=True,no_slot_position_embedding=True,known_and_valid_masks_permuted=True,
                    roles_selected_from_known_mask=True,known_t_is_one=True,free_t_unchanged=True,
                    arbitrary_S3_equivariance_in_real_arithmetic=True),
    caution="Order-dependent floating-point reductions can violate finite-tolerance permutation checks while fixed-input repetition remains bitwise stable. No particular internal operator root cause has been proven.",
    historical_gate=dict(source="experiments/mf_h1_local0/native_patch_run.py:159-168",
       input="Official EMA with zero roles; B3 mixed K4/8/12, free-only OT training fixture",
       face_transform="Reverse112 only",corner_transform="Uniform [2,0,1] only, separate forward",
       recorded_face_max_abs=.03125,recorded_face_RMS=.004135464783757925,recorded_cycle_max_abs=0),
    current_gate=dict(input="A_continue1500 with learned roles; B1 P0K12 t=.5, no OT, CPU Gaussian seed12345",
       transform="seed24680 random112 face permutation combined with per-face all-S3"),
    implementation_issue="The prior geometry pilot reused tolerance values but expanded the whole-backbone test transform instead of reusing the historical numerical test exactly.",
    model_forwards=0,model_backwards=0,optimizer_updates=0,OT_calls=0)
(RUN/"source_review.json").write_text(json.dumps(review,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
with np.load(RUN/"hidden_trace.npz",allow_pickle=False) as trace:
    traces=[]
    for comparison in result["traces"]["current_combined"]["permutation_comparisons"]:
        site=comparison["site"]
        ref=trace["current_reference_observed__"+site]
        rms=float(np.sqrt(np.mean(ref.astype(np.float64)**2)))
        traces.append(dict(site=site,reference_hidden_RMS=rms,difference_RMS=comparison["RMS"],
                           difference_to_reference_RMS=comparison["RMS"]/rms if rms else None,
                           bitwise=comparison["bitwise"],max_abs=comparison["max_abs"]))
(RUN/"hidden_trace_summary.json").write_text(json.dumps(traces,indent=2)+"\n",encoding="utf-8")
counts=result["counts"]
independent_path=RUN/"independent_output_audit.json"
independent=json.loads(independent_path.read_text(encoding="utf-8-sig")) if independent_path.exists() else {"status":"PENDING"}
lines=["# CODE_PATH_DIAGNOSIS_REPORT","","本轮是用户要求的代码排查，未恢复几何编码实验。新增9次冻结主干BF16单步前向；0反向、0训练更新、0完整采样。原报告和失败证据保持不变。","",
"## 当前结论","",
"在本次固定输入上，证据指向现有纯T1对面排列的数值敏感性。没有发现这次几何编码接入改变默认T1输出、mask/time/role错位或输出逆排列错误的证据。",
"另确认上一轮测试实现存在口径偏差：沿用了历史容差，却把历史分开的倒序面测试与统一循环角测试改成了随机面序＋逐面全S3组合硬门禁。该覆盖扩大不应表述为原样沿用旧测试。",
"但测试口径偏差不是当前失败的充分解释：在同一A1500权重和固定诊断输入上，恢复历史的面倒序变换仍有12/1008个分量越阈。不能据此把原失败改记PASS。","",
"## 原版代码与当前代码直接对照","",
"原版模型文件从本地基准提交2502d5f读取，在独立诊断脚本中加载。两份wrapper共享同一个只读backbone，role参数逐位复制；两者完整state hash一致。未改生产模块、未替换官方代码、未另建随机模型。","",
"| 输入 | 当前版本 vs 基准提交原版 | 最大差 |",
"| --- | --- | --- |",
"| 原固定输入 | 逐位一致 | 0 |",
"| 随机面＋S3组合置换输入 | 逐位一致 | 0 |","",
"当前文件去掉默认关闭入口的7行增量后，与基准提交逐字一致；受保护的官方模块、数据与运行配置在统一换行表示后也与基准一致。源码证据见source_review.json。",
"这直接排除了已测试路径上的新增入口回归；不代表新增编码器本身的GPU路径已经通过验证。","",
"## 将两类置换拆开","",
"下表均使用同一A_continue累计1500模型、同一P0/K12、t=0.5、相同原Gaussian，以及原严格确定性/BF16/TF32关闭/SDPA math配置。atol=0.015625、rtol=0.03125保持不变。‘历史方式’仅指变换方式相同，不是复现历史EMA/K混合夹具。","",
"| 变换 | 最大绝对差 | RMS | 越阈分量/1008 | 原门槛 |",
"| --- | --- | --- | --- | --- |"]
for label,name in [("随机面重排，角不变","current_face_only"),("逐面S3，面不变","current_corners_only"),
                   ("随机面＋逐面S3","current_combined"),("历史方式：面倒序","current_historical_reverse"),
                   ("历史方式：统一循环角","current_historical_cycle")]:
    d=result["comparisons"][name]
    lines.append(f"| {label} | {d['max_abs']:.12g} | {d['RMS']:.12g} | {d['violating_elements']} | {'PASS' if d['allclose'] else 'FAIL'} |")
lines+=["","随机面重排单独就复现了原组合失败的误差；单独S3及统一循环角在所有观测层和最终输出均逐位等变。本例不能归咎于反向绕序或训练仅做循环角增强。",
"新进程原输入、组合输入均逐位复现上一轮保存数组；观测器开/关输出逐位一致。因此这不是本例同输入运行不稳定。","",
"## 差异首次出现的位置","",
"坐标嵌入＋role，以及进入第1层的hidden，在恢复排列后均逐位相同；进入第2层时首次不同，即差异发生在第1层内部。后续层放大/传播该差异。","",
"| 观测位置 | 原hidden RMS | 差值RMS | 相对RMS | 最大绝对差 |",
"| --- | --- | --- | --- | --- |"]
for d in traces:
    if d["site"] in ("embedding","block_1","block_2","block_3","block_6","block_12","final"):
        lines.append(f"| {d['site']} | {d['reference_hidden_RMS']:.9g} | {d['difference_RMS']:.9g} | {d['difference_to_reference_RMS']:.9g} | {d['max_abs']:.9g} |")
lines+=["","hidden的尺度不同于最终速度；此表仅用于定位，不把速度容差当作各层hidden的验收标准。所有已记录的role/逐面时间选择检查均通过。",
"第一层内部尚未做子算子重放。源码上，面排列会改变attention的键顺序及softmax/加权求和的浮点归约顺序；这是待核实的候选机制，当前不能宣布已证明具体是SDPA、某个GEMM或某一步舍入。",
"XEmbedder返回的首次三角mean未被Native使用，不能把它误认为本例误差传播源。实际使用的逐层三角mean位于native_t1/model.py；本例单独角排列完全一致，也不支持将当前差异归因于角均值。","",
"## 保护、执行账本与后续边界","",
f"- 本轮完成主干前向：{counts['model_forward_returns']}；反向、optimizer更新、完整生成、OT均为0。加上上一轮3次失败门禁前向，累计12次。",
f"- 本轮记录耗时：{result['seconds']:.6f}秒（包括加载、诊断、CPU观测和证据保存，不是纯GPU核函数耗时）。",
f"- 本轮PyTorch峰值allocated显存：{result['peak_cuda_allocated']} bytes；不能补填为上一轮未记录的峰值。",
f"- 前后模型state SHA256：{result['state_after']}，完全一致。",
f"- 起点checkpoint文件SHA256：{result['initialization']['checkpoint_sha256']}。",
"- 生产源码、运行配置、原失败报告/门禁、旧模型和数据保持不变；没有checkpoint保存、Git提交或推送。",
"- 本轮有限9次前向预算已完成。不修改容差，不换精度，不自动恢复三臂训练。",
"下一步如继续，应先对第一层attention及相邻运算做有界数值对账，区分实现错误与有限精度误差，再讨论主干置换门禁的适用性。几何特征/图的S3不变性、零出口回归及重复一致性要求继续保留。","",
"## 证据文件","",
"- [results.json](results.json)：9次实际前向、比较、运行配置与参数哈希。",
"- [source_review.json](source_review.json)：源码等价及历史测试口径对照。",
"- [independent_output_audit.json](independent_output_audit.json)：独立CPU输出对账。",
"- [velocities.npz](velocities.npz)：所有9个单步速度输出。",
"- [hidden_trace.npz](hidden_trace.npz)：只读观测到的原输入/组合输入hidden。",
"- [registration.json](registration.json)：执行前固定的9次预算、输入及源码身份。",""]
(RUN/"CODE_PATH_DIAGNOSIS_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
print(json.dumps(dict(report=str(RUN/"CODE_PATH_DIAGNOSIS_REPORT.md"),source_review=review["status"],
                      diagnostic_status=result["status"],independent_audit=independent.get("status"),
                      model_calls_this_report=0),ensure_ascii=False))
