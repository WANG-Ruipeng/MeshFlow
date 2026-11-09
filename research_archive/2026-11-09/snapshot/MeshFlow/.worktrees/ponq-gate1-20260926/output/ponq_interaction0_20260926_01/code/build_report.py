"""Build reports from measured artifacts. No new extraction or metric execution."""
from common import *
import pandas as pd, base64, html, importlib.metadata
r=ROOT
df=pd.read_csv(r/'LEDGER.csv');active=df[df.status=='EXTRACTED'];su=json.loads((r/'SUMMARY.json').read_text())
names={'sphere':'球体','box':'盒子','thin_u':'闭合薄壁 U','torus':'环面'}
state_names={'clean128':'干净128','clean256':'干净256','position_light':'位置2%','position_strong':'位置10%','normal_light':'法向2°','normal_strong':'法向5°','qem_light':'QEM因子2%','qem_strong':'QEM因子10%','joint_light':'联合轻度'}
def row(obj,state):return active[(active.object==obj)&(active.state==state)].iloc[0]
def rms(x):return '<1e−10' if x<1e-10 else f'{x:.5f}'
clean=['|对象|128元素：实际面数|128元素RMS (%)|256元素：实际面数|256元素RMS (%)|256元素p95 (%)|256元素法向绝对角误差均值 (°)|表面Betti|','|---|---:|---:|---:|---:|---:|---:|---|']
for o,n in names.items():
 a,b=row(o,'clean128'),row(o,'clean256');clean.append(f'|{n}|{int(a.faces)}|{rms(a.rms_pct_bboxdiag)}|{int(b.faces)}|{rms(b.rms_pct_bboxdiag)}|{rms(b.p95_pct_bboxdiag)}|{b.normal_abs_mean_deg:.3f}|({int(b.beta0)},{int(b.beta1)},{int(b.beta2)})|')
pert=['|对象|干净256|位置2%|位置10%|法向2°|法向5°|Q因子2%|Q因子10%|联合轻度|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for o,n in names.items():pert.append('|'+n+'|'+'|'.join(rms(row(o,s).rms_pct_bboxdiag) for s in ['clean256','position_light','position_strong','normal_light','normal_strong','qem_light','qem_strong','joint_light'])+'|')
amp=['|对象|2% Q因子：平均 Δv/h|2% Q因子：最大 Δv/h|10% Q因子：平均 Δv/h|10% Q因子：最大 Δv/h|','|---|---:|---:|---:|---:|']
for o,n in names.items():
 a,b=row(o,'qem_light'),row(o,'qem_strong');amp.append(f'|{n}|{100*a.position_shift_over_h_mean:.2f}%|{100*a.position_shift_over_h_max:.2f}%|{100*b.position_shift_over_h_mean:.2f}%|{100*b.position_shift_over_h_max:.2f}%|')
assets=[
 {'file_id':'file_00000000e3a08230838afb420ec1e631','filename':'MESHFLOW_BOTTLE_ORIGINAL_225TRAIN_24TEST.zip'},
 {'file_id':'file_000000006c4c822f8e999e8da8a65dd5','filename':'A_BOTTLE_DATA_INTAKE_20260923_RESULTS.zip'},
 {'file_id':'file_00000000cae081f896cdf09968eaa97d','filename':'A_LOUDSPEAKER_CLEAN16_DIAGNOSTIC_20260923.zip'},
 {'file_id':'file_00000000ff2481f8bd340e4f2e87ee57','filename':'A_CATEGORY_SCREEN_20260923_RESULTS.zip'},
 {'file_id':'file_00000000d614822f9cddfdb2f643cab5','filename':'A_F512_CONTINUE_20260924_REPRO.zip'}]
for a in assets:a.update(status='MATERIALIZATION_FAILED',tool_error='This Project file does not have an authorized raw-byte materialization path.')
dump(r/'ASSET_ACCESS.json',dict(archives=assets,missing_object_slots=4,missing_extraction_slots=36,substituted_objects=0,local_object_ids_selected=False,notes='Search/list/report text succeeded, raw-byte materialization failed. No original asset coordinates were available. No synthetic proxy substituted for any local object.'))
report=f'''# PoNQ 无训练表示筛查：Gate 1 部分执行报告

日期：2026-09-25。判定：**PARTIAL_GATE1_SYNTHETIC_EVIDENCE_ONLY_NO_TRAINING_PROMOTION**。

## 1. 本轮真正完成的范围

实际执行四个合成对象，共 **36 次官方提取器调用 = 8 次干净重建 + 28 次扰动重建**。全部调用返回网格、没有异常或重试。另四个本地对象的36个计划槽位没有运行，明确标为 `NOT_RUN_MISSING_RAW_ASSET`，不是零误差或成功。没有网络、没有训练、没有进入第二或第三关。

Library中找到了历史音箱和瓶子归档及相关报告，但五个原始字节取用尝试全部返回“没有授权的原始字节物化路径”。因此没有选择或重新审核两个真实瓶子，也没有用合成瓶子/音箱冒充本地资产。`ASSET_ACCESS.json` 保存精确文件标识与错误。历史文本核对显示原瓶子结构候选76、径向候选15；差值61并不代表已经通过精确几何审核。原16音箱的源网格在单独的音箱诊断归档中，不能拿其中径向缓存作为本轮参考。

**本报告不是完整八对象72次实验，也不是“第一关已通过”。**

## 2. 可引用的新观察

四对象两预算的8个干净输出均为闭合、定向二维流形，表面Betti与参考一致；环面孔道保留。16个轻度扰动输出（位置2%、法向2°、QEM因子2%、联合轻度，四对象）同样没有拓扑变化，但QEM与联合扰动已产生明显几何误差。

强QEM扰动下，薄壁U发生结构损坏：2个连通分量、1个非流形顶点，表面Betti从(1,0,1)变为(2,0,3)，RMS从0.07171%升至2.48531%。其余35个输出拓扑匹配。所有36个输出均未检出非法三角形相交或精确退化。这正说明“无自交、无边界边”不等于“二维流形或正确拓扑”。

本轮采用的是完整齐次4×4 QEM因子噪声。其扰动会改变解算位置，不能解释成固定位置下纯粹的各向异性度量噪声，也不能直接视为未来网络的误差分布。

## 3. 干净参数重建

所有距离均为**参考网格包围盒对角线的百分比**。两方向各8192个独立于拟合的面积样本，求到真实三角面的最近距离，不是点到点Chamfer。RMS使用两方向均方距离等权平均后开方；p95为两方向样本合并分位数，不是Hausdorff上界。法向主列取绝对点积角，账本同时保留定向法向误差。

{chr(10).join(clean)}

盒子源网格只有12面，U源网格28面；此实验不是网格压缩，干净盒子误差接近浮点舍入量不构成优于传统简化的证据。U的256元素p95接近零而RMS非零，是大多数面积与原平面重合、误差集中于少量局部区域的表现；不能因此忽略局部缺失。环面表面β₁=2代表一个把手，不是两个孔道。这里计算的是**曲面复形Betti**，不是实体内部Betti。

## 4. 所有扰动的RMS

{chr(10).join(pert)}

正常量旋转确实施加到了提取器输入法向。球、盒、U在2°和5°下的输出几何保持不变，说明本轮离散提取对这些扰动未切换输出，不是没有实际修改参数；角度与模长核验见逐次 `perturbation_audit.json` 和后验一致性检查。环面法向扰动改变少量三角连接，差异已计入指标。

上述数据只有一组对象级固定种子，曲线连线仅帮助阅读三个剂量点，不是拟合规律、置信区间或全域鲁棒保证。

## 5. 重要诊断：QEM噪声与位置漂移耦合

定义 hᵢ 为K256干净v*到其他四个最近v*的平均间距。QEM扰动后仍以同一hᵢ计算实际位移：

{chr(10).join(amp)}

薄壁U的2%因子噪声，已诱发平均11.21%局部间距的位移；强档平均56.55%，最大154.82%。环面强档最大位移甚至超过两个局部间距。因此本轮不能归纳为“只有2%几何误差就会失败”，也不能直接据此否定QEM表示。需要区分被回归参数的误差、解算器对误差的放大、以及提取拓扑对有效位置的响应。

强U输入仍是256个元素，但实际使用的输出顶点从256降至162，面数从508降至314。没有删除失败样本、过滤小分量或追加重试。顶点62的link经独立NetworkX检查为两个分离的环：这是明确的非流形顶点见证，见 `audit/THIN_U_PINCH_WITNESS.json`。

## 6. 实现与冻结协议

### 官方提取核心

固定官方 `nissmar/PoNQ` 提交 `dde842b2c00b4a96f67ad07343ae3d48b43fe8ea`。`vendor/PoNQ_to_mesh.py` 10399字节，与官方Git blob `0fd392adf0a88555a5ae71a1dcb30c3e98033aab` 完全一致；本轮未修改评分、Delaunay、min-cut或构面规则。底层使用CPU、float64，`grid_scale=32`，`correct_tet_color=True`，保留8个外包角点，不加扰动jitter。36输出均未包含外包角点，记录中没有非有限或负min-cut容量。

为了移除本轮不用的依赖，单独摘取官方 `SIGNS`、`tet_circumcenter` 和 `vstars_from_quadrics` 函数。没有安装PyTorch3D、torch_scatter、igl，也没有假称重放官方优化notebook。官方许可保留在 `vendor/LICENSE`。源码身份见 `SOURCE_PROVENANCE.json`。

### 几何编码

使用自写确定性CPU几何拟合器：每对象65536个面积样本，FPS初始化，固定80次双向Chamfer交替分配/二次更新，随后按Voronoi分区聚合面法向与齐次QEM。v*由官方截断SVD函数求出，相对阈值0.05，保留其以P为基点的低秩回退行为。**这是一个实际可运行的干净拟合点，不是证明的最优“表示上限”。**不能将未达到更高质量归因于表示本身，也没有在结果出来后逐对象追加优化。

球体为subdivision4的icosphere，盒子长宽高1.4/1/0.8，U总宽2、壁厚0.14、闭合挤出厚0.25，环面R=1、r=0.32，96×32周期剖分。四者均只做bbox居中及对角线归一化为2。参考始终为这些原始三角面，不使用A的径向主场。源网格也经过独立精确面交与退化审计。

### 参数扰动定义

位置臂直接修改提取器接收的v*，位移长度严格为0.02hᵢ或0.10hᵢ，方向为冻结随机方向。为避免人为制造位置—平面不一致，同步对完整Q做平移合同变换，A不变。此臂不是“所有Q数组保持不变”，也不是只修改原采样点P；操作发生在实际提取器接口层。

法向臂绕随机垂直轴旋转固定2°或5°，保留每元素原模长。

QEM臂计算完整4×4齐次Q的主PSD平方根F，冻结同一方向G，令F'=F+α||F||_F G/||G||_F，Q'=F'F'^T，再以冻结P重算v*与特征值。没有直接向Q元素加噪。用于平方根的微小负特征值按0处理；噪声后最小特征值、矩阵相对变化、截断秩变化均有记录。此过程可以改变常数残差项，且不同于未来采用“独立v*＋3×3 A因子”的参数化；不能直接迁移噪声容限。

联合臂按Q因子2%、位置2% h_clean、法向2°顺序执行。所有强弱档共享相应随机方向，无重抽。

`FROZEN_PROTOCOL.json` 在首次提取前保存，后验哈希复核通过。程序拒绝在同一目录隐式覆盖原协议或重跑。

## 7. 几何审核和独立检查

最终OBJ使用17位有效数字写出，逐项重读后与原输出坐标/面数组逐位相同。精确审核把最终binary64坐标转成公共尺度的二进整数，使用任意精度整数谓词及Fraction有理交点；共面使用有理裁剪。包括相邻三角形，排除的仅为合法共享索引顶点/边；额外交叠、非索引接触均计入非法面交。

这是本轮自研审核器，不是CGAL复现，也不是对任意参数的形式证书。11个定向测试与100个随机共面三角对的独立Shapely交叉检查共111项通过。度量候选搜索另与全三角暴力最近距离对照，四对象120查询零超容差分歧。36输出的导出、哈希、Euler、边闭合、扰动剂量等后验检查共188项全部通过。

结果：36/36零非法面交，36/36零精确退化，36/36零边界边，36/36零非流形边，**35/36零非流形顶点并为闭合定向二维流形**。没有从有问题的U结果中移除非流形顶点再计算指标。

## 8. 成本

环境4核配额/4GiB、无GPU；主要数值线程设为1。这是一次小样本测量，不是与其它方法的公平性能基准。

8次几何拟合共{su['fitting_seconds_total']:.3f}s。36次提取调用合计{su['extraction_seconds_total']:.3f}s，中位{1000*su['extraction_seconds_median']:.3f}ms，范围{1000*su['extraction_seconds_min']:.3f}–{1000*su['extraction_seconds_max']:.3f}ms。提取计时不含拟合、参数扰动、文件IO、指标、自交审核或图像渲染。逐项完整账本有提取/度量/精确审核耗时；这些不能称为噪声到生成网格的时延。

## 9. 判定与下一步边界

已得到的正证据是：在四个固定合成对象上，低预算QEM集合能经官方提取器形成拓扑正确的干净网格，轻度扰动没有导致本轮拓扑突变。已得到的负证据是：完整Q因子噪声可显著放大为位置和几何误差，强噪声在薄壁U上破坏拓扑且产生非流形顶点。

这支持继续补齐第一关，不支持跳入FM训练或宣称架构优于MeshFlow/QEMesh。仍缺两个真实音箱和两个经新几何审核的原径向拒绝瓶子；没有测试多种子、原始顺序置换、OT、神经网络、自由集合vs体素、QEM简化参照或生成分布。

下一步应保持本轮原结果不变，补回真实网格字节后单独冻结对象名单、取消径向和genus0筛选重新审核，再运行缺失36项。若未来选择直接生成v*与3×3 A因子，应另立参数化隔离小实验，不把它和本轮完整4×4扰动混合统计或把新的更好结果覆盖本次失败。

## 10. 文件入口及复现

`LEDGER.csv`/`LEDGER.json`：72槽位，36真实结果及36明确未运行项。`runs/`：全部36输出OBJ、参数NPZ、四面体标签、逐次指标和精确审核。`inputs/`：四个参考网格、拟合参数、80步优化轨迹和评价样本。`GALLERY.html`：自包含图册，全部干净/扰动视图及三条剂量曲线；不是挑选成功样本。`code/`：拟合、提取接入、审核、汇总与渲染脚本。

运行 `bash reproduce.sh /absolute/path/to/a/new/empty/directory` 会在新目录重算四对象36项；不会访问旧权重或网络训练，也不会假造缺失四对象。它需要事先安装环境依赖，版本清单为 `DEPENDENCIES.json`。`audit/`存审核控制与非流形见证。绘图首次因目录组写权限失败，修复后仅重新渲染；未重跑或改变提取结果，记录于 `logs/figure_attempts.log`。

### 原始公开方法与软件入口

- PoNQ论文：https://arxiv.org/html/2403.12870v1
- 官方固定源码：https://github.com/nissmar/PoNQ/tree/dde842b2c00b4a96f67ad07343ae3d48b43fe8ea
- 提取器：https://github.com/nissmar/PoNQ/blob/dde842b2c00b4a96f67ad07343ae3d48b43fe8ea/src/utils/PoNQ_to_mesh.py

外部文献只支持方法身份，不支持本报告的新实验数值；数值来自本轮实际存档。
'''
(r/'REPORT_zh.md').write_text(report)
# Standalone image gallery, with every measured mesh rather than a selected subset.
def image(file,caption):
    data=base64.b64encode(file.read_bytes()).decode()
    return f'<figure><img loading="lazy" src="data:image/png;base64,{data}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
header='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>PoNQ Gate 1 · 实测图册</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1500px;margin:30px auto;padding:0 22px}h1,h2,h3{line-height:1.3} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}figure{margin:0;border:1px solid #ddd;padding:8px;border-radius:8px}img{width:100%;height:auto}figcaption{padding:8px;font-size:14px} .notice{background:#f5f5f5;border-left:4px solid #777;padding:14px}table{border-collapse:collapse;overflow:auto}td,th{padding:8px 12px;border:1px solid #ddd}details{margin:20px 0}summary{cursor:pointer;font-weight:600}</style><h1>PoNQ Gate 1：实际重建与扰动图册</h1><p>2026-09-25 · 无网络、无训练 · 官方提取核心固定提交 dde842b2</p><div class="notice"><b>仅完成四个合成对象36次提取；不是完整八对象72次验证。</b>本地两音箱、两瓶子原始字节不可获取，另外36槽位未运行。RMS单位为参考bbox对角线百分比；所有视图直接渲染最终审计OBJ。QEM因子噪声会同时改变解算位置；不等同于纯位置噪声。</div>'''
parts=[header,'<h2>干净重建：参考 / 128元素 / 256元素</h2>']
for obj,title in names.items():
 parts.append(f'<h3>{title}</h3><div class="grid">')
 for s in ['reference','clean128','clean256']:parts.append(image(r/'figures'/f'{obj}__{s}.png',f'{title} · {s}'))
 parts.append('</div>')
parts.append('<h2>预设强QEM扰动：全部四对象</h2><p>薄壁U出现2个连通分量和1个非流形顶点；其它对象虽保留Betti，几何也显著变差。</p><div class="grid">')
for obj,title in names.items():parts.append(image(r/'figures'/f'{obj}__qem_strong.png',title+' · QEM因子10%'))
parts.append('</div><h2>扰动剂量曲线</h2><p>每个对象每种因素只有一组固定随机方向，连线不是拟合或置信区间。</p><div class="grid">')
for tag in ['position','normal','qem']:parts.append(image(r/'figures'/f'curve_{tag}.png',tag))
parts.append('</div><h2>全部扰动逐输出视图</h2>')
for obj,title in names.items():
 parts.append(f'<details><summary>{title}：七种扰动，全部保留</summary><div class="grid">')
 for s in ['position_light','position_strong','normal_light','normal_strong','qem_light','qem_strong','joint_light']:parts.append(image(r/'figures'/f'{obj}__{s}.png',state_names[s]))
 parts.append('</div></details>')
cols=['object','state','vertices','faces','rms_pct_bboxdiag','p95_pct_bboxdiag','normal_abs_mean_deg','beta0','beta1','beta2','nonmanifold_vertices','exact_intersection_pairs','extraction_seconds']
parts.append('<h2>36个实际输出账本</h2><div style="overflow-x:auto">'+active[cols].to_html(index=False,float_format=lambda x:f'{x:.5g}',escape=True)+'</div>')
parts.append('<h2>未运行对象</h2><p>loudspeaker_local_1、loudspeaker_local_2、bottle_local_1、bottle_local_2：共36槽位，NOT_RUN_MISSING_RAW_ASSET。没有选择父ID、没有重新几何审核、没有用合成代理替换。详情见同包 ASSET_ACCESS.json 和 REPORT_zh.md。</p></html>')
(r/'GALLERY.html').write_text(''.join(parts))
deps={p:importlib.metadata.version(p) for p in ['numpy','scipy','torch','trimesh','networkx','shapely','pandas','matplotlib']}
dump(r/'DEPENDENCIES.json',deps)
print('REPORT_CHARS',len(report),'GALLERY_BYTES',(r/'GALLERY.html').stat().st_size,'DEPENDENCIES',deps)
