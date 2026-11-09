"""Finalize reports from completed fixed-panel receipts; no model imports or forward calls."""
from pathlib import Path
import csv,json,hashlib,shutil,difflib,statistics,datetime
ROOT=Path('/mnt/e/MeshFlow'); OUT=ROOT/'experiments/r2_postprocess_avmr_pmr_v1'; CODE=ROOT/'r2/tools/postprocess_avmr_pmr_v1'
def j(n): return json.loads((OUT/n).read_text(encoding='utf-8-sig'))
def rows(n): return list(csv.DictReader((OUT/n).open(encoding='utf-8-sig')))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
 return h.hexdigest()
def write(n,s):
 p=OUT/n; assert p.resolve().is_relative_to(OUT.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8')
def js(n,v): write(n,json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(headers,rs): return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rs])
def csvsave(n,rs):
 with (OUT/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def pct(v): return f'{float(v)*100:.2f}'
reg=j('registration.json'); summary=rows('summary.csv'); quality=rows('quality_per_output.csv');qs=rows('quality_summary.csv');mods=rows('modifications_per_output.csv');top=rows('topology_per_output.csv')
labels=['R2_FROZEN','R2_CONT_OLD','R2_CONT_USER','R2_USER_MIX']; stage='DENOISE_C_GEO_C'
def q(label,s,group='ALL'): return next(r for r in qs if (r['label'],r['stage'],r['group'])==(label,s,group))
def a(label,s,setting,group='ALL'): return next(r for r in summary if (r['label'],r['stage'],r['setting'],r['group'])==(label,s,setting,group))
qkeys=['CF_proper_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','ConflictCov_C_cross','ConflictCov_C_dup','free_minimum_angle_lt5_fraction']
qnames=['CF穿插受影响面↓','FF非法相交受影响面↓','I冲突覆盖↓','I_cross↓','I_dup候选↓','free小角<5°↓']
byq={(r['label'],r['request_id'],r['stage']):r for r in quality}
pairs=[]
for record in reg['records']:
 label,req=record['label'],record['request_id'];r=byq[label,req,'RAW'];p=byq[label,req,stage]
 for key in qkeys:
  pairs.append(dict(label=label,request_id=req,parent_id=r['parent_id'],region_type=r['region_type'],candidate=r['candidate'],metric=key,RAW=float(r[key]),POST=float(p[key]),delta_POST_minus_RAW=float(p[key])-float(r[key]),units='fraction',lower_is_better=True))
csvsave('quality_paired_RAW_POST.csv',pairs)
paired_summary=[]
for label in labels:
 for group in ['ALL','seat','back']:
  for key in qkeys:
   rr=[r for r in pairs if r['label']==label and r['metric']==key and (group=='ALL' or r['region_type']==group)];ds=[r['delta_POST_minus_RAW'] for r in rr]
   paired_summary.append(dict(label=label,group=group,metric=key,outputs=len(rr),mean_delta=statistics.mean(ds),median_delta=statistics.median(ds),min_delta=min(ds),max_delta=max(ds),improved=sum(v < -1e-12 for v in ds),tied=sum(abs(v)<=1e-12 for v in ds),worsened=sum(v>1e-12 for v in ds)))
csvsave('quality_paired_summary.csv',paired_summary)
mainrows=[]
for label in labels:
 r,p=q(label,'RAW'),q(label,stage)
 mainrows.append([label]+[pct(r[k])+' → '+pct(p[k]) for k in [qkeys[0],qkeys[1],qkeys[2],qkeys[5]]])
maintable=table(['模型','CF（%）','FF（%）','I（%）','小角（%）'],mainrows)
matching=[]
for label in labels:
 t=[r for r in top if r['label']==label and r['stage']=='POST_INDEXED']
 matching.append([label,pct(a(label,stage,'raw_tau_1e-6')['A_VMR']),pct(a(label,stage,'raw_tau_1e-6')['PMR']),pct(a(label,'RAW','L_tau_0.01')['A_VMR'])+' → '+pct(a(label,stage,'L_tau_0.01')['A_VMR']),str(sum(int(r['C_open_edges_with_free'])>0 for r in t))+'/12'])
matchtable=table(['模型','POST严格A-VMR↑（%）','POST严格PMR↑（%）','A-VMR@1%L RAW→POST（%）','POST存在C/free共边的输出'],matching)
fullmatch=[]
for st in ['RAW',stage]:
 for label in labels:
  fullmatch.append([st,label]+[pct(a(label,st,x['name'])['A_VMR'])+'/'+pct(a(label,st,x['name'])['PMR']) for x in reg['settings']])
fulltable=table(['阶段','模型']+[x['name'] for x in reg['settings']],fullmatch)
costrows=[]
for label in labels:
 r=[x for x in mods if x['label']==label];t=[x for x in top if x['label']==label and x['stage']=='POST_INDEXED'];deleted=sum(int(x['deleted_free_faces']) for x in r);orig=sum(int(x['N'])-int(x['K']) for x in r)
 costrows.append([label,f'{deleted}/{orig} ({deleted/orig*100:.2f}%)',max(int(x['deleted_free_faces']) for x in r),f"{max(float(x['denoise_max_free_displacement']) for x in r):.5f}",f"{statistics.mean(int(x['shared_edge_components']) for x in t):.2f}",f"{statistics.mean(int(x['nonmanifold_edges']) for x in t):.2f}"])
costtable=table(['模型','free删面总数/原数','单例最大删面','去噪最大free角点位移（模型单位）','POST共边分量均数','POST非流形边均数'],costrows)
regionrows=[]
for label in labels:
 for group in ['seat','back']:
  r,p=q(label,'RAW',group),q(label,stage,group)
  regionrows.append([label,group,r['outputs']]+[pct(r[k])+' → '+pct(p[k]) for k in [qkeys[0],qkeys[1],qkeys[2],qkeys[5]]])
regiontable=table(['模型','区域','输出数','CF%','FF%','I%','小角%'],regionrows)
conditions=sorted({(r['parent_id'],r['region_type'],r['tier'],r['K'],r['N']) for r in reg['records']})
conditiontable=table(['parent_id','区域','范围','K','原N'],conditions)
ffwins=sum(r['delta_POST_minus_RAW']<0 for r in pairs if r['metric']==qkeys[1])
findings=f'''# 同一48份RAW经过完整后处理后的结论

已完成官方Chair学习式去噪48次、C原样恢复、GEO_C几何整理及48份复评分。没有重新生成网格，没有训练，也没有新抽噪声。全部48份最终输出的C坐标、顺序、重复数和绕序保持不变。

**你的判断有实验证据支持：单看RAW会明显高估最终生成面之间的冲突。** 同一评分器下，FF在全部{ffwins}/48个配对输出中下降。但后处理对“C和生成主体真正接合”的帮助有限，不能用FF改善替代接合结论。

数值为百分比，箭头表示同一输出集RAW→POST；CF/FF/I/小角是各自不同的量，均低为好。

{maintable}

**USER_MIX：FF 89.65%→12.02%，CF 5.60%→4.88%，I 3.77%→3.69%。** 它在本面板POST的这三项仍最低；这是几何冲突证据，不等于完整生成质量或接合最好。小角面17.18%→17.06%，几乎没解决，座面子集反而31.93%→33.28%。

{matchtable}

RAW严格A-VMR和PMR四组原本均为0。POST虽然多了少量精确顶点匹配，严格PMR仍全部为0；真正存在至少一条C开放边与free共享的输出，仅OLD和MIX各1/12。MIX的4份座面在最宽2%L阈值下依旧没有边界顶点匹配。匹配域是C所有开放边，仅作描述性观察，未把所有边登记成必须接合接口。

图集24张（12请求×2视角）全部实际审阅，使用统一相机和同请求统一范围。主要外形和C的位置关系大多延续RAW，没有普遍修复错层/悬空C的图像证据。33ec中等座面n0的MIX细腿明显变细；n1的FROZEN两视角中可清楚看到的细腿变少。删面不一定都是坏事：f596 n1大量减面后主体轮廓仍接近RAW。完整逐例观察属于AI_PROXY，不是人评或盲评。

**目前可以采用的结论：后续比较应同时保留RAW和同一固定POST，不能只用RAW自交给生成器定性；目前也不能把接合问题归因于“只是没开后处理”。** 本轮不修改loss、默认模型或训练计划，固定任务到此结束。

[完整流程、全阈值和代价](REPORT.md) · [全部图集](gallery/index.html) · [逐输出几何评分](quality_per_output.csv) · [逐输出A-VMR/PMR](per_output.csv) · [最终网格索引](post_index.csv)

准备阶段曾因路径变量错误误写1份父网格缓存，已从历史交付包按原SHA256恢复原字节；77份原输入终态hash全部匹配，正式去噪在恢复后开始，RAW未被改写。[事故与恢复记录](engineering_registration_path_incident/RESTORATION.json)。
'''
write('FINDINGS.md',findings)
report=f'''# R2 官方Chair去噪＋C保护几何整理与配对复评分

日期：2026-10-10。状态：48/48完整执行，固定任务结束。版本：r2_official_denoise_c_geo_c_evaluation_v1。

本轮读取四模型已有同请求RAW，补齐学习式去噪和几何整理后再评分。最终结果：**FF普遍明显下降，C边界匹配和实际共享边没有普遍改善。** 正负结果一并保存，不将C精确保留称为接合成功。

## 1. 样本和真实执行量

四模型R2_FROZEN、R2_CONT_OLD、R2_CONT_USER、R2_USER_MIX各12份：以下6个USER开发条件各2份既有噪声，共48。历史RAW、请求、条件、噪声数组和原评分保持可追溯；新结果全部在本独立目录。不是新增holdout，也不把两个噪声当两次独立训练。

{conditiontable}

48次官方去噪前向全部成功，无新增生成器前向、训练、优化器更新或Gaussian。48次C保护几何整理、48次POST几何评分；RAW数值读取历史评分，质量适配器另对1份RAW回算核对40个量完全一致。原注册的8档A-VMR/PMR全部保留。每个条件使用原冻结2048个C查询点，没有重新抽点。

## 2. 实际完整流程与适配边界

`已有RAW → 固定坐标缩放 → 官方Chair denoiser单次前向 → FP32残差相加/逆缩放 → 将C原字节写回 → GEO_C顶点合并/退化重复free面删除 → indexed PLY/NPZ → 统一复评分`

官方资产：[qsun2001/meshflow，固定HF revision](https://huggingface.co/datasets/qsun2001/meshflow/tree/98d9f4a16bf5e8a187c02f12c4812000e8181614/denoiser/002_500M_chair)，文件0030000.pt。大小7,289,979,546字节，SHA256 `6a31f4c7dcbea7a6ac58c26075b20c897fd19aad4c33b5295cc4c93ed7078c36`。保存在assets/，不是新训练权重。下载1次成功、0重试。

匹配源码：[qiisun/MeshFlow commit db86bc32423c316c00bcc2f645a26243e8010f20](https://github.com/qiisun/MeshFlow/tree/db86bc32423c316c00bcc2f645a26243e8010f20)。14个原文件保存在source_snapshot/official_denoiser_db86bc3/。当前denoise分支HEAD与此权重PE不匹配，未用不匹配版本前向。加载实际EMA参数455,592,963个，296个state keys，strict=True无缺失/额外key。加载证据在strict_load.json，资产及历史版本核查在asset_audit/。

匹配历史数据和后处理源码的坐标换算为`x_denoiser = 0.95 × 0.3762 × x_model = 0.35739 × x_model`，输出除以同一个常数。没有按生成bbox归一、clip或重新定尺度。几何合并固定工程阈值`0.015 / 0.35739 = 0.0419709561`模型单位；没有扫阈值。论文原运行时阈值单位仍不可完整追溯，不能称该阈值严格复现论文。

这次是**官方发布Chair denoiser的保持C适配流程**，并非原版无约束导出的逐位复现。必须披露以下差别：

- 原导出会量化坐标、合并顶点和整理法向；为保留C，这里用已有GEO_C替换最终导出整理，没有全局2048档量化、法向翻转、平滑或补洞。C先还原再几何整理，只移动/删free。
- 每例独立且无padding，使用`mask=None`；避开历史masked-SDPA布尔方向问题。用SDPA math代替未安装的FlashAttention。原SelfAttention的`mixed_precision='fp32'`在这里仅选择后端，网络仍BF16 autocast，参数和残差累加为FP32，TF32关闭、deterministic严格开启。
- 从官方源码原样提取RMSNorm和3个sin/cos函数，避开未用的fairscale/megfile依赖；原源码未修改。没有调用随机encode/sample。48个receipt均确认前后RNG状态未消耗。
- 历史脚本是BF16原位残差相加，这里为本地精度合同采用FP32残差相加。官方发布实现带face-count AdaLN，与论文关于denoiser描述并不逐项一致；使用真实发布权重，不宣称重建了文字描述的另一模型。
- 去噪器本身没有C硬约束，会提出C移动；先保存完整无约束proposal及位移，再恢复原C。各组最大C建议位移约0.19–0.29模型单位。不能把恢复C之前的结果当满足条件输出，也不能从本结果推断“允许移动C”时的表现。

GEO_C采用唯一C锚点吸附、free原始点代表合并，角点移动不超过固定半径；非传递聚类，不作均值。只删free的精确退化或重复面。C保留坐标、顺序、重复数和绕序，不承诺恢复输入未提供的任意顶点ID图。最终48个PLY与NPZ实际索引几何逐位一致。

## 3. 几何质量RAW→POST

下表为逐输出宏平均百分比，均低为好。每模型6父×2噪声均衡，等于先噪声平均再父等权。

{maintable}

CF是与C发生严格穿插的受影响free面比例；FF是发生严格穿插或正面积共面重叠的受影响free面比例。**不是相交面配对数除以所有面配对数，也不是有问题物体的比例。** 读取并调用原扫描算法，保留接触分类；POST分母用实际保留下来的free面数，不能假装N仍未变化。

I=ConflictCov_C是冻结C面积查询点上的冲突覆盖；cross与近并行dup候选的并集，保留交集和upper/status。dup候选可能包含合法薄壁，不直接等同非法重复层。无唯一GT排名；未登记必要接口，S/SeamCov记NA，不启用S_join；没有外露任务，不算E。未算FID/KID或新增特征前向，不将低CF/FF/I称完整形状质量认证。

FF在{ffwins}/48对均下降。这个收益包含学习式位移、合并后合法共边/接触形成及删面的合并作用；本轮没有额外去噪-only/几何-only对照，不能把全部收益单独归功于其中一步。最终FF下降约74–78个百分点，删面加权比例约10–15%；两者分母和事件定义不同，不作简单相减或纯贡献归因。

另对保存的面对分类与source_face_ids核账（未重评分）：MIX原2620个非法FF对中，646对涉及被删面，1886对两面都保留却变为共享点956/共享边791/分离139，88对仍非法；另新增120个非法对，最终208。全48例也出现543个新增非法对，已完整保留。**FF改善不仅由删面造成**，但此指标按面计数，对相邻三角形的轻微穿插敏感，不代表等量的大体积穿透消失。[独立分解及语义核对](independent_review/interpretation.md)。

细分区域完整保留如下，避免靠背平均掩盖座面问题。

{regiontable}

## 4. A-VMR／PMR与实际连接

沿用[MeshFIM附录A](https://arxiv.org/html/2605.08744v1#A1)的匹配形式：对C源ID拓扑开放边的唯一顶点，查**仅F角点**最近欧氏距离，严格`d < tau`；每例比例宏平均为A-VMR，全部边界顶点都匹配的输出比例为PMR。两者高为好，many-to-one可用，不包含C自匹配。

这里的匹配域是全部C开放边，尚未由用户登记哪些必须接合；坐标尺度与论文评分兼容性UNKNOWN，因此是本地适配，不是论文表格数值复现。固定尺度L=`sqrt(3)*2/.3762=9.208138264587333`，不按输出bbox改变。严格模型单位1e-6的旧设置名`raw_tau_1e-6`也用于POST，仅为兼容字段名。

{matchtable}

四模型RAW严格A-VMR/PMR均0。POST精确匹配很少；严格PMR仍0/48。MIX在1%L甚至6.75%→5.56%，2%L为17.13%→16.40%；MIX全部4份seat在2%L仍0，说明并非只差微小焊接精度。不能据此说C整张表面都远离生成面，因为此处度量是边界顶点到F顶点而非到表面。

全部8档表，每格A-VMR/PMR（%）：

{fulltable}

RAW拓扑采用严格相同坐标的索引，POST采用实际导出vertices/faces。检测C开放边是否与free使用同一对索引；全48输出只有OLD和MIX各1例有1条这种共享边，也各存在双面反向绕序。不能从两条局部边认定整个接口接通、水密或流形。最终48例均非所有边恰好两面；保留边界边、非流形边和共边连通分量表，没有只报一个看起来有利的连接率。

## 5. 修改代价和图像

{costtable}

位移列是去噪阶段相对RAW的最大free角点移动；后续几何合并另有≤0.0419709561位移，不能把该半径冒充全流程总位移上界。删面总数/原数为加权统计，其他分数仍逐输出宏平均。连通分量按共享完整边，不代表语义部件；非流形边数低也不足以认证全网格流形。

[24张完整图集](gallery/index.html)：每个请求2视角，四模型各RAW/POST并排；同请求8幅子图同坐标范围和相机，蓝C/灰F，未隐藏或美化删除三角面。[逐图AI_PROXY观察](gallery/AI_PROXY_REVIEW.md)另外记录；不是盲评、人评或通用质量分数。整体形状/C相对位置多数延续RAW，部分三角缝和碎边被整理；个别细构件损伤见33ec n0 MIX和33ec n1 FROZEN。大量减面不一定损伤外形，例如f596 n1。小角面也不是这条流程必然删掉的对象。

## 6. 执行、资源和核验

先记录RTX5080总16303MiB及Windows/WSL资源、GPU进程、CPU24核、可用主机RAM约44GiB。真实首例之后测峰再决定保留1个去噪worker；48个短前向总共约5.13秒阶段墙钟，多开一个大权重进程成本大于剩余工作。无额外探针/采样。GPU常驻FP32模型及单例前向scope的PyTorch峰allocated约1.752GiB，reserved约1.779GiB；设备使用约8761→10757MiB含其他进程，不能冒充本进程峰。WDDM无法可靠提供逐进程显存，已保留原nvidia-smi输出。

成功去噪进程总墙钟34.10秒（包含加载等），48次forward计时之和2.60秒，forward阶段含I/O等5.13秒。资源复核后CPU评分用2worker、每worker2线程，墙钟48.58秒；图集1CPU/1线程并行7.01秒。独立终态审计3.61秒。资产下载743秒与源码准备不计入推理速度；并行阶段墙钟不能当成独立性能benchmark。

独立审计没有导入主评分器：NumPy源ID边计数、SciPy cKDTree、手动binary PLY解析及索引edge incidence/union-find，复算768指标行、496顶点距离、192汇总、96拓扑行与384配对，全部一致，最大距离差0。C48/48逐位保持。denoise账本48START+48COMPLETE，无GPU重试。主评分结束171份保护路径核验通过，其中原RAW分析保护的77份原文件最终hash全匹配。[独立报告](independent_review/REPORT.json)。

**工程事故必须保留：** 准备阶段run.py局部路径变量重用，注册JSON曾误写一份父网格缓存`native_t1/runs/control_mvp48_v1/data/meshes/f59627a077dcc2566c63b8b48495261a.npz`。首个执行因此在CPU严格加载后、任何GPU前向前停止。用历史交付ZIP中同路径14375字节副本恢复，SHA256 `df43456d0e2bfa5321d3985d768c6c34774b09852745e9c2b33c5e3942739a0b`与事故前一致，77原输入复核全部通过；随后改用独立变量并加入只允许写本轮OUT的强制边界和负例检查。原始误写记录和恢复证据均在engineering_registration_path_incident/。CPU权重加载共2次，不能写成仅1次；正式GPU48次没有重试。另有未用megfile导入预检失败发生在模型构造前，修复方式见运行适配，不隐瞒这些准备错误。

## 7. 交付和停止边界

- 最终48个可打开的索引PLY与完整NPZ：outputs/；post_index.csv记录完整请求、父、模型和路径。
- denoised/保留恢复C前完整proposal、恢复后中间结果、位移和每例receipt。
- per_output.csv：RAW/POST 768指标行；per_vertex_distances_POST.csv：496边界顶点距离；summary.csv及paired_RAW_POST.csv保留全部阈值、seat/back和逐噪声配对。
- quality_per_output.csv：96行；quality_paired_RAW_POST.csv、quality_paired_summary.csv保存均值/中位数/尾部和改善/退化数。topology_per_output.csv、modifications_per_output.csv保存连接和改动代价。
- scores/：原几何逐对扫描、C逐点冲突向量、逐面形状、完整身份/hash。source_snapshot/、source_hashes.csv、code.diff保存实际代码。asset_manifest.json记录大权重，不将它伪装成小交付。
- 可恢复入口：先`run.py --stage prepare`，再`denoise_run.py`，随后`run.py --stage postprocess`、`run.py --stage score --workers 2`，最后`gallery.py`。所有路径以本工作区注册身份为准；已有有效receipt复用，不重新前向。不要用这个固定面板入口悄悄扩样本。

本轮完成后停止。未自动commit/push，未修改默认模型、训练入口或loss；没有自动开始新训练/阈值扫描。优先修正今后的评价流程：始终配对报告固定RAW和POST，同时单列C接合与细构件损伤；现有证据尚不足以指定某种新loss一定解决接合。
'''
write('REPORT.md',report)
# Freeze executed tool files and readable diff without touching git or old files.
for file in CODE.glob('*.py'):
 dst=OUT/'source_snapshot/local'/file.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dst)
changes=[]
for file in sorted(CODE.glob('*.py')):
 changes.extend(difflib.unified_diff([],file.read_text(encoding='utf-8-sig').splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+file.relative_to(ROOT).as_posix()))
write('code.diff',''.join(changes))
source_rows=[dict(path=f.relative_to(OUT).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted((OUT/'source_snapshot').rglob('*')) if f.is_file()]
csvsave('source_hashes.csv',source_rows)
asset=OUT/'assets/chair_denoiser_0030000.pt'; actual=sha(asset);expected='6a31f4c7dcbea7a6ac58c26075b20c897fd19aad4c33b5295cc4c93ed7078c36';assert actual==expected
js('asset_manifest.json',dict(status='FINAL_SHA256_PASS',path=str(asset),bytes=asset.stat().st_size,sha256=actual,download_receipt='asset_audit/weight_download_receipt.json',source_commit='db86bc32423c316c00bcc2f645a26243e8010f20',source_snapshot_files=len(source_rows)))
print(json.dumps(dict(status='REPORTS_WRITTEN',FF_improved_pairs=ffwins,source_files=len(source_rows),quality_pairs=len(pairs),quality_paired_summaries=len(paired_summary),asset_hash='PASS'),ensure_ascii=False))