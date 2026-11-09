"""Build the evidence report from complete saved results, no model calls."""
from . import common as C
import csv

def main():
    s=C.read(C.OUT/'results_v2/summary.json')['rows'];by={(x['group'],x['label'],x['stage']):x for x in s}
    vis='| 条件组 / 模型 | RAW采用 YES / NO / UNCERTAIN | POST有合法接口 | POST合法共边总数 |\n|---|---:|---:|---:|\n'
    for group,name in [('ALL','全部24请求'),('NEW_ONLY','四条改变FULL的16请求'),('CACHE_ONLY','两条原FULL的8请求')]:
        for arm in ['SMALL_C200','CREF_C200']:
            v=by[group,arm,'RAW'];p=by[group,arm,'POST'];vis+=f"| {name} / {arm} | {v['C_adoption_YES']} / {v['C_adoption_NO']} / {v['C_adoption_UNCERTAIN']} | {p['valid_interface_outputs']}/{p['outputs']} | {p['valid_interface_edges']} |\n"
    quality='| 全24请求宏均值 | SMALL RAW | CREF RAW | SMALL POST | CREF POST |\n|---|---:|---:|---:|---:|\n'
    for label,m in [('CF非法相交影响free面比例','CF_illegal_affected_free_fraction'),('FF非法相交影响free面比例','FF_illegal_affected_free_fraction'),('I / ConflictCov_C','ConflictCov_C'),('free小于5度角面比例','free_minimum_angle_lt5_fraction'),('方向一致可达free面积','opposite_winding_reached_free_area_fraction'),('A-VMR，tau=0.001L','A_VMR__L_tau_0.001'),('PMR，tau=0.001L','PMR__L_tau_0.001')]:
        vals=[by['ALL',a,t][m+'_mean']*100 for t in ['RAW','POST'] for a in ['SMALL_C200','CREF_C200']];quality+='| '+label+' | '+' | '.join(f'{v:.3f}%' for v in vals)+' |\n'
    parent='| 父 / K | SMALL RAW采用 Y/N/U | CREF RAW采用 Y/N/U | SMALL POST接口输出 | CREF POST接口输出 |\n|---|---:|---:|---:|---:|\n'
    r=C.read(C.OUT/'registration.json')
    for c in r['conditions']:
        uid=c['parent_id'];a,b=[by[uid,m,'RAW'] for m in r['labels']];ap,bp=[by[uid,m,'POST'] for m in r['labels']]
        fmt=lambda v:'/'.join(str(v['C_adoption_'+k]) for k in ['YES','NO','UNCERTAIN'])
        parent+=f"| {uid[:6]} / {c['K']} | {fmt(a)} | {fmt(b)} | {ap['valid_interface_outputs']}/4 | {bp['valid_interface_outputs']}/4 |\n"
    txt=f'''# R2_REGULAR_PATCH_CONTROL_V1

状态：COMPLETE。2026-10-13 本地实际执行；零训练。冻结 SMALL_C200 与 CREF_C200，完成输入构造、真实采样、单次D0、CPU评分、全部图审与独立核验。

## 结论先行

**完整、连贯且K≥2的用户式输入值得作为后续主面板，但这次没有证明它能普遍解决座面采用与接续。** 在相同完整输入下，CREF在少数请求上更像把C当座面使用；其合法接续更少、CF和I代价更高。不能据此晋级CREF，也不能把旧失败全部解释为单三角条件太难。

新两三角矩形610b16在两模型、四噪声中都没有合法C/F共边，仍有明确错层或不确定接合。局部正结果主要来自3f7417：SMALL的FULL在3/4噪声保有合法接口，边数11/12/0/12；其旧HALF是2/5/0/8。合法边总数15→35，但有接口输出仍是3/4。这个输入后缘有凹口、且在先前条件训练中已经出现，不能推广为任意规则四边形都成功。

## 数据和冻结合同

六父、每父一条既有真实SEAT_FULL、四份既有噪声n0–n3、两模型。K分别20/14/12/14/2/2，N分别231/168/166/238/228/140；全部源ID边连通分量1、边界环1、Euler=1、无C非流形边及来源C内部绕序矛盾。

- 610b16、608b46：原生两个三角形构成平面矩形。
- 73b369、85b169、2d08a6：完整座面，含原来的弧边或倒角；没有简化成新平板。
- 3f7417：完整共面上表，但后缘有凸块与凹口。它是单独披露的复杂轮廓对照，不称严格矩形。

直接读取现成FULL源面，保存原face IDs、vertex IDs及可逆角序；没有重三角化、移动顶点、拼无关面或修完整源椅。85/2d/608来源既有FF风险分别18/41/12对，其C内部及C/F既有非法计数为0；这不认证完整来源无交。

85b169与608b46旧输入已经是FULL，C、N、完整噪声、实际查询、权重和处理合同身份完全一致，16组模型请求RAW/POST和32条评分严格复用。其余四父构成16请求，每模型真实新增16份：共32 RAW、1600次生成前向、32次官方D0、64条新CPU评分。合计48 RAW、48 POST、96逐输出评分，未产生新高斯抽样或新权重。

所有RAW先完成，再做单次官方D0→GEO_C→来源C角序恢复及free方向整理。Euler50，BF16前向、FP32状态积分、确定性算法及原注意力合同不变；无RBF、无新loss、无多次去噪。

噪声沿用旧完整N槽数组，按新K将known槽清零；共同绝对槽的数值不变，但整个free数组长度已改变。CREF仍将同一标准高斯作为局部坐标起点，并加FULL的面积中心回世界坐标。未为凑相同世界起点偷偷去均值。

四条HALF→FULL同时改变K、面积、边界、free集合和CREF中心。面积比约1.9965、1.4777、2.2061、2；中心位移约0.7266、0.3287、0.8706、0.9934模型单位。**这不是纯K消融，也不是纯坐标消融。** 73/3f/85 FULL在EXP67实际6000步记录中曝光165/183/185次；后续200步只直接集中于73/3f HALF、85 FULL等既定条件。三DEV父未参与条件训练；官方预训练是否见过UNKNOWN。六父均已有研究观察，不称新父盲测。

## 采用与接续分开

以下采用判断来自两位非盲AI_PROXY：全24页有限截面图加全24页逐像素深度补图均实际查看，96个模型/阶段逐项记录。不是人工用户评测，也不是水密认证。RAW/POST采用标签在本轮最终V2中一致，但两阶段原表仍分开。

{vis}
{parent}

三个配对NO→YES采用改善是73b369 n2与2d08a6 n2/n3；CREF把蓝色C组织到上座面，SMALL仍有错层。可是三份CREF输出均没有合法接口：73的局部结构仍拥挤，2d的背顶翘片、后伸或局部缝隙也没有因此消失。610 n0/n3从明确错层到不确定关系，不计成YES收益。

全部24请求有接口的7/24与3/24包含两条原FULL缓存。**四条真正改变的FULL子面板是SMALL 3/16、CREF 0/16。** 再排除唯一后缘凹口对照3f，三个改变的较规整条件12请求，两方法均为0/12合法接口。这个分层是完整披露，不是删掉3f后重定义主面板。

RAW按原精确索引/坐标定义，两模型均0合法共边。它不等于RAW毫无条件响应，也不能靠图上贴近替代POST实际索引共边检验。合法共边、方向可达面积与完整边界封接不同；全组PMR在严格/宽松容差的完整八档均已保存，本文展示0.001L仅便于阅读。

## 几何代价

{quality}

CF/FF分别是被非法对影响的free面比例，分母随POST删除面变化，逐输出分母完整保存；I使用各条件固定的2048个面积查询点，cross/dup/union及不确定量均保留。I不是采用率，近平行薄底面不能自动当冗余，图上采用也不能豁免真实相交。全部原指标数值定义不变。

只看新FULL的16请求，SMALL→CREF POST：CF 3.134%→7.611%，FF 15.194%→18.241%，I 6.421%→8.627%。因此局部用途判断的改善没有带来更好的接续或统一质量优势。

原HALF与新FULL的I面积查询域和A-VMR边界域已改变，不能把两者百分数差称作同一指标域上的净控制收益。历史逐例分数及实际图审保存在historical_context；只作上下文。两条恒等FULL的旧/新图审也有变化，说明跨评阅方式的“成功率提升”不能都算模型收益。

## 图像显示修正及图审版本

原V1图复用Matplotlib分立C/F Poly3DCollection，平均深度painter排序可能把可见蓝C错误遮住。原网格、旧图和96条V1图审保留。新的depth_gallery使用逐像素有限三角形重心插值深度比较，同一真实网格、双正交视图、条件内统一尺度；24页一律补图，未挑例。独立解析测试验证近F会遮住C、近C会遮住F，斜交面按每像素选近；只有精确或1e-12内深度平局保留先绘的C。

两位评阅者统一重看全24补图，结合不受painter影响的原有限剖面，形成V2而不覆盖V1。可见性修正导致的U→YES属于图审修正，**不是新增模型收益或后处理收益**。深度图仍不能认证次像素缝隙、共面重复层或实际拓扑；两条中央剖面也可能漏掉偏离切面的结构。两个版本的表、变更原因和图片hash全部交付。

## 执行、恢复和保护

输入独立CPU审计1397项PASS；执行/数值独立审计41121项PASS，2187文件/hash核验。1600前向reserve/return完整，32次D0严格单次，所有48 RAW/POST身份与C保持成立；96分数逐查询mask重汇总及配对算术通过。没有失败模型前向或模型重试。

RTX5080 16303MiB；开始约2600MiB全卡占用。逐阶段资源快照、CPU/RAM及GPU进程已记录，保留用户进程。抽样观测全卡最高4806MiB；这只是观测值，不能当精确进程峰值。生成allocator reserved约614MiB/进程。测到首worker后曾追加另一独立生成进程，但实际前向时间没有重叠，不能宣称GPU并行加速。两臂16次采样阶段各约57/60秒，D0阶段约39/40秒；CPU评分实际两个worker约88秒。准备、渲染、图审、报告时间另计。

工程修正均登记：准备时按真实索引解析缓存路径；恢复调度只认本进程新完成receipt并清理过期内存估计；新增显示器统一RAW/POST数组形状。都没有改变冻结科学定义、增加预算或覆盖旧资产。

WSL环境的总恢复入口：

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.regular_patch_control_v1.run --stage status
# 必要时 --stage all：校验已完成产物并跳过已完成神经调用；不静默重跑未完成cell。
```

新代码位于r2/pilots/regular_patch_control_v1；结果独立位于experiments/r2_regular_patch_control_v1。旧权重、模型、评分源码、RAW和旧报告保持只读。没有切换默认模型、commit或push。

## 交付与下一步判断

- evaluation/per_output.csv：完整96行；paired_per_output.csv、paired_per_parent.csv及八档AVMR_PMR均保留。
- results_v2/paired_visual.csv、summary.csv：最终配对图审与分组结果；V1留在results和gallery原CSV。
- preparation/condition_audit.json及输入图：六条完整区域、来源面ID、拓扑及规则性核对。
- evaluation/depth_gallery/index.html：建议先看；evaluation/gallery/index.html保留全部有限剖面。
- registration、inputs、查询、calls账本、执行/数值/显示审计及原始轨迹均在本地；审阅ZIP包含逐例CSV、96份精简最终几何、全部图、实际评分/新入口源码、code.diff及hash，不含大权重。

**优先保留这套规整完整输入作为主评测任务，不因CREF局部用途改善就切换默认模型。** 当前最具体的未解问题是：F的座框/侧壁能响应C的大致布局，却仍不能稳定落到C的实际边界上。它是待检验的机制假说，不是本实验单独证明的根因。若以后继续快速学习检验，应在同一规整FULL主面板下检查采用和实际接续能否同时改善，并保留CF/FF/I代价；本轮不自动训练、不扩大面板或扫描新方案。
'''
    C._write_bytes(C.OUT/'REPORT.md',txt.encode('utf-8'));print({'status':'COMPLETE','report_sha256':C.sha(C.OUT/'REPORT.md')})
if __name__=='__main__':main()