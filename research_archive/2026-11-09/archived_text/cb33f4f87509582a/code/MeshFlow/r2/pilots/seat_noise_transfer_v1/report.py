"""Write evidence-grounded report from fixed outputs; zero model calls."""
import csv,json
from pathlib import Path
from . import common as C
O=C.OUT

def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    panel=rows(O/'results/per_panel.csv');cond=rows(O/'results/per_condition.csv')
    labels=('EXP67_FROZEN','SMALL_C200','FULL_C200')
    lookup={(r['panel'],r['label'],r['stage']):r for r in panel}
    short={'TRAIN_FAMILIAR':'熟悉4C，新噪声','DEV_HELDOUT':'DEV 4C，新噪声'}
    lines=['# R2_SEAT_NOISE_TRANSFER_V1', '', '2026-10-13。已实际完成：零训练，三冻结模型，144份新RAW、144份单次D0 POST；48逐请求图、8款式总览、288条实际AI_PROXY图审与24条款式记录。', '',
    '**两个问题的答案不同：集中200步的部分POST局部接续收益可以在新噪声上重复；座面采用的总体改善没有复现。对未进67父条件训练的DEV，只有很弱的局部迁移信号，不能据此认定学到了普遍座面控制，也不能把结果单因归为记住训练椅子。**', '',
    '熟悉32请求中，SMALL相对EXP67的局部合法共边输出12→15，新增3、丢失0；明确采用16→16，1例NO→YES、1例YES→NO。原核心3f7417在新8噪声上的采用两者都是5/8；本轮没有再次显示集中组相对起点的采用优势。DEV16请求中接口0→1、明确采用3→5，但新增YES均来自UNCERTAIN→YES，并非两个确定失败转成功。', '',
    '## 固定模型、数据与实际执行', '',
    '- EXP67_FROZEN：原67父训练6000步的固定generator_state推理导出。本轮加载的不是另选EMA。',
    '- SMALL_C200：上一轮从同一EXP67/AdamW出发，4父4C集中续训200步；每父400实例曝光。名称SMALL表示条件池小，不表示C面数更少。',
    '- FULL_C200：上一轮从同一EXP67/AdamW独立分叉，原67父194条件普通续训200步；每父23或24实例曝光。它不是SMALL后再混合训练的模型。',
    '- 本轮三者全部冻结，无更新、反向或checkpoint挑选。熟悉4C各8新噪声=32请求；4个DEV父各4新噪声=16请求；三模型共享每个请求完整数组。',
    '- 所有144RAW保存并校验后才开始POST。统一Euler50、BF16前向/FP32参数与积分、严格确定性、TF32关闭、SDPA math；每份单次官方D0→C保护/GEO_C→来源C角序恢复与free定向。无RBF、辅助模块、新loss或多次D0。',
    '- 每个请求保存全N标准Gaussian与实际z；known槽置零只为旧推理合同，free保持同一原始数组切片。48新seed与可用343122个历史唯一seed无碰撞；8个2048点查询库原坐标/权重字节复用。',
    '- 模型/输入/评分冻结注册在registration.json，manifest SHA256为2ae65293d79b0de11477cb26769bd969deefab1b278548a91c185cc141a2c122。旧噪声只保留在单独OLD_NOISE_REGRESSION_REFERENCE_ONLY.csv，未并入本轮。', '',
    'DEV选父在采样前按已有合格来源、FULL/HALF及父ID顺序冻结；未用生成结果选择。排除了67父ID、group、相同几何hash和已知PAIR04跨group近重复。它们曾用于开发评估；官方生成器/D0是否预训练见过、其余未穷尽近重复仍UNKNOWN，故不是全新无泄漏测试认证。', '',
    '两面板各只有4个父，重复噪声不是新增独立父。先在条件内平均噪声再对4父等权；不合并熟悉与DEV，不作统计显著性或广泛泛化主张。', '',
    '## 熟悉条件：接续有所重复，采用不是整体上升', '',
    '下表均为POST，百分比为原指标的面板宏均值。图审采用列按YES/NO/UNCERTAIN列出，RAW与POST的采用判断本轮相同。局部合法边只指至少一条合格的C/F双面共边。', '']
    metrics=[('局部合法共边输出','valid_edge_outputs','count'),('采用YES / NO / UNCERTAIN',None,'review'),('严格A-VMR','A_VMR__raw_tau_1e-6','pct'),('严格PMR','PMR__raw_tau_1e-6','pct'),('方向可达free面积','opposite_winding_reached_free_area_fraction','pct'),('CF非法free面比例','CF_illegal_affected_free_fraction','pct'),('FF非法free面比例','FF_illegal_affected_free_fraction','pct'),('I / ConflictCov_C','ConflictCov_C','pct'),('free小角<5度','free_minimum_angle_lt5_fraction','pct'),('实际free面数均值','actual_free_faces','num')]
    def table(p):
        out=['|指标|EXP67|SMALL_C200|FULL_C200|','|---|---:|---:|---:|']
        for title,key,kind in metrics:
            vals=[]
            for label in labels:
                r=lookup[p,label,'POST']
                if kind=='review':v='/'.join(r['C_adoption__'+x] for x in ('YES','NO','UNCERTAIN'))
                elif kind=='count':v=r[key]+'/'+r['n']
                elif kind=='pct':v=f'{100*float(r[key]):.3f}%'
                else:v=f'{float(r[key]):.3f}'
                vals.append(v)
            out.append('|'+title+'|'+'|'.join(vals)+'|')
        return out
    lines+=table('TRAIN_FAMILIAR')
    lines += ['', 'SMALL新增局部接口是新3f7417/n0、8c9b20/n0与n7，分别2、17、3条方向一致合法共边，可达free面积99.620%、68.400%、62.709%。同噪声原EXP67为0条；其中前两条请求FULL也新增接口，故它们不是集中曝光独有收益。SMALL保住全部12个原有接口输出；FULL新增2、丢失3，合计11/32。', '',
    '采用的确定改善只有8c9b20/n6；确定退化为85b169/n4。85b169/n1还从UNCERTAIN变为NO。3f7417的新8噪声：EXP67和SMALL同为5 YES/3 NO，FULL为3 YES/5 NO。其新n0三模型本已采用，只有局部接口改变；不能沿用旧n0故事，把新的同名n0误认为同一噪声或新增采用。73b369仍只有1 YES、6 NO、1 UNCERTAIN，没有解除主要错层。', '',
    'SMALL对熟悉面板的POST CF、FF、I及小角宏均略降或下降，但RAW CF从4.186%升到4.349%。所有144RAW严格A-VMR、PMR及实际合法共边均为0；精确连接是生成几何与相同后处理共同形成，不能归功为RAW已生成正确拓扑。', '',
    '## DEV：保留局部正信号，迁移证据较弱', '']
    lines += table('DEV_HELDOUT')
    lines += ['', '唯一新增接口位于608b46/n1：SMALL有1条方向一致局部合法边，只可达2.931%的free面积；面板宏均0.183%。图审此请求从UNCERTAIN→YES；5b1d0d/n3也从UNCERTAIN→YES，2d08a6/n1则NO→UNCERTAIN。因此明确YES从3增到5，不能写成两个确定失败被修复。FULL只有5b1d0d/n3的UNCERTAIN→YES，未形成局部合法接口。', '',
    '两个DEV HALF（2d08a6、610b16）共8请求，SMALL没有明确采用YES，也没有局部合法接口。5b1d0d虽4/4看起来采用，仍0/4合法接口。这个差别再次说明视觉位置合理、顶点靠近和真正索引接续是三种证据。', '',
    'DEV的SMALL POST CF由15.246%升至16.050%，FF和I略降；RAW CF由9.450%升至9.835%、I由5.472%升至7.373%。不能用POST某个均值降低掩盖RAW代价，也不能说迁移没有代价。', '',
    '## 逐父保留全部噪声', '', '|面板/父|类型 K/N|EXP67采用 Y/N/U；接口|SMALL采用 Y/N/U；接口|FULL采用 Y/N/U；接口|','|---|---|---|---|---|']
    reqs=C.read(O/'inputs/manifest.json')['requests']
    bycid={q['condition_id']:q for q in reqs}
    clook={(r['condition_id'],r['label'],r['stage']):r for r in cond}
    for cid in sorted(bycid,key=lambda c:(bycid[c]['panel']!='TRAIN_FAMILIAR',c)):
        q=bycid[cid];vs=[]
        for label in labels:
            r=clook[cid,label,'POST'];vs.append('/'.join(r['C_adoption__'+v] for v in ('YES','NO','UNCERTAIN'))+'；'+r['valid_edge_outputs']+'/'+r['n'])
        lines.append('|'+short[q['panel']]+'/'+q['parent_id'][:6]+'|'+q['condition_type']+' '+str(q['K'])+'/'+str(q['N'])+'|'+'|'.join(vs)+'|')
    lines += ['', '## 是不是只记住了几个椅子pattern？', '',
    '**支持“有模式偏好”，不支持“已经证明只是复制训练集”。**', '',
    '- 3f7417：EXP67与SMALL都在5/8噪声出现弧顶竖条四腿款，同时保留桶形与其他背板/支撑形态。该重复簇在冻结起点已经存在，不能全归因于200步集中训练。',
    '- 85b169：SMALL在新8噪声里出现实板厚座、高背、弧顶实板及竖条等变化。旧两噪声很相近，不足以推广为全部噪声坍缩；本轮多款式仍有错层及采用退化。',
    '- 8c9b20：SMALL有7/8维持横档四腿粗款式，另外两模型变化更明显；这是款式集中信号，也伴随部分局部接续收益。不能将其直接等同于坐标复制、整体多样性坍缩，或把变化较大的失败形体都算作有用多样性。',
    '- DEV 5b1d0d在续训两组都高度重复实板背/板式支撑，冻结起点也已有优势款式簇。即使没有参与67父条件训练，也可能被同一个生成模式吸引；本轮没有完成到训练全池的几何最近邻/复制检测。', '',
    'DEV还同时改变了C表示：5b1d0d FULL K32/N168（19.048%）超出EXP67同类型K上限21和K/N上限12.651%，物理面积比例却在范围内；608b46 FULL只有K2。两个DEV HALF均K1，focused对应K6/11且物理覆盖更大。K1在原EXP67 HALF本来常见，所以也不能简单说模型完全没见过。父身份、三角划分粒度和区域大小是混杂因素，详见inputs/COVERAGE_LIMITS.md。', '',
    '**可以排除的过强说法：所有新噪声都只有同一款；200步使每个座面条件都更好；新父只要C在座位高度就已接好。仍未解决的问题：集中曝光改善的是可迁移的座面约束，还是主要把熟悉C推向更易被GEO连接的形状簇。**', '',
    '## 定义、质量与审阅限制', '',
    'C_preserved_exact、来源C连接身份、I状态和局部拓扑状态在288行均为PASS/True。这里的拓扑PASS表示被审计的邻接分类没有UNKNOWN，不等于网格流形、水密、全局无交或完全可定向。原源FF风险保留：熟悉两条FULL有54/18非法源面对；DEV亦有2/12/41等源风险。本轮没有修源网格或把来源款式当唯一正确答案。', '',
    '共边合法性只检查该两张邻面，不能排除它们与第三张面的冲突。大可达面积可经一条边到达，不是边界封口比例。严格A-VMR是C来源边界顶点对free顶点的1e-6模型坐标阈值匹配；PMR要求全部边界顶点匹配。完整八阈值、CF proper/illegal、I上界/不确定性、删除面数与各自分母均保留在原逐例表，不只报告有利项。', '',
    'POST删除free，三个模型的实际free分母不同，已表列。小角可能表达必要细结构；本轮不把小角下降单独当整体质量更好。主体完整图审熟悉EXP67/SMALL/FULL分别30/29/30 YES，其余2/3/2为UNCERTAIN；DEV分别14/15/14 YES、2/1/2 UNCERTAIN，没有明确标NO的主体缺失。部分图形异常保留不确定，不能把没有确定NO写成已认证全部完整。', '',
    '所有48请求页和8总览均实际查看，先保存匿名逐行判断并核对图hash，再统一解盲。A/B/C只在同一C内固定，跨C不可当同一模型。评阅者是AI_PROXY并有先前图像/实现背景，不是人类双盲；字段独立且允许UNCERTAIN，同一C三模型由同一评阅者审。8字段288行和24款式记录全部交付。', '',
    '## 资源、账本与恢复', '',
    '实际144 Euler50=7200次生成器前向，144次单次D0；训练/反向0、悬空调用0、科学重试0。原权重、输入、旧代码与评分实现共125个保护文件最终hash通过；新报告/图审代码独立保存，未改公共模型或旧结果。', '',
    'RTX5080 16303MiB，24逻辑CPU、WSL约62.5GiB RAM。遵守约6GiB实验预算，GPU单worker递增分阶段加载；CPU评分2worker并与独立绘图重叠。保持其他图形程序运行。生成器PyTorch allocated/reserved峰约560/614MiB，D0约1794.44/1824MiB；这是allocator口径，不是完整进程显存。整卡边界采样最高占用8706MiB、最低空闲7274MiB，包含用户程序；WDDM单PID峰不可得，不冒称证明精确6GiB上限。', '',
    '生成循环419.561秒；含加载与单阶段审计487.191秒。三个D0/GEO/导出内部阶段合计180.169秒，其中D0前向14.567秒。CPU288阶段评分331.248秒，绘图31.748秒。总入口另有全RAW屏障、hash与收集时间；子计时不能重复相加，也不等于从准备到交付的总用时。完整时间在ledger/resource记录，非GPU工作包括实现、输入核验和实际图审。', '',
    '工程记录包含预冻结资源检查位置修正、Windows工具1385后的只读图像读取替代、shell引号/WSL路径校验修正；均未重放科学模型调用或改变条件定义。未发生OOM，不自动改变精度、步数或噪声。', '',
    '恢复入口：`python -B -m r2.pilots.seat_noise_transfer_v1.run --stage all`。完成单元校验后复用；未完成模型调用预留拒绝隐式重试。`analyse`校验全部实际图审后解盲；`report`重建本报告；`package build`需COMPLETE及独立PASS后执行。环境与分阶段命令见README_RESUME.md。', '',
    '## 当前决定与下一项检验假说', '',
    '**不晋级SMALL为通用座面模型，也不把FULL普通续训当作更好的默认。保留集中200步作为能改变局部接续的快速候选；原EXP67参照不变。** 已经重复的是熟悉条件局部接续，尚未重复的是总体采用提升；DEV只给一个小接口和不确定判定变化。', '',
    '若继续，优先用少量、与focused在K/区域面积接近但父身份不同的座面C，作同三冻结模型的配对迁移检查，而不是直接延长四C到1000步。检验假说：当前收益主要依赖熟悉条件附近的形状簇与可焊接几何；若匹配表示后的新父也稳定获得RAW采用及POST接口且代价可控，才支持可迁移学习；若仍只改善熟悉4C，则更支持窄条件适配。此为后续建议，本轮没有构造或运行该新实验。', '',
    '交付：', '',
    '- [全部RAW/POST图集](evaluation/gallery/index.html)，56原图本地保留；小包包含全部56张统一压缩副本，真实审阅hash指原图，副本另有hash映射。',
    '- [288逐输出数值](evaluation/per_output.csv)、[全部8阈值](evaluation/AVMR_PMR_all8.csv)、[逐父条件](results/per_condition.csv)、[逐请求配对](results/paired_decisions.csv)、[完整原指标配对](evaluation/paired_per_output.csv)。',
    '- [288真实图审](evaluation/gallery/AI_PROXY_review_unblinded.csv)、[24款式记录](evaluation/gallery/AI_PROXY_noise_style_unblinded.csv)、[指标合同](METRIC_AND_REVIEW_CONTRACT.md)、[覆盖限制](inputs/COVERAGE_LIMITS.md)。',
    '- [调用账本](call_accounting.json)、[独立复核](independent_audit/final_audit.json)、[恢复说明](README_RESUME.md)。实际模型导出沿用历史文件，本轮零新权重。',
    '- 实际新代码与原评分源码快照、code.diff、source_file_hashes.json、SOURCE_FILES_SHA256.csv由package入口保存，小包不含权重与RAW/POST大数组。完整本地产物不删。', '',
    '固定任务到此结束。没有自动继续训练、切默认、commit或push。', '']
    (O/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8-sig')
    print(json.dumps({'status':'REPORT_WRITTEN','path':str(O/'REPORT.md'),'bytes':(O/'REPORT.md').stat().st_size},ensure_ascii=False))
if __name__=='__main__':main()