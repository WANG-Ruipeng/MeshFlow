"""Assemble completed, reviewed evidence. No model, scoring or rendering calls."""
import csv
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'native_t1/runs/chair_capped500_gradbudget_v1'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def table(header,values):
    return '\n'.join(['| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(header))+' |']+
                     ['| '+' | '.join(str(x).replace('|','/') for x in row)+' |' for row in values])+'\n'

def run(out=OUT):
    out=Path(out)
    account=read(out/'accounting.json');runtime=read(out/'audit/runtime_final.json')
    training=read(out/'training_audit/final/summary.json');diag=read(out/'training_diagnostics/final/summary.json')
    visual=read(out/'visuals/reviews/completion.json');analysis=read(out/'analysis/results.json')
    assert runtime['status']==training['status']==diag['status']==visual['status']=='PASS'
    assert analysis['status']=='COMPLETE' and visual['label_outputs_reviewed']==288
    assert account['returns']['optimizer_update']==3000 and account['returns']['sampling']==240
    assert not account['pending'] and not account['failed']
    cpu=read(out/'status/cpu_completion_receipt.json')
    checkpoints=rows(out/'audit/checkpoint_manifest.csv')
    parts=['# CAPPED500 Replication and Gradient Budget Report\n',
      '任务：`CHAIR_CAPPED500_REPLICATION_AND_GRADIENT_BUDGET_V1`。执行日期：2026-10-07（Asia/Shanghai）。固定三臂、两个终点已实际完成；所有生成和权重保留在本独立目录。\n',
      '## 实际执行账本\n',table(['项目','实际完成','登记上限/范围'],[
        ['正式optimizer更新','T/L/G各1000，共3000','3000'],
        ['正式训练模型前向/反向','27000 / 27000','每更新9F/9B'],
        ['真实接线预检','12F、12B；1次隔离optimizer副本更新并丢弃','≤24F/12B；不保留预检更新'],
        ['新完整Euler50','240次尝试/240成功；12000 NFE','≤290次尝试/14500 NFE'],
        ['历史OLD缓存','48条直接核验复用','没有重生成'],
        ['最终评价','288条：9标签×8父×4噪声','OLD144/FRESH144'],
        ['顶层模型前向',str(account['top_level_forward_attempts']),'≤41524'],
        ['模型反向总数','27012（含12预检）','CPU玩具测试另列'],
        ['CPU小模型测试','累计180F/184B/18玩具更新；最终15项通过','真实模型0、CUDA false'],
        ['模型调用失败/未闭合调用','0 / 0','加载/展示工程故障另列'],
        ['GPU累计持锁',f"{runtime['GPU_occupancy']['closed_seconds']:.6f}s / {runtime['GPU_occupancy']['closed_seconds']/60:.3f}min",'≤14400s；最大1 GPU worker'],
        ['反馈刷新/新增池/CONFIRM评价','0 / 0 / 0','固定原反馈池和DEV父'],
      ]),
      '\n调用原件为 `ledger/calls_*.jsonl`，阶段组织为 `ledger/orchestrator.jsonl`，持锁原件为 `ledger/gpu_occupancy.jsonl`。总前向39012=27000训练+12000采样+12预检；正式更新3000与预检副本1次明确分列。CPU审计自身的 model_calls=0 不代表本实验未训练。\n',
      '最终运行审计PASS：17段占用全部闭合，855个登记保护来源hash一致；直接重读288个NPZ/PLY与51状态路径全部PASS。最终训练审计438533项检查通过，三组各1000更新、3000组跨臂主输入配对及1000组L/G辅助身份配对一致，每臂42个真实参数更新快照。没有默默换坏batch、补噪声或中途修改科学定义。\n',
      '设备RTX 5080，总显存16303MiB，设备显存监测峰值6320MiB；CPU24逻辑核，起始Windows可用RAM68.326GiB、WSL可用RAM观测58.018–61.138GiB。GPU遵守本任务单worker锁，尽管16GiB卡还有余量，也未扩大并发；CPU匹配、评分与审计独立重叠，进程各限2线程。设备显存峰值含桌面/其他用户占用，WSL/WDDM未给出可靠进程显存时记NA，未拿权重大小或PyTorch allocated代替进程VRAM。持锁时间包含加载、保存及持锁CPU等待；不同阶段墙时不能当成孤立kernel基准。\n',
      '评分原始调用合计339.0924秒（48历史分数复用、240新输出评分），0评分失败；完整OLD/FRESH图集CPU渲染69.1483/72.4487秒。时间来源 `status/cpu_completion_receipt.json` 及各原summary；恢复后watcher的completed数组为空是展示计数缺口，18个完成面板与288 cells均已独立核验，不据此误计为0。\n',
      '后半程4000个已登记主FM OT匹配在CPU预计算，原OT定义、输入与随机流未变；4000次均返回、无失败/重试。计算109.2325秒，含启动及保护核验180.9830秒；三臂后续实际输入hash和cache使用均一致。这不是新的反馈池匹配或额外训练样本。\n',
      '### 各GPU阶段实际持锁及设备峰值\n']
    resource={read(p)['stage']:read(p) for p in (out/'resources').glob('*_summary.json')}
    cost=[]
    for r in runtime['GPU_occupancy']['intervals']:
        s=resource.get(r['stage'],{}) if r['status']=='PASS' else {}
        cost.append([r['stage'],r['status'],f"{r['seconds']:.3f}",
            s.get('observed_device_used_peak_MiB','NA'),s.get('observed_own_process_peak_MiB') or 'NA'])
    parts += [table(['阶段','状态','持锁s','设备峰值MiB','进程峰值MiB'],cost),
      '\n失败加载段0.135382秒无模型调用；该失败短段的设备峰值记NA；同名成功恢复段的观测不借用于失败段。\n',
      '### 保护身份与实现范围\n',
      '共同START的SHA256为 `d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80`。全参和AdamW step5000、动量与二阶矩均恢复。结构仍是Native/T1 +实际Geo(P=I)+HYBRID-FM；无新增参数。旧teacher500和CAPPED500仅冻结推理参考，不作本轮训练起点。\n',
      '三臂共享新的主流seed **3594377816**，由登记SHA命名空间推导且不同于旧流。原32父/453 READY、effective batch8/microbatch1、每批8父、4个alpha=1与4个U[.9,1.1]、logit-normal t、free-only nested OT、HYBRID .25、整批free坐标归一、x1头到velocity转换均保留。原24状态/12轨迹、S/I/E队列及父内最坏噪声不刷新；teacher-line辅助epsilon继续使用冻结旧seed20261008。\n',
      '三臂同载体、BF16前向和FP32参数/积分，严格deterministic且warn_only=False、TF32关闭、SDPA math、原eval/dropout约定；AdamW 1e-5/(.9,.95)/wd0，总clip1。没有新scheduler/EMA、时间加权、端点残差、guidance或几何后处理。生成器输入仍只含当前C/N/Gaussian/mask；GT、Gamma、UID和目标尺度不输入网络。\n',
      '协议规范JSON SHA为 `c155f532569415880a13830e4c13f45be5238369c25015a077c3dc4c36c7c8bb`；协议文件字节SHA为 `fd6f43284da0ea914e8f669dc142ba613a6e362c721bef5a23ddc3dde7ab6410`，二者是不同hash定义。训练计划、Gaussian实际数组hash、辅助队列及每步输入身份在 `registration/`、`manifests/` 和完整训练CSV。\n',
      '### 接线数值检查与工程修复\n',
      '真实预检前向逐位一致，8微批主FM归一残差全部0；T/L合成梯度相对L2差为0.00037981/0.00669546，余弦0.9999999282/0.9999776751，最大绝对差3.0518e-5/3.3490e-4。原加权aux在BF16反向中引入系数，新载体在未加权反向后于FP32乘系数，数学相同而舍入位置不同。预先冻结的combined相对L2≤.01、cos≥.9999、aux相对L2≤.05、归一残差≤1e-8均通过，未在看到结果后放宽。隔离副本真实更新范数0.02614091，相对参数范数2.3500e-5；之后原参数和AdamW均恢复。\n',
      '保留的工程故障：注册hash误含正在写入的log；CPU测试初稿的1ULP期望与重复JSON参数；总入口冻结前的账本缩进错误；原START硬链接别名不被历史严格加载路径识别（0模型调用），改用同SHA原路径恢复；最终相机冻结时Python tuple/JSON list比较不等，做JSON规范化后恢复，冻结相机文件字节SHA仍完全相同。没有改模型数学、loss、rho、数据、评分定义或相机几何。原件、回执及diff保存在 `engineering/`、`preflight/`、`failures/`。\n',
      '文献仅作研究背景与边界说明，见 `LITERATURE_SCOPE.md`；未运行这些方法，不声称完整算法复现或套用理论保证。\n',
      (out/'report_interpretation.md').read_text(encoding='utf-8'),
      '\n## 最终梯度与真实更新证据\n']
    phases=rows(out/'training_diagnostics/final/phase500_summary.csv')
    parts.append(table(['臂/阶段','loss cap%','grad cap%','clip%','aux/FM均值','aux/FM中位','cos均值','真实更新均值','真实更新快照'],[
        [r['arm']+'/'+r['phase'],f"{100*float(r['loss_cap_fraction']):.1f}",f"{100*float(r['gradient_cap_fraction']):.1f}",
        f"{100*float(r['total_clip_fraction']):.1f}",f"{float(r['actual_weighted_aux_FM_norm_ratio_mean']):.6f}",
        f"{float(r['actual_weighted_aux_FM_norm_ratio_median']):.6f}",f"{float(r['cosine_mean']):+.6f}",
        f"{float(r['actual_update_norm_mean']):.8f}",r['actual_update_snapshot_count']] for r in phases]))
    parts += ['\nT在本新流实际1000步均未触发loss cap。L后半程aux/FM均值1.677667、最大10.80486，说明loss cap不等于梯度cap。G总1000步触发455次，理论预算最大残差0；实际FP32乘法/范数归约有104步正残差，最大1.1920928955e-7，实际比最大1.0000001781，不能写成浮点逐位严格≤1。\n',
      '42对注册的真实参数更新中，G比L小37次、大4次、相等1次；G/L比均值0.948312、中位0.955519；前后半中位0.977050/0.936787。step501 G/L=0.02970094/0.03424206；step1000=0.02810124/0.02982294。它是特定配对快照的实测结果，不是所有步更新界，也不能由这个单一新流作普遍因果推断。\n',
      '全部步额外主梯度缓冲521604620 bytes（497.441MiB），同样用于T/L/G；各臂/半程capture均值10.50–11.22ms，scalar reduction14.13–14.43ms，combine13.36–13.85ms，载体总37.99–39.35ms。原始LF、raw LA、加权LA、系数、范数、夹角、clip、输入/辅助身份、有限性与阶段时间均在3000行逐步表。\n',
      '![完整训练诊断](delivery_compact/training_overview.webp)\n',
      '## 全量实际视觉审阅\n',
      '标识为 **AI_VISUAL_REVIEW**。288/288输出均实际查看了同父共同相机下四视角+wire和S/I/E叠图；不是生成图片后直接打勾。四个并行审阅范围各72条，逐条记录合并去重后为288。GT及输出均为C蓝/free橙，未平滑、补洞、裁异常或改变几何。共同相机含全部标签/噪声的所有顶点；OLD/FRESH各800个原mesh面板（含C与GT）及432个叠图，全部顶点入镜和几何未变核验PASS。\n',
      '完整本地图集入口：[REVIEW_INDEX.html](visuals/REVIEW_INDEX.html)。逐输出观察为 `visuals/reviews/all_output_observations.csv`，每条有请求、标签、主体/重叠/薄片/合理性、确定程度及图页hash。没有人类用户研究；不报告人类可用率。图像能确认层叠、尖片和碎片化外观，不能单凭投影把所有疑似重叠断言为精确共面重复三角面。\n',
      '明确坏例：OLD 341b/n1/G500相对START/T500/L500的座下承托变成尖碎片和交叉长三角；G1000恢复较完整箱式外形。FRESH 534b/n2，历史CAPPED500与G500的free支撑进一步缩成悬空碎片，L1000呈错位板与纠缠箱状承托；G1000恢复更多高背/腿，但仍有大尖片。FRESH 341b/n3的历史CAPPED500也出现相对START/历史teacher明确新增的座面/下部碎片化与背部裂解。上述判断依据实际碎片化与支撑不连贯，不以椅型、背高或腿数不同于GT为失败理由。\n',
      '其他父也保留：866e的FRESH局部座面交叉片/尖片，86ec/n2/G1000新增腿间大三角薄片；bdd历史CAPPED OLD n1腿间薄片变大；f596历史CAPPED FRESH n2背部不规则孔；c1b FRESH n2各标签主体基本保持。确定的局部缺陷与是否构成重大主体失败分开写，possible不偷换成confirmed。T自身的座面尖片、悬挂片和START已有缺陷也记录在案。\n',
      table(['面板','候选/匹配T','视觉判断','实际依据'],[[r['panel'],r['candidate']+'/'+r['matched'],r['status'],read(r['path'])['rationale']] for r in visual['comparisons']]),
      '\n视觉INCOMPLETE表示全量已看完但重大缺陷程度仍有不确定性，不表示漏看请求；它不会被当成PASS。视觉PASS也仅指本次比较未见明确新增重大主体缺陷，不保证无裂缝/相交或人类可用。数值联合门槛保持独立。\n',
      '![500步控制与覆盖方向](delivery_compact/primary500_control_and_shape.webp)\n![持续训练变化](delivery_compact/stability500_to1000.webp)\n## 完整结果表\n', (out/'report_tables.md').read_text(encoding='utf-8'),
      '\n## 严重父控制代价\n', (out/'severe_parent_costs.md').read_text(encoding='utf-8'),
      '\n## 检查点与导出合同\n',
      '每个500/1000权重保存参数、完整AdamW、RNG、主流游标、辅助next step、recipe和protocol hash；500=conditional_total5500，1000=6000。每100步恢复保存不增加评价终点；最后latest恢复状态仍保留，不据其挑最好分数。以下为源文件真实SHA；详细模型/优化器/RNG/游标hash见 `audit/checkpoint_manifest.csv`。\n',
      table(['标签','条件总步','bytes','SHA256'],[[r['label'],r['conditional_total_step'],r['bytes'],r['actual_file_sha256']] for r in checkpoints if r['role']!='ROLLING_RECOVERY']),
      '\n所有输出N、K按输入固定，N范围129–246、K26–49，平均free面数149.625；288例的51个模型坐标状态中C逐位保持。最终NPZ以及导出PLY重读核对C坐标、定向面多重集（包括重复面）、映射和绕序，全部PASS。模型坐标FP32路径合同与world导出坐标合同分开核验，不声称任意文件格式逐字节相同。逐输出回执在 `audit/runtime_final_outputs.csv`。\n',
      '## 文件导航与复核\n',
      table(['材料','路径/用途'],[
        ['冻结定义','protocol.json、hypotheses.json、recipes/、task_contract.txt'],
        ['输入及身份','registration/、manifests/；训练/辅助/OLD/FRESH Gaussian hash'],
        ['逐步梯度与实际更新','training_diagnostics/final/per_step_training.csv、actual_parameter_updates.csv、G_vs_L_actual_updates.csv'],
        ['严格日志核验','training_audit/final/、audit/runtime_final.json'],
        ['全部288原逐输出评分','evaluation/{label}/{OLD,FRESH}/per_output_metrics.csv，18份各16行'],
        ['配对/逐父/噪声/留一父','analysis/per_output.csv、per_parent.csv、per_noise.csv、leave_one_parent_out.csv、summary.csv；ALL四噪声也保留'],
        ['持续训练及DID','analysis/drift_difference_in_differences_{per_output,per_parent,per_noise,summary}.csv'],
        ['原双参考晋级裁判','analysis/original_joint_gates/；原定义与严重父flag不改'],
        ['可复核数值图','report_figures/：500效应、持续训练图，配套displayed_values.csv及hash'],
        ['全量图与AI观察','visuals/REVIEW_INDEX.html、visuals/reviews/；完整图本地保留'],
        ['实际源码与差异','review_sources/、SOURCE_HASHES.json、code.diff；含原评分依赖及新梯度实现'],
        ['最小入口与恢复','REPRODUCE.md；run/worker/evaluation/analysis/audit模块'],
        ['紧凑审阅包','delivery/CAPPED500_REPLICATION_GRADIENT_BUDGET_REVIEW.zip；REVIEW_INVENTORY.json逐文件hash和CSV行数'],
      ]),
      '\n审阅ZIP保留全逐例/逐父/逐噪声CSV、逐步梯度和真实更新、实际评分源码、协议、审计与数值图及接触图（无损WebP，解码RGBA像素逐字节相同）；为保持紧凑不装RAW、大权重或全分辨率图集。完整RAW、权重、原PNG图集、逐mesh绘图回执和逐调用原始账本均保留本地；ZIP的delivery_compact/保存全图集manifest摘要及原文件hash，不含重复QA裁片；ZIP不是脱离登记资产就可独立重现模型的环境。未清理受保护资产，未切换默认模型，未自动Git提交或推送。\n']
    target=out/'CAPPED500_REPLICATION_GRADIENT_BUDGET_REPORT.md'
    target.write_text('\n'.join(parts),encoding='utf-8')
    links=[]
    for panel in ('OLD','FRESH'):
        latest=read(out/'visuals'/f'latest_{panel}.json')
        p=Path(latest['index']).relative_to(out/'visuals').as_posix()
        links.append(f'<li><a href="{html.escape(p)}">{panel} full gallery: 144 reviewed outputs</a></li>')
    (out/'visuals/REVIEW_INDEX.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Completed AI visual review</title><style>body{font-family:system-ui;max-width:1000px;margin:40px auto;line-height:1.7}</style><h1>AI_VISUAL_REVIEW — 288/288 complete</h1><p>All 9 labels, 8 parents, 4 fixed noises; four views + wire and S/I/E overlays actually reviewed. Human study NOT_RUN.</p><p>Original frozen gallery pages retain their creation-time PENDING note. This index and reviews/completion.json record the later completed review; original images/manifests were not rewritten. Possible findings are not promoted to confirmed body collapse. No smoothing, geometry correction or anomaly crop.</p><ul>'+''.join(links)+'</ul><p><a href="reviews/all_output_observations.csv">All per-output observations CSV</a> · <a href="reviews/completion.json">Completion and comparison receipts</a> · <a href="../CAPPED500_REPLICATION_GRADIENT_BUDGET_REPORT.md">Main report</a></p></html>',encoding='utf-8')
    return dict(status='PASS',path=str(target),bytes=target.stat().st_size,model_calls=0,scoring_calls=0)

if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False))
