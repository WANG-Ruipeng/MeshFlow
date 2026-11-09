"""Build final Chinese report from completed evidence plus an explicit root assessment.
No model calls, metric recomputation, automatic promotion, or premature conclusions.
"""
from pathlib import Path
import json
from . import data as D

NAMES={'BASE':'FULL_S6000','NFT':'NFT','SELF_FM':'SELF_REFLOW'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def mean(g,key):return g['metrics'][key]['equal_parent_distribution']['mean']
def number(x,percent=False):return 'NA' if x is None else f'{x*(100 if percent else 1):.3f}'
def event(g,key):
 e=g['events'][key];return f"{e['YES']}/{e['total']} (未知{e['UNKNOWN']})"
def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
 out=D.OUT; final=read(out/'evaluation/analysis/summary.json')
 if final['status']!='COMPLETE':raise RuntimeError('Final evaluation incomplete')
 decision=read(out/'report_decision.json')
 if decision.get('status')!='ROOT_REVIEW_COMPLETE':raise RuntimeError('Actual root assessment required')
 candidates=[]
 for p in (out/'learning_analysis/snapshots').glob('*/summary.json'):
  v=read(p)
  if v.get('status')=='COMPLETE':candidates.append((p.stat().st_mtime,p,v))
 if not candidates:raise RuntimeError('Complete training evidence missing')
 _,lp,learn=max(candidates)
 audit=read(out/'audit/latest.json')
 if audit.get('status')!='COMPLETE':raise RuntimeError('Final execution audit not passed')
 ap=D.resolve(audit['summary_path'])
 if D.sha(ap)!=audit['summary_sha256']:raise RuntimeError('Audit pointer hash differs')
 au=read(ap)
 if au['status']!='COMPLETE' or au['budget_checks']['status']!='PASS' or au['protected_inputs']['status']!='PASS' or au['violations'] or au['partial_reasons']:raise RuntimeError('Final audit incomplete or failed')
 groups=final['groups'];pairs=read(out/'evaluation/analysis/paired_summary.json')
 rows=[]
 for noise in ('OLD','FRESH'):
  for split in ('TRAIN','DEV'):
   for label in ('BASE','SELF_FM','NFT'):
    g=next(x for x in groups if (x['noise_partition'],x['split'],x['label'])==(noise,split,label))
    rows.append([noise,split,NAMES[label],event(g,'RAW_adoption_known'),str(g['POST_interface_outputs'])+'/'+str(g['outputs']),event(g,'joint_RAW_adoption_POST_legal_CF'),event(g,'joint_both_adoptions_body_and_CF')])
 quality=[]
 for noise in ('OLD','FRESH'):
  for split in ('TRAIN','DEV'):
   for label in ('BASE','SELF_FM','NFT'):
    g=next(x for x in groups if (x['noise_partition'],x['split'],x['label'])==(noise,split,label))
    quality.append([noise,split,NAMES[label]]+[number(mean(g,'POST_'+k),True) for k in ('CF_illegal_fraction','FF_illegal_fraction','I_lower','I_upper','free_small_angle_fraction','free_retained_fraction')])
 paired=[]
 for p in pairs:
  if p['noise_partition'] not in ('OLD','FRESH') or p['split'] not in ('TRAIN','DEV'):continue
  paired.append([p['noise_partition'],p['split'],NAMES[p['candidate_model']]+' − '+NAMES[p['reference_model']],p['joint_gained'],p['joint_lost'],p['joint_unknown_pairs']]+[number(p['differences']['POST_'+k+'_delta']['equal_parent_distribution']['mean'],True) for k in ('CF_illegal_fraction','FF_illegal_fraction','I_lower')])
 reward=read(lp.parent/'reward_conditions.json');bycondition={}
 for r in reward:bycondition.setdefault(r['condition_id'],[]).append(r)
 conditions={r['condition_id']:r for r in D.read(out/'inputs/manifest.json')['train_conditions']}
 reward_rows=[]
 for cid,rs in sorted(bycondition.items()):
  reward_rows.append([rs[0]['parent_id'][:6],conditions[cid]['K'],sum(r['count'] for r in rs),sum(r['RAW_adoption__YES'] for r in rs),sum(r['reward_unknown'] for r in rs),sum(r['preference_known'] for r in rs),str(sum(r['preference_known']>0 for r in rs))+'/'+str(len(rs)),sum(r['strict_joint_pass_count'] for r in rs)])
 totals=learn['totals'];search=learn['frozen_FULL_round00_search']
 text="""# R2_TERMINAL_REWARD_NFT_V1 实验报告

{conclusion}

本轮固定任务已实际完成。主结果只使用预定 round15，两臂各256次逻辑 AdamW 更新；没有挑 best checkpoint、切换默认模型、commit 或 push。

## 1. 实际比较的是什么

从同一 FULL_S6000 live 生成器（local6000/global12000，非EMA）开始，冻结主干原参数、role与Geo，只训练48个主干线性层的 rank4、alpha4 LoRA，共589,824参数。两臂使用各自 fresh AdamW，lr=1e−5、microbatch1、累积8、梯度裁剪1；BF16前向，FP32参数与Euler积分，TF32关闭、SDPA math、严格确定性运行。

- NFT：当前策略每轮4条TRAIN C×8份固定噪声，共16轮512份RAW；单次官方D0、GEO_C及来源定向导出后，结合实际AI_PROXY用途评阅与几何量分配终态奖励。
- SELF_REFLOW（代码标签SELF_FM）：使用同一批NFT生成端点、同一重加噪/时间/曝光流，做无奖励的归一化端点回归及相同参考约束。它是共享在线端点池的 off-policy 对照，不是原来源GT上的OT/HYBRID普通续训，也没有自己的生成池。
- 冻结FULL_S6000：原24份RAW/POST复用；另生成预先冻结的12份新噪声输出。每个最终模型均评价同一36请求。

训练条件从47条TRAIN FULL按输入K、面积和轮廓特征预选8条，未按新生成结果挑父。每条64次在线生成、均等轮次；85b169也在TRAIN评价面板中，不能把该父称为未见训练父。DEV不进入奖励训练。

所有采样仍是原Euler50；奖励对RAW对应的终态POST评分，学习端点保持原N−K个RAW free面。没有把删面后的POST塞回定长训练数组，没有新绑定、RBF、B、READOUT、蒸馏、去噪器适配或多次去噪。

## 2. 主结果：新旧噪声和TRAIN/DEV分开

“联合”沿用 RAW实际采用YES 且 POST至少一条合法C/F共边；“严格联合”还要求POST采用YES和主体完整YES。未知单列，不能默认填NO；一个接口也不等于整圈接好或主体可用。最终108份输出均实际匿名图审，AI_PROXY，非人类金标准。

{main_table}

{observations}

## 3. 逐请求配对与几何代价

下表的获得/丢失只统计双方判断明确的配对；有未知的配对单列。CF/FF/I差值为百分点，先在父内平均再等权聚合父；负值表示相应冲突量下降。样本很小，不作统计显著优越的结论。

{pair_table}

POST原指标，单位为百分比；I保留原ConflictCov_C下/上界，不能把下界当精确碰撞体积。小角面仅作技术观察。CF/FF分母为POST实际free面数，free保留率分母为原N−K；I为C上的冲突覆盖上下界。

{quality_table}

{costs}

所有RAW/POST原分数、逐父与逐请求差值、未知状态都在 evaluation/analysis，未按结果删行。RAW完整指标也随包保存，不用POST均值掩盖生成阶段的变化。

## 4. 奖励与学习曲线

同条件8份输出中，实际评阅的RAW采用、POST采用、POST主体完整全部YES才通过用途门；任一NO给零任务奖励，任一UNKNOWN弃权。硬错误为C未保持或非有限数值。通过用途门的奖励为：

`(.25 + .25×has_legal_CF + min(legal_CF_length / C_open_perimeter, 1)) × (1−CF)×(1−FF)×(1−I)×(1−near_degenerate_fraction)×free_retained_fraction^0.25`。

这是一组预先冻结的pilot权重，没有调到最优；全部C开放边仅作描述性长度归一化，不表示每条边都必须封闭。组内均值/总体标准差归一化后截断优势到±2，映射r=.5+adv/4。少于两个已知结果或方差过小的组没有偏好项，保留参考约束。

NFT采用独立Gaussian重加噪 `x_t=(1−t)epsilon+tF_RAW`，与原OT/HYBRID路径不同；正负隐式分支以轮初旧策略为参照，beta=.1，按free端点误差的stop-gradient MAE归一化。参考项系数.01，是速度场MSE，不是精确分布KL。旧策略每轮开始精确快照，轮内固定。

{learning_observations}

下表列出全部8条训练条件的实际奖励覆盖，属于在线训练数据，不是固定测试成功率。均等生成次数不等于均等有效偏好监督。

{reward_table}

训练图审非盲，最终评阅者与训练评阅者分离且模型标签匿名，但仍属于同类AI评阅者；因此不能据此认证人类偏好提升。训练在线奖励面对不断变化的策略、交替条件组和不同固定噪声，不能把上涨曲线当固定测试收益。

## 5. Best-of8只能说明多少

冻结FULL的搜索对照仅覆盖round0四个TRAIN条件、32份输出。同组8份均匀抽取的平均奖励为{random_reward}，八选一平均为{best_reward}，差值{search_gain}。它是同批候选上的选择收益，不是模型已经学会；其余轮的Best-of8来自在线NFT，不是冻结FULL。没有为DEV额外生成八选一搜索池。

## 6. 真实预算、工程事件与复现

- 新增RAW596份：训练512＋最终84；最终另复用FULL旧24份，形成108份配对模型输出。
- Euler生成29,800次前向，额外1次零LoRA一致性验证；单次D0共596次。
- 两臂逻辑提交各256更新，共512。正常训练前向NFT为6144次、SELF_REFLOW为4096次；它们匹配更新/端点预算，不是相同前向计算量。加上恢复后，实际训练前向{physical_forwards}次；物理完整更新{physical_updates}次，更新尝试{physical_attempts}次。
- round0发生记录函数重复关键字错误：NFT16更新及检查点已保存；SELF进程被控制器退出打断。修复日志与等待兄弟进程逻辑后，NFT只做CPU加载核对，没有重训；SELF仅执行一次已记录的工程恢复。额外14完整更新、15尝试、228训练前向计入物理成本，不能伪称完全等于原计算预算。
- SELF恢复前14步的实际loss与裁剪前梯度范数逐值一致；原中断参数状态没有完整保存，不能扩写成原全状态逐位一致证明。原失败日志、恢复授权、源码版本和diff都保留。

资源按实际阶段测量后逐个启动：采样按请求分片并行，两训练臂独立并行；每轮仍先完成采样、评分和图审再更新，官方D0单实例处理整批，CPU评分2worker且限制线程；保留至少max(2GiB,10%总显存)余量，不停止用户进程。WDDM部分进程显存为NA，报告保留全GPU快照、CUDA allocated/reserved及测量范围，不把权重文件大小当训练显存。并发墙钟不作孤立吞吐比较。早期部分记录缺少CPU负载，被中断SELF原进程未保存完整资源轮询；这些观测缺口在`resource_summary/latest.json`指向的逐worker表中保留。

完整训练审计：`{learning_path}`；执行审计：`audit/latest.json`。受保护原资产、原数据/RAW/评分定义按hash核验；所有新实现和改动都在独立pilot目录。

## 7. 结论与下一步

{decision}

这是小规模终态奖励后训练验证，覆盖8条TRAIN条件、已反复观察的6个评价父及新增噪声；它不能单独证明新父普遍泛化、RL整体可行/不可行、真实物理可靠性或奖励最优性。固定任务到此停止。

## 8. 交付定位

- 两臂最终可恢复LoRA：`checkpoints/NFT/round15.pt`、`checkpoints/SELF_FM/round15.pt`，各自配套JSON包含hash；大FULL基座继续外置且受保护。
- 冻结数据/噪声/初始化：`inputs/manifest.json`；注册合同：`registration.json`。
- 训练512份RAW、POST、实际图审与奖励：`training/round_00`至`round_15`。
- 最终全组匿名图集：`evaluation/blind_gallery/index.html`；真实评阅：`evaluation/blind_gallery/reviews_actual.json`。
- 全逐输出/逐父/逐请求配对CSV与多样性描述：`evaluation/analysis/`。
- 可恢复入口：`r2/pilots/terminal_reward_nft_v1/resume.ps1`（默认只读status）；最终阶段：`python -m r2.pilots.terminal_reward_nft_v1.final_run --stage all`，完成记录复用，未完成模型调用不自动重放。
- 小审阅包的清单、源代码hash与code.diff见`evidence_pack/`；大权重和全RAW留本地，不能把小包冒充完整训练资产。

方法依据：[DiffusionNFT论文](https://arxiv.org/abs/2509.16117)、[官方训练代码](https://github.com/NVlabs/DiffusionNFT/blob/main/scripts/train_nft_sd3.py)。本地适配与论文实现的差异如上，原始代码和版本hash随证据保存。
"""
 text=text.format(reward_table=table(['训练父','K','生成数','RAW采用YES','奖励未知','参与偏好样本','有区分组','严格联合训练样本'],reward_rows),conclusion=decision['conclusion'],main_table=table(['噪声','父分组','模型','RAW采用','POST有合法接口','联合','严格联合'],rows),observations='\n\n'.join(decision['observations']),pair_table=table(['噪声','分组','比较','联合获得','联合丢失','未知配对','ΔCF pp','ΔFF pp','ΔI下界 pp'],paired),quality_table=table(['噪声','分组','模型','CF%','FF%','I下界%','I上界%','小角%','free保留%'],quality),costs='\n\n'.join(decision['costs']),learning_observations='\n\n'.join(decision['learning_observations']),random_reward=number(search['mean_condition_random_eight_reward']),best_reward=number(search['mean_condition_best_reward']),search_gain=number(search['mean_bestof8_gain']),physical_forwards=totals['physical_forward_returns'],physical_updates=totals['physical_completed_updates'],physical_attempts=totals['physical_update_attempts'],learning_path=str(lp.parent.relative_to(out)),decision=decision['next_step'])
 p=out/'REPORT.md'
 if p.exists():
  if p.read_text(encoding='utf-8')!=text:raise RuntimeError('Refuse overwrite completed report')
 else:p.write_text(text,encoding='utf-8')
 D.write(out/'report_receipt.json',{'status':'COMPLETE','report_sha256':D.sha(p),'decision_sha256':D.sha(out/'report_decision.json'),'final_summary_sha256':D.sha(out/'evaluation/analysis/summary.json'),'learning_summary_path':str(lp),'learning_summary_sha256':D.sha(lp),'audit_latest_sha256':D.sha(out/'audit/latest.json'),'builder_sha256':D.sha(__file__),'model_calls':0})
 print(str(p))
if __name__=='__main__':main()
