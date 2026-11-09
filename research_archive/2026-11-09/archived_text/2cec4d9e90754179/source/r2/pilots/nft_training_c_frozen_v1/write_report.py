"""Render complete paired evidence, using a separately authored decision."""
from .bootstrap import bind,OUT
D,E=bind()
def run():
    s=D.read(OUT/'evaluation/analysis/summary.json');assert s['status']=='COMPLETE'
    decision=D.read(OUT/'report_decision.json')
    pairs=D.read(OUT/'evaluation/analysis/paired_summary.json')
    cond=D.read(OUT/'evaluation/analysis/per_condition.json')
    pc=D.read(OUT/'evaluation/analysis/paired_per_condition.json')
    groups={g['label']:g for g in s['groups']};names={'BASE':'FULL_S6000','NFT':'NFT','SELF_FM':'SELF_REFLOW'}
    def f(x,d=4):return 'NA' if x is None else format(float(x),'.'+str(d)+'f')
    def event(g,k):
        e=g['events'][k];return str(e['YES'])+'/'+str(e['total'])+' (UNKNOWN '+str(e['UNKNOWN'])+')'
    def table(h,rows):
        return '| '+' | '.join(h)+' |\n| '+' | '.join(['---']*len(h))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+'\n'
    text='# R2_NFT_TRAINING_C_FROZEN_V1\n\n'+decision['conclusion']+'\n\n'
    text+='## 合同与实际完成\n\n原八条奖励训练C全部保留，每C四份新噪声，三模型共96份RAW及96份POST。实际4,800生成前向、96单次D0、192原评分单元、32页原图匿名AI_PROXY审阅、96条标签。零训练、零反向、零额外验证前向、零失败重试。这是已获奖励监督C上的新噪声检查，不是新父泛化。\n\n'
    text+='FULL_S6000仍为原live global12000（非EMA）；NFT与SELF均固定round15/256更新终点。原Euler50、BF16前向/FP32参数与积分、确定性math SDPA、单次官方D0/GEO_C/来源C绕序合同不变。三模型共享32份数组，种子和数组hash与旧512训练、36评估请求无重复；八个原2048查询bank直接复用。没有挑checkpoint、改变奖励、条件、噪声或评分定义。SELF_REFLOW继承上一轮使用同一NFT生成端点池与重加噪流的无奖励对照；它是共享端点的off-policy对照，不是原GT上的普通续训，也没有自己的在线生成池。\n\n'
    text+='## 全32请求\n\n'
    text+=table(['模型','RAW采用','POST合法接口','联合：RAW采用且接口','严格联合：两阶段采用+主体+接口'],[[names[l],event(groups[l],'RAW_adoption_known'),str(groups[l]['POST_interface_outputs'])+'/32',event(groups[l],'joint_RAW_adoption_POST_legal_CF'),event(groups[l],'joint_both_adoptions_body_and_CF')] for l in names])
    text+='UNKNOWN不填NO。至少一条合法共边不等于整圈接好或成品可用；图像采用/主体标签不认证几何拓扑。三模型的RAW采用、POST采用与主体标签逐请求完全一致。e6ea5e的采用虽未知，但接口确定为0，因此联合事件确定为0，并非把未知采用填成NO。\n\n## 八条C完整分布\n\n'
    rows=[]
    for cid in sorted({c['condition_id'] for c in cond}):
        g={c['label']:c for c in cond if c['condition_id']==cid}
        rows.append([g['BASE']['parent_id'][:8]]+[event(g[l],'joint_both_adoptions_body_and_CF') for l in names]+[str(g[l]['POST_interface_outputs'])+'/4' for l in names])
    text+=table(['C父前缀','FULL严格联合','NFT严格联合','SELF严格联合','FULL接口','NFT接口','SELF接口'],rows)
    text+='全部采用、主体和数值分布见per_condition.csv，所有噪声配对见paired_per_request.csv；八条C没有筛选删除。\n\n## 固定原始奖励\n\n'
    text+='原样调用geometry_review_reward，不用各模型自行组标准化的r_NFT作比较。共同已知奖励在同一请求上作差，先C内平均再对可用C等权；UNKNOWN原始R保持缺失，并另报全32请求差值的可行区间。这不是统计置信区间，也不是把未知填零。\n\n'
    rows=[]
    for r in pairs:
        rows.append([names[r['candidate_model']]+' − '+names[r['reference_model']],r['reward_common_known_pairs'],r['reward_common_known_condition_count'],f(r['reward_common_known_candidate_equal_condition']['mean']),f(r['reward_common_known_reference_equal_condition']['mean']),f(r['reward_common_known_delta_equal_condition']['mean']),'['+f(r['reward_all_registered_delta_lower'])+', '+f(r['reward_all_registered_delta_upper'])+']'])
    text+=table(['比较','共同已知/32','可用C/8','候选R','参照R','配对ΔR','含未知全32差区间'],rows)
    rows=[]
    for r in pc:
        if r['reference_model']=='BASE':
            e=r['events']['joint_both_adoptions_body_and_CF']
            rows.append([r['parent_id'][:8],names[r['candidate_model']],r['reward_common_known_pairs'],f(r['reward_common_known_delta']['mean']),e['gained'],e['lost'],e['unknown_pairs']])
    text+=table(['C','相对FULL','共同已知/4','ΔR','严格联合新增','丢失','未知配对'],rows)
    text+='## 质量代价\n\n'
    keys=['POST_CF_illegal_fraction','POST_FF_illegal_fraction','POST_I_lower','POST_free_small_angle_fraction','POST_free_retained_fraction']
    text+=table(['模型','CF %','FF %','I下界 %','小角面 %','free保留 %'],[[names[l]]+[f(100*groups[l]['metrics'][k]['equal_condition_distribution']['mean'],3) for k in keys] for l in names])
    text+='CF/FF分母为每POST实际free面数；I沿用ConflictCov_C，不是相交体积。全部192 I状态PASS且上下界一致；小角不自动等于不可用。全部指标和不利尾部均保存。\n\n'+decision['quality_interpretation']+'\n\n'
    text+='## 本次出口与决定\n\n'+decision['exit_interpretation']+'\n\n'
    text+='## 限制、图审和资源\n\n每C仅四噪声，不能宣称统计稳定优势。全部C属于奖励监督范围；不与上轮不同噪声的在线奖励均值混比。图审由独立新上下文AI_PROXY实际逐张查看32页原图，匿名A/B/C逐请求重新排列，未提供私有映射、数值、模型目录和历史标签；96标签与32实际查看回执先封存后解盲。它不是人类评估，薄层和遮挡仍可不确定。view_image曾遇Windows1385，改用提权只读同一原图字节原尺寸展示，保留回执；没有改图、换样本。\n\n'
    text+='RTX5080总显存16,303 MiB；最高并发3采样进程。353次快照整卡峰值10,803 MiB，最低空闲5,177 MiB，利用率最高76%；24CPU核，可用RAM最低57,955 MiB。采样并集187.55秒、D0阶段58.16秒（含加载46.61秒）、CPU247.92秒（2worker×1thread），不含准备、图审和整理，不是孤立吞吐基准。WDDM进程显存NA保留，采样可能漏瞬时尖峰。\n\n独立数值复核14,128项通过、零异常，最大浮点差1.42e−14。预算、配对、RAW全51态C、POST来源C、冻结检查点和来源hash均通过；本轮没有训练更新、默认切换、commit或push，固定检查完成后停止。\n\n'
    text+='## 交付入口\n\n- evaluation/analysis/per_output.csv：96逐输出；per_output_stage.csv：192阶段指标。\n- evaluation/analysis/paired_per_request.csv、paired_per_condition.csv：三组完整配对。\n- evaluation/blind_gallery/index.html：32张原尺寸图；reviews_anonymous_actual.json、view_receipts.json为真实图审。\n- audit/latest.json、numeric_crosscheck/comparison.json：执行/数值独立核验。\n- inputs/manifest.json：C、噪声、查询与权重身份；所有RAW/POST在evaluation各模型目录。\n- README_RESUME.md与r2/pilots/nft_training_c_frozen_v1/resume.ps1：恢复入口，默认status只读。\n- evidence_pack/NFT_TRAINING_C_FROZEN_REVIEW.zip：全表格、全部32页统一预览、实际源码/diff/hash；大网格与原尺寸图保留本地并列外部路径。\n'
    target=OUT/'REPORT.md'
    if target.exists() and target.read_text()!=text:raise RuntimeError('Refuse report overwrite')
    target.write_text(text,encoding='utf-8')
    D.write(OUT/'report_receipt.json',dict(status='COMPLETE',report_sha256=D.sha(target),decision_sha256=D.sha(OUT/'report_decision.json'),analysis_sha256=D.sha(OUT/'evaluation/analysis/summary.json'),source_sha256=D.sha(__file__),model_calls=0))
    print('REPORT_WRITTEN',D.sha(target),flush=True)
if __name__=='__main__':run()
