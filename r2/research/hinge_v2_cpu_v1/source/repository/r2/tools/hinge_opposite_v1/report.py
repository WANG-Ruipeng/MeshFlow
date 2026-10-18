"""Build the evidence-backed closeout after completed scoring and independent audit."""
from pathlib import Path
import collections, csv, difflib, hashlib, json
from .analyze import OUT, ROOT, read

def main():
    audit=json.loads((OUT/'independent_audit.json').read_text(encoding='utf-8'))
    assert audit['cells']==56 and audit['new_new_same_direction_edges']==0
    summary={(r['group'],r['metric']):r for r in read(OUT/'metric_summary.csv')}
    led=read(OUT/'connection_ledger.csv');funnel=read(OUT/'rejection_funnel.csv')
    stages=['POST','HINGE_OPP015','HINGE_OPP025']
    lines=['# PLAIN_S2000 + HINGE：全部新增双面边反向守卫复验',
        '', '**决定：保留 HINGE_OPP015（1.5%L），保留 HINGE_OPP025（2.5%L）作对照，归档为有限但明确的后处理正结果。** 原 PLAIN_S2000、旧 HINGE、全部 RAW/POST 与评分结果保留。当前证据不支持“所有指标严格零退化”，也没有解决 DEV 的大面积 seam→bulk 接续。',
        '', '## 范围与唯一变化', '',
        '固定同一28个保存请求：4个熟悉 TRAIN 父×4噪声=16、3个反复查看过的 DEV 父×4噪声=12。沿用 PLAIN_S2000 的单次官方 D0、GEO_C、来源定向 POST。两个位移上限分别独立从原 POST 运行，未串联其他绑定。零训练、零模型前向、零新噪声、零重新 D0；本轮56次新CPU后处理与原评分。', '',
        '旧实现已检查被提名 hinge 边的反向绕序，但整簇合并还会附带产生其他双面共边。新 `require_opposite_winding=True` 对当前→试算状态中**全部新增 incidence=2 边**检查两半边是否反向，覆盖附带 CF/FF 和几何不合法的边。身份使用原 source_face_id/source_corner_id，避免代表顶点编号改变而误判旧边。出现任一新增同向边即回滚整次合并。', '',
        '本轮不翻面、不删面、不新聚类；旧同向缺陷保持原身份。候选、顺序、整簇 PRE_GEO 累计位移、形状/法线、零新增非法面对、旧合法边保护均沿用旧定义。因此“修掉3条”是拒绝引入它们的合并，并非强行给旧产物翻面。025原有4条，两个设置共7格受影响。', '',
        '## 全组主要指标', '',
        '以下比例为逐输出宏均值，28条ALL按16 TRAIN/12 DEV加权；单面孤岛、合法共边计数为全组总和。原124数值字段＋11版本化拓扑字段全部保存在CSV，不只报告有利字段。', '',
        '| 指标 | 原 POST | OPP 1.5%L | OPP 2.5%L |', '|---|---:|---:|---:|']
    metrics=[('FF受非法面对影响的free面比例','FF_illegal_affected_free_fraction','pct'),('CF受非法面对影响的free面比例','CF_illegal_affected_free_fraction','pct'),('I：固定C查询冲突覆盖','ConflictCov_C','pct'),('free小角面<5°','free_minimum_angle_lt5_fraction','pct'),('free q>10比例','free_q_gt10_fraction','pct'),('free平均q','free_q_mean','float'),('单面孤岛总数（不滤非法边）','hinge_diag_isolated_1face_unfiltered_edge_components','sum'),('全部合法图共边总数','graph_valid_edges','sum'),('合法CF边总数','C_F_valid_edge_count','sum'),('bulk可达面积均值','free_only_reached_bulk_area_fraction','pct'),('bulk可达≥30%输出数','bulk_reach_ge_0_30_descriptive','count'),('同向双面边总数（含旧缺陷）','EXACT_COORD_same_direction_two_face_edges','sum'),('非流形边总数','nonmanifold_edges','sum')]
    for label,key,kind in metrics:
        r=summary['ALL',key];vals=[]
        for s in stages:
            v=float(r[s+('_sum' if kind in ('sum','count') else '_mean')])
            vals.append(f'{v*100:.4f}%' if kind=='pct' else f'{v:.0f}/28' if kind=='count' else f'{v:.0f}' if kind=='sum' else f'{v:.5f}')
        lines.append('| '+label+' | '+' | '.join(vals)+' |')
    lines += ['', '主档逐请求：FF改善11、持平17、恶化0；CF改善2、持平26、恶化0；单面孤岛减少15、持平13、增加0；合法共边增加23、持平5、减少0。单面孤岛指实际索引共边图的单面分量，不把共一个顶点或单张截面的散线段自动当作接合。', '',
        '## TRAIN与DEV必须分开', '', '| 分组／指标 | 原 POST | OPP 1.5%L | OPP 2.5%L |','|---|---:|---:|---:|']
    for group in ('TRAIN','DEV'):
        for label,key in [('FF','FF_illegal_affected_free_fraction'),('CF','CF_illegal_affected_free_fraction'),('I','ConflictCov_C'),('bulk可达面积','free_only_reached_bulk_area_fraction')]:
            r=summary[group,key];lines.append('| '+group+' '+label+' | '+' | '.join(f'{100*float(r[s+"_mean"]):.4f}%' for s in stages)+' |')
    lines+=['', 'TRAIN 的 bulk≥30% 从13/16提高到14/16；DEV两个设置仍为0/12。1.5%L的DEV bulk可达仍为0，2.5%L仅0.0336%宏均值。局部闭口/孤岛减少可成立，但不能宣称DEV整体接续突破。A-VMR/PMR保持原值；不把它们当作整段接口完成率。', '',
        '## 真实连接账本与守卫代价', '', '| 来源身份配对账本（28请求） | OPP 1.5%L | OPP 2.5%L |','|---|---:|---:|---:|']
    for title,key in [('保留原合法CF边','CF_retained_legal_edges'),('保留原合法FF边','FF_retained_legal_edges'),('新增合法CF边','CF_new_legal_edges'),('新增合法FF边','FF_new_legal_edges'),('消除旧非法CF面对','CF_illegal_pairs_removed'),('消除旧非法FF面对','FF_illegal_pairs_removed'),('丢失旧合法CF边','CF_lost_legal_edges'),('丢失旧合法FF边','FF_lost_legal_edges'),('新增非法CF面对','CF_new_illegal_pairs'),('新增非法FF面对','FF_new_illegal_pairs'),('删除来源自由面','deleted_source_faces')]:
        vals=[sum(int(float(r[key])) for r in led if r['label']==s) for s in stages[1:]]
        lines.append(f'| {title} | {vals[0]} | {vals[1]} |')
    decisions={}
    for s in stages[1:]:
        c=collections.Counter()
        for r in funnel:
            if r['label']==s:c.update(json.loads(r['decisions']))
        decisions[s]=dict(c)
    (OUT/'rejection_totals.json').write_text(json.dumps(decisions,indent=2),encoding='utf-8')
    lines += ['', '主档旧HINGE净增111条合法边，新守卫后105条；025从132变124。收益略减是显式守卫代价：旧015单面孤岛118，新015为121；旧015 FF13.4638%，新015为13.4902%。没有把这些差异藏成“同等收益且无损修复”。', '',
        '独立审计逐步精确回放176次接受操作：所有新增双面边均反向，原4624条合法边全部保留，零新非法CF/FF面对，零新增删面、重复面或非流形边，C与seam锚逐位不变，来源面/角点身份完整。旧015的3条、旧025的4条新增同向边均不再产生；原网格已有209条同向边仍保留，未认证全网格可定向或水密。', '',
        '## 必须保留的反面结果', '',
        '- `610b1621_n1` 的I仍增加0.048828个百分点（固定2048个C查询中的1点；DEV宏均值增加0.004069个百分点）。零新非法面对不保证相同面对覆盖区域不扩大。',
        '- 主档平均q有5/28请求略差，最坏`608b46b0_n0` +0.009096；该例角度p05下降0.044428°。025有4例平均q略差，最坏`3f741759_n1` +0.020713，其角度p05下降0.106664°。小角率、q>10比例未因此变差，但不能用阈值计数遮住连续指标代价。',
        '- 原共边保留不保证可达面积比例逐例不降，移动改变分子/分母。主档方向可达面积比例9例微降，最坏`2d/n3` -0.083130个百分点；应区分可达面身份与面积变化。',
        '- 本轮28请求属于已反复查看的开发证据，不能作为独立新父泛化证明。没有唯一来源GT排名，也没有新的人类盲评。图审标为AI_PROXY。', '',
        '## 执行、验证与保留方式', '',
        '实测RTX5080当前空闲显存约13GiB，但GPU工作数为0。第一条授权请求测得CPU峰值RSS约106MiB，24逻辑CPU、可用内存约61GiB、低系统负载后启用4个独立CPU进程（每进程1线程）。56格执行127.06秒；工程测试、审计、图集与归档另外计时，不能把这当作GPU吞吐。原1172个保护资产hash不变。', '',
        '工程检查包括关闭守卫逐数组/候选记录复现旧608/n0结果、全部4个已知旁生绕序反例、恒等/已有缺陷等构造检查。实际独立审计不调用新guard辅助函数；终态非法面对基于原冻结完整评分，未重新认证连续运动无碰撞。独立统计复算157,410项检查一致，22,680条配对值及全部退化记录完整。', '',
        '实际固定截面图审：28张图全部生成，7个预定病例的4列均由AI_PROXY实际查看，其余21张未宣称已评阅。73/n3侧截面、8c/n0前截面的局部缺口改善保留；四个绕序受影响例未见明显新增主体破损，但不能把计数变化冒称肉眼改善。608/n3的分段、错层仍可见。图审非盲、非人类评分，不能认证三维拓扑。', '',
        '保留研究组合：`PLAIN_S2000 → 单次D0 → GEO_C/来源定向 → HINGE_OPP015`；2.5%L只保留对照。不删除旧结果，不修改维护CLI默认，不训练新权重。选定组合及适用资产写入 `retained_profile.json`。', '',
        '本地恢复入口：`python -B -m r2.tools.hinge_opposite_v1.run run --workers 4`，只在冻结资产齐全时恢复缺失请求；完整格核验hash后跳过，未完成尝试保留并拒绝静默重试。已完成时不会新增处理。`analyze`重建全135字段配对，`audit`独立核对实际数组，`gallery`生成原位置截面。跨机器CPU数组入口、实际评分源与缺资产行为见发布目录README。', '',
        '## 完整证据文件', '',
        '- `combined_per_output.csv`：140行（原POST、旧015/025、新015/025）×135个数值字段及状态/身份。',
        '- `paired_per_output.csv`：22,680个逐请求逐字段配对值，含新/旧守卫成本；`metric_summary.csv`：405条ALL/TRAIN/DEV汇总；`paired_per_parent.csv`：945条逐父汇总。',
        '- `all_regressions.csv`保留全部不利数值，`connection_ledger.csv`、`rejection_funnel.csv`及`independent_audit/`保留来源面身份账本。',
        '- `gallery/`：同相机同截面位置的28张四列图；图审记录另列，不能把自动指标冒称肉眼观察。',
        '- `protocol.json`、`ledger.jsonl`、资源/完成回执、`code.diff`、实际源文件hash及发布归档。',
        '', '**最终定位：可复现的局部几何与连接改进，新增绕序缺陷得到修正；保留小幅I/形状代价和DEV未接通的局限，不表述为全局无缺陷后处理。**','']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    orig=ROOT/'r2/pilots/seam_hinge_quality_v1/hinge.py';new=ROOT/'r2/tools/hinge_opposite_v1/engine.py'
    (OUT/'code.diff').write_text(''.join(difflib.unified_diff(orig.read_text(encoding='utf-8').splitlines(True),new.read_text(encoding='utf-8').splitlines(True),fromfile=str(orig.relative_to(ROOT)),tofile=str(new.relative_to(ROOT)))),encoding='utf-8')
    profile=dict(status='RETAIN_BOUNDED_POSITIVE',generator='PLAIN_S2000',checkpoint='experiments/r2_seam_pool47_couple_v1/checkpoints/PLAIN3000_step2000.pt',checkpoint_uploaded=False,postprocess=['single_official_D0','GEO_C','source_winding_export','HINGE_all_new_two_face_edges_opposite_v1'],delta_main=.015,delta_comparison=.025,require_opposite_winding=True,old_results_preserved=True,maintained_CLI_default_changed=False,validated_requests=28,DEV_bulk_ge30='0/12',scope='saved source-identity and original-GEO-cluster assets required; not arbitrary mesh-only repair',zero_new_illegal_pairs=True,zero_old_legal_edges_lost=True,zero_additional_deleted_faces=True,all_metrics_zero_regression=False)
    checkpoint=ROOT/profile['checkpoint'];sidecar=checkpoint.with_suffix('.json')
    identity=json.loads(sidecar.read_text(encoding='utf-8-sig'))
    assert checkpoint.is_file() and checkpoint.stat().st_size==identity['bytes']
    profile.update(checkpoint_sha256_recorded=identity['sha256'],generator_state_sha256_recorded=identity['generator_state_sha256'],global_step=identity['global_step'],local_step=identity['local_step'],checkpoint_bytes=identity['bytes'],checkpoint_exists=True,checkpoint_identity_check='Existing frozen JSON identity and current file size; large weight not rehashed or loaded in this CPU evaluation')
    (OUT/'retained_profile.json').write_text(json.dumps(profile,indent=2),encoding='utf-8')
    print(json.dumps(dict(report=str(OUT/'REPORT.md'),decisions=decisions)))

if __name__=='__main__':main()
