"""Frozen solver pilot summary; no model or scoring calls."""
from pathlib import Path
import json
from .registration import OUT,ASSETS
from experiments.mf_h1_local0.jedge5_diagnosis.common import read_json,write_json,file_sha256

def tab(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    ev=read_json(OUT/'evaluation/results.json');assert ev['status']=='PASS'
    rows=ev['rows'];by={(r['case_id'],r['method']):r for r in rows};methods=['Euler50','E_quad50','Midpoint25','Heun25','Euler100','Heun50']
    def metric(cid,m,k):return by[cid,m]['scalars'][k]
    rms='remainder_symmetric_RMS_bbox_pct';p95='boundary_p95_bbox_pct';ff='FF_illegal_pair_count';cf='CF_proper_pair_count'
    rt=tab(['方法','NFE','patch0/10401','patch0/10402','patch4/10401','patch4/10402'],[[m,by[0,m]['nfe']]+[f'{metric(c,m,rms):.4f}' for c in range(4)] for m in methods])
    qt=tab(['方法','p4/10401 Γp95','p4/10402 Γp95','p4/10401 FF','p4/10402 FF','p4/10401 CF','p4/10402 CF'],[[m,f'{metric(2,m,p95):.4f}',f'{metric(3,m,p95):.4f}',metric(2,m,ff),metric(3,m,ff),metric(2,m,cf),metric(3,m,cf)] for m in methods])
    oldgeom=read_json(OUT.parent/'jedge5_blackbox_20261002_113600/geometry/endpoint_results.json');gt={r['seed']:r['GT_free_total_area'] for r in oldgeom['rows'] if r['patch']==4}
    at=tab(['方法','10401 free→R12','10401 R12→free','10401面积比','10402 free→R12','10402 R12→free','10402面积比'],[[m]+sum(([f"{metric(c,m,'free_to_R12_RMS_bbox_pct'):.4f}",f"{metric(c,m,'R12_to_free_RMS_bbox_pct'):.4f}",f"{100*metric(c,m,'free_total_surface_area')/gt[by[c,m]['seed']]:.2f}%"] for c in (2,3)),[]) for m in methods])
    cvt=tab(['方法','10401 Γ覆盖@0.001L','10402 Γ覆盖@0.001L','10401 CF受影响free','10402 CF受影响free'],[[m]+[f"{100*metric(c,m,k):.2f}%" for k in ('boundary_coverage_fraction','CF_proper_affected_free_fraction') for c in (2,3)] for m in methods])
    workers=[read_json(OUT/f'raw/case{i}/results.json') for i in range(4)];assert all(w['status']=='PASS' for w in workers)
    timing=dict(shared_lock_wait_seconds=sum(w['lock_wait_seconds'] for w in workers),GPU_worker_wall_excluding_lock_seconds=sum(w['seconds'] for w in workers),
        saved_rollout_wall_seconds=sum(r['wall_seconds'] for w in workers for r in w['rows']),
        model_cuda_event_seconds=sum(r['audit']['model_forward_cuda_event_seconds'] for w in workers for r in w['rows']),
        CPU_scoring_seconds=sum(x['seconds'] for x in ev['scoring_worker_seconds']),
        note='GPU lock waits excluded from worker walls. CPU scoring overlapped GPU work. Per-phase walls and CUDA events are not additive and are not isolated hardware benchmarking.')
    findings=dict(best_followup_candidates=['Midpoint25','E_quad50'],default_changed=False,
        interpretation='Case-specific zero-training gains at equal NFE, with quality tradeoffs. No claim of stiffness diagnosis, convergence, unseen-case generalization, or solved topology.',
        Midpoint25=dict(patch4_RMS_before=[metric(c,'Euler50',rms) for c in (2,3)],patch4_RMS_after=[metric(c,'Midpoint25',rms) for c in (2,3)],
            patch4_FF_before=[metric(c,'Euler50',ff) for c in (2,3)],patch4_FF_after=[metric(c,'Midpoint25',ff) for c in (2,3)],
            normal_FF_before=[metric(c,'Euler50',ff) for c in (0,1)],normal_FF_after=[metric(c,'Midpoint25',ff) for c in (0,1)]),
        E_quad50=dict(patch4_RMS_after=[metric(c,'E_quad50',rms) for c in (2,3)],FF_improvements_all_selected=sum(metric(c,'E_quad50',ff)<metric(c,'Euler50',ff) for c in range(4))),
        Heun25='Not retained as preferred pilot: patch4/10401 is worse than Euler50, normal-case FF/CF deteriorate.',
        Heun50='100 NFE; partial gains versus Euler100, but not a uniform win and not a 50-NFE method.')
    result=dict(status='PASS_EXECUTION',final_audit='final_audit.json',scope='User-selected zero-training solver verification only',
        actual_new_full_generations=sum(w['counts']['complete_rollouts'] for w in workers),actual_model_forwards=sum(w['counts']['model_forward_returns'] for w in workers),
        model_forward_attempts=sum(w['counts']['model_forward_attempts'] for w in workers),optimizer_updates=0,model_backward_calls=0,OT_calls=0,
        failed_model_calls=0,extra_replays=0,geometry_guidance_calls=0,shared_GPU_lock='/tmp/wrp_meshflow_jedge5_diag_gpu0.lock',maximum_concurrent_GPU_workers=1,
        model_state_sha256=ASSETS['J']['state'],findings=findings,timing=timing,
        source_evaluation=dict(path='evaluation/results.json',sha256=file_sha256(OUT/'evaluation/results.json')),
        artifacts=dict(per_output='evaluation/per_output.csv',paired='evaluation/paired_equal_NFE.csv',identity='case_identity_manifest.json',code_diff='code.diff',raw='raw/',plots='evaluation/case0_comparison.png through case3_comparison.png'))
    assert result['actual_new_full_generations']==16 and result['actual_model_forwards']==1000
    write_json(OUT/'results.json',result)
    report=f'''# JEdge5：固定预算、零训练求解器验证

本轮得到实际的零训练改善，但没有得到一个在所有指标上获胜的求解器。相同50 NFE下，Midpoint25在两个异常案例恢复了更多未知区域覆盖；非均匀Euler50也显著改善，且四例FF相交数均减少。正常案例和CF接口仍存在代价，公共默认采样器未改。

## 合同与实现

用户明确选择“先零训练求解器验证，最多16条新生成”。因此只运行4个预选P0案例（patch0/4 × seed10401/10402）与4个固定方法，不训练，不实现CFG/面积引导，不添加loss。共同模型是JEdge5累计3000，state SHA `{ASSETS['J']['state']}`，同C、同原始自由Gaussian；GT只用于离线评分。

- E_quad50：Euler50，t_k=(k/50)^2，50次前向。首dt=.0004、末dt=.0396，**前密后疏**，没有额外末期加密。
- Midpoint25：25个均匀区间，每区间起点和中点各1次前向，共50 NFE，最后查询t=.98。
- Heun25：25个均匀区间，每区间起点和预测终点各1次前向，共50 NFE。
- Heun50：50个均匀区间，共100 NFE；两种Heun都真实执行最后t=1的校正前向，没有截断或省略。

前向BF16、参数和积分FP32；严格确定性、TF32关闭、SDPA math。C在全部接受状态和中间阶段逐位固定，raw已知槽速度不施加到C。所有16条首速度与旧档同t=0查询逐位相同，参数前后hash一致。CPU解析线性ODE夹具的Heun25/50误差比约3.946；这仅验证公式，不证明BF16神经速度场中的二阶收敛。

复用旧BF16 Euler50/100的8条结果，不重新生成对照。50 NFE三候选只与Euler50作主要配对；Heun50与Euler100配对。没有重复另外两个窗口的全FP32 RK4/Dopri5或局部步长误差矩阵。

## 剩余几何：完整生成结果

排除C的剩余对称RMS，单位% bbox，越小越好。全部开发案例均保留。

{rt}

Midpoint25把patch4的10401从4.2320%降至0.8285%，10402从5.0320%降至1.0038%；后者已明显优于旧Euler100的3.5470%，而只用50 NFE。E_quad50对应0.3335%/1.7067%，10401更好、10402不如Midpoint。Heun25在10401上反而从4.2320%恶化到5.4866%，说明“名义二阶必然更好”不成立。

两个正常patch0的Midpoint RMS分别增加约0.0017和0.0044个百分点；E_quad分别+0.0271和-0.0029个百分点。不能只报异常案例恢复而省略正常例变化。

## 覆盖恢复与面积

以下两方向仍使用原独立评价器的固定面积采样，面积比=生成free总表面积/GT未知部分总表面积。面积比用于离线解释，从未输入求解器。

{at}

10402的Midpoint25把覆盖方向RMS从5.9455%降到0.4255%，面积比从62.33%回到约98.25%。这次没有增加面积监督或几何外力，已经能恢复大部分覆盖。因此“观测到小面积”本身不能证明模型因Edge而主动缩面积作弊。面积恢复也不等于拓扑合法；仍须检查接口和相交。

下图按同相机/范围展示六方法的完整输出、free-only和真实GT未知区域覆盖热图。底排保留未被生成表面覆盖的目标区域，固定色阶，原512个GT查询点，不用逐面RMS相加替代总体指标。

![10402 完整输出与覆盖](evaluation/case3_comparison.png)

[10401同口径图](evaluation/case2_comparison.png)；[正常10401](evaluation/case0_comparison.png)；[正常10402](evaluation/case1_comparison.png)。

## 接口和相交：真实代价

Gamma p95单位% bbox；FF为非法free-free pair数，CF为proper known-free穿刺pair数。完整Γ RMS、所有coverage阈值、共面/未决分类等54项原标量全部保留在逐输出表。

{qt}

{cvt}

Midpoint25在两个异常例把FF从151/141降到56/62，Gamma p95从8.1584/9.2485降到0.4084/2.1364。但CF pair都从2增加到7，受影响free比例也增加；正常两例FF从63/52升到85/65。它改善了异常覆盖与很多自由相交，却没有形成全面无代价的改善。

E_quad50在四例均降低FF，异常例为41/111；10402的Gamma尾部仍有4.9025%，覆盖不及Midpoint。其正常10401 CF由0变2，亦非全指标支配。

Heun25在正常例FF均升到105，CF升到24/26，并使异常10401严重恶化，本轮不把它作为优先候选。Heun50用100 NFE，在10402相对Euler100改善，但10401 FF增加、正常例多项质量退化；不能称作更便宜或全面更好的方案。

各方法仍有非法相交，没有水密、流形或零自交保证。正式拓扑认证未执行，标NOT_RUN；原评价器已有的局部FF/CF/共面/退化、精确共边及输入C异常统计仍保留。所有新输出NaN/Inf均为0，数值执行失败0，不代表全部几何指标过关。

## 解释与下一步边界

这轮支持“采样离散方案是可用干预方向”，尤其Midpoint25与E_quad50。尚不能反推出唯一病因是Edge向心力，不能把质量更好等同于更接近精确ODE解，也不能把Heun的变化单独归因于t=1查询；求解公式、阶段状态与时间位置同时不同。

现有证据优先保留Midpoint25和E_quad50作为实验候选：前者在两个异常例覆盖更均衡，后者四例FF均下降。两者只经过一个父对象、两个已观察patch和两个seed，未验证新对象/新上下文；不替换公共Euler50默认、不启动训练侧修复，不从这里自动扩展矩阵。

CFG需要与训练方式匹配；当前官方面数CFG不等同于Native几何上下文CFG。部署也不知道完整GT面积，因此不能直接将其用于推理面积Guidance。总面积增大不保证铺到缺失区域，训练期几何链式因子也不能解释成采样当步外力。上述方向本轮未执行。

## 计账、资源与交付

实际16条新完整生成、1000/1000次模型前向返回；0优化器更新、0模型反向、0OT、0几何Guidance、0额外完整重放。另有CPU解析mock测试752次调用尝试/751次返回（含一次预设失败），与真实模型预算分列。16份NPZ保存接受状态、全部stage状态/速度/时间，便于追溯。

每个case在CUDA初始化前获得同一共享锁 `/tmp/wrp_meshflow_jedge5_diag_gpu0.lock`，每次独占1个GPU worker；使用不提前关闭的OS文件描述符，进程退出才释放。没有因显存空余绕过其他窗口。每次获取锁前后记录显存、GPU利用率、进程、CPU/RAM；保留2GiB余量，CPU单线程评分与GPU工作重叠，未停止其他窗口。

四worker锁等待共{timing['shared_lock_wait_seconds']:.3f}s；扣除等锁后的worker墙钟合计{timing['GPU_worker_wall_excluding_lock_seconds']:.3f}s，其中16条采样墙钟合计{timing['saved_rollout_wall_seconds']:.3f}s；离线正式几何评分{timing['CPU_scoring_seconds']:.3f}s。绘图距离复算与正式评分单独登记，重叠时间不能相加当端到端耗时；采样事件计时不是独占硬件benchmark。

交付：[结果JSON](results.json)、[24行逐输出表](evaluation/per_output.csv)、[相同NFE配对差](evaluation/paired_equal_NFE.csv)、[身份与hash](case_identity_manifest.json)、[代码diff](code.diff)、[最终审计](final_audit.json)。官方权重、维护模型、历史输出及公共实现均未修改；没有新增或复制checkpoint。到此停止。
'''
    (OUT/'SOLVER_PILOT_REPORT.md').write_text(report,encoding='utf-8');print('REPORT READY')
if __name__=='__main__':main()
