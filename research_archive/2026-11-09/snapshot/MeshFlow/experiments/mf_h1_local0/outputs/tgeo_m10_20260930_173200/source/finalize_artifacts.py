"""Final read-only integrity audit and human-reviewed reporting; no model calls."""
from pathlib import Path
import json, subprocess, time
from experiments.mf_h1_local0 import tgeo_m10_run as run
out=Path('experiments/mf_h1_local0/outputs/tgeo_m10_20260930_173200').resolve()
r=json.loads((out/'results.json').read_text())
assert r['status']=='PASS_EXECUTION'
expected=dict(model_forward_attempts=11652,model_forward_returns=11652,backward_attempts=8002,
    backward_returns=8002,optimizer_step_attempts=1000,optimizer_updates=1000,
    training_OT_calls=8000,training_OT_returns=8000,gate_OT_calls=8,gate_OT_returns=8,
    sampling_forward_attempts=3650,sampling_forward_returns=3650,
    formal_rollouts=72,replay_rollouts=1,complete_rollouts=73,rollout_attempts=74,euler_updates=3650)
assert all(r['counts'][key]==value for key,value in expected.items())
sources=run.verify_sources(r);assets=run.verify_assets(r)
assert r['arms']['Geo_continue']['stream_audits']==r['arms']['Geo_M10']['stream_audits']
assert len(r['arms']['Geo_continue']['stream_audits'])==500
assert all(row['C_all51_bitwise'] for row in r['generated']+r['replays'])
assert r['replays'][0]['all51_bitwise_equal']
checked=[]
for arm in run.ARMS:
    assert r['arms'][arm]['completed_updates']==500 and r['arms'][arm]['status']=='PASS'
    for step,record in r['arms'][arm]['checkpoints'].items():
        actual=run.record(record['path'])
        assert actual['sha256']==record['sha256'] and actual['bytes']==record['bytes']
        checked.append(dict(arm=arm,additional_step=int(step),**actual))
for row in r['generated']:
    assert run.file_sha256(row['raw_path'])==row['raw_sha256']
for row in r['scores']:
    assert run.file_sha256(row['metrics_path'])==row['metrics_sha256']
git=subprocess.check_output(['git','-c','core.autocrlf=true','status','--short'],text=True).strip()
assert not git
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert head==r['protocol']['published_head']
assert all((out/'source'/p.name).read_bytes()==p.read_bytes() for p in run.PROJECT.glob('tgeo_m10_*.py'))
r['final_integrity']=dict(status='PASS',execution_sources=sources,protected_assets=assets,
    checkpoints_verified=checked,raw_hashes_verified=72,score_hashes_verified=72,
    source_snapshot_files=5,all500_input_batches_equal=True,all73_trajectories_C_preserved=True,
    repetition_all51_bitwise=True,counts_verified=expected,public_git_status=git,published_head=head,
    no_new_commit_or_push=True,optimizer_updates_after_training=0,source=run.record(__file__),
    completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
r['io_recovery']['cause_evidence']='Atomic rename PermissionError coincided with PowerShell reading the same file; Windows handle-lock explanation is inferred, not verified by OS handle tracing.'
r['interpretation']=dict(status='NO_CONSISTENT_ADDED_BENEFIT_FROM_M10',
    scope='Two trained parents, one restored training stream; original K12 patch eight new noises and each extra patch two noises.',
    working_model_recommendation='Keep the existing T1+T_geo default; do not promote M10 as a general improvement.',
    CF='Lower mean CF proper-affected fraction on original P0/P1; this is a real full-generation tradeoff.',
    FF='Original patch mean M10-minus-continue: P0 0.0 pairs, P1 -0.375; no reproduced large FF reduction.',
    geometry='Original P0 free RMS worse6/8, P1 better5/8 but target coverage and Gamma tails worsen in mean.',
    tails='P0 mean free-error cost is strongly driven by seed9802; Gamma p95 tails P0 seeds9802/9805 and P1 seeds9802/9807. No samples removed.',
    extra_patches='Patch1 P0 degrades while P1 partly improves; patch2 errors increase in mean for both. Each parent/patch has only2 noises.',
    causal_limit='Training recipe was isolated, but these observations do not establish a unique mechanism or unseen-object generalization.')
r['report']['visual_review']='PASS_ACTUALLY_VIEWED_BOTH_PNGS'
r['report']['visual_review_method']='Direct PNG rendering via exec image output after view_image sandbox helper failed (Windows1385). No figures edited.'
for fig in r['report']['figures']:
    assert run.file_sha256(fig['path'])==fig['sha256']
    fig['visual_review']='PASS: full/free labels visible; no panel clipping; same registered view and parent scale.'
report=out/'T_GEO_M10_REPORT.md'
text=report.read_text()
text=text.replace('图像视觉验收由调用方完成；report记录PENDING_CALLER_REVIEW直到实际查看。',
    '两张PNG已由主代理实际查看：full/free标签、已知C着色及统一视角/范围可读，无面板裁切；固定图不能替代全部样本统计。')
lead="""## 本轮判决

实现与执行通过，但没有建立M10在既有T_geo上的稳定增量收益。建议保留现有T1+T_geo工作基线，不将M10改为默认；M10终点仅保留作CF穿刺与覆盖/接口精度取舍的研究分支。

原patch、每对象8个新噪声的均值如下。误差单位为百分比bbox对角线；CF为受proper穿刺影响的free面百分比，FF为非法面片对数。

| 指标 | P0 Geo_continue | P0 Geo_M10 | P1 Geo_continue | P1 Geo_M10 |
| --- | --- | --- | --- | --- |
| 排除C的对称RMS (% L) | 0.7935 | 1.1371 | 0.5566 | 0.5255 |
| 目标→free RMS (% L) | 0.8396 | 1.3338 | 0.4730 | 0.6048 |
| Gamma p95 (% L) | 2.7481 | 4.2468 | 2.6870 | 4.0638 |
| CF受影响free面 (%) | 12.375 | 7.125 | 6.375 | 3.625 |
| FF非法对均数 | 74.000 | 74.000 | 66.250 | 65.875 |

P0自由面RMS配对差均值+0.3436个百分点，但配对中位仅+0.0045；9802是主要均值退化来源，仍有6/8变差。P1自由面RMS有5/8改善，Gamma p95有6/8变差。P0边界尾部主要受9802/9805影响，P1主要受9802/9807影响。保留全部异常样本，不将均值变化说成每个噪声均匀变化。

CF下降进入了完整生成，但FF大幅减少没有复现。原patch两对象的FF均差分别为0与−0.375对；部分额外patch存在局部FF收益，同时有对象/patch退化，不能合并为普遍收益。既有额外patch每对象每片仅2个噪声，分别报告。

纯FM继续训练相对Geo_initial也并非全面改善：原patch自由RMS均值P0 0.7014→0.7935、P1 0.4828→0.5566，而FF均数83.125→74、90.375→66.25。这里不能把“继续训练”或“叠加M10”任何一项自动当成更好的默认模型。

本轮仅两个已训练父对象与一个训练流；新噪声通过不等于新对象泛化。没有在看到结果后调整FC权重、门槛、seed、patch或训练长度。

"""
text=text.replace('## 对照与实际执行',lead+'## 对照与实际执行',1)
text+='\n\n最终完整性核对：1000新增更新；11652前向、8002反向；72正式完整生成+1逐位一致重放。74次调度包括1次未进入模型的I/O失败，原始失败快照与恢复源码均保留。文件锁原因是基于同时读取与原子替换失败的推断，未做OS句柄追踪。5份实验/恢复源码快照、4个本轮检查点、72原始数组、72评分文件及全部已登记原资产hash核对通过；公开Git工作树仍干净，HEAD未变，没有自动提交或推送。\n'
report.write_text(text)
r['report']['report']=run.record(report)
run.write(r,out)
print(json.dumps(dict(status=r['status'],integrity=r['final_integrity']['status'],counts=r['counts'],
    recommendation=r['interpretation']['working_model_recommendation'],report=r['report']['report']),indent=2))
