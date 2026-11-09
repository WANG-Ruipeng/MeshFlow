$ErrorActionPreference='Stop'
$taskRoot='E:\MeshFlow\native_t1\maintenance\capped_retirement_20261007'
$workspaceRoot='E:\MeshFlow'
$utf8=[Text.UTF8Encoding]::new($false)
$verify=Get-Content -LiteralPath (Join-Path $taskRoot 'verification.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if($verify.status -ne 'PASS' -or $verify.deleted_files -ne 16022 -or $verify.protected_files -ne 43405){throw 'Final verification required'}
$notice=@'
# 2026-10-07 权重与派生缓存退役

用户已要求结束本轮反馈/CAPPED探索，并统一回到原始 Chair HYBRID START baseline。

本目录全部实验 PT 已清理，JSON 身份记录保留原样。原始 RAW、条件/Gaussian/查询、反馈池、报告、逐例 CSV、评分及源代码、图集和最终审阅 ZIP 均保留。

反馈探索与复现实验中 streams 下的正式派生 OT NPZ 已清理；*.attempt.json 和训练计划保留，避免旧入口静默重跑。小型首批门禁测试缓存保留。HALF 本身无 OT 缓存。

历史报告中的“权重保留”及全文件审计 PASS 是执行当时的状态；当前删除清单以 [清理报告](../../../maintenance/capped_retirement_20261007/CLEANUP_REPORT.md) 为准。旧恢复/训练/半幅构建命令不再是可直接恢复的承诺。不得因缺文件而自动重训、重生成或恢复该路线。

正式 baseline 为 native_t1/checkpoints/chair_hybrid_start_total5000.pt，SHA256 d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80；正式 CLI 默认 chair-hybrid 已验证。旧原件、官方权重及其它既定保留模型不在本轮删除范围。
'@
$markerPaths=@()
foreach($name in @('chair_control_feedback_auto_v1','chair_capped500_gradbudget_v1','chair_capped500_half_v1')){
    $markerPath=Join-Path $workspaceRoot ('native_t1\runs\'+$name+'\checkpoints\RETIREMENT.md')
    if(Test-Path -LiteralPath $markerPath){if([IO.File]::ReadAllText($markerPath) -ne $notice){throw 'Retirement marker exists with different content'}}
    else{[IO.File]::WriteAllText($markerPath,$notice,$utf8)}
    $markerPaths+=$markerPath
}
$workspaceNote=@'

## 2026-10-07 当前决策：回到原始 START，结束 CAPPED 探索

按用户明确要求，以原始 Chair HYBRID START / total5000 继续作为 baseline。正式默认 profile 一直为 chair-hybrid，已实际核验解析及一次 CPU strict load；零模型前向、零训练、零生成。START 两条硬链接路径及完整 AdamW/RNG/输入流状态均保留。

已删除 chair_control_feedback_auto_v1、chair_capped500_gradbudget_v1 和 chair_capped500_half_v1 的22份实验权重，以及两处正式 OT 派生缓存的16,000个NPZ，共33,426,755,776逻辑字节（约33.43GB）。OT尝试日志、训练计划、小型测试缓存、原始RAW、冻结条件/噪声/查询、反馈池、全部评分/报告/CSV/图集/源码和审阅ZIP保留，43,405个保留文件前后hash一致。

相关历史代码留作评分依赖和证据，不自动运行旧恢复入口，也不因权重/缓存缺失而重训或重生成。原报告中的旧文件存在状态以本轮删除清单为准。官方权重、Geo4000、START、旧显式FM/JEdge5及其它既定保留checkpoint不在本轮删除范围。没有修改正式代码或commit/push。

详见 [清理报告与完整账本](native_t1/maintenance/capped_retirement_20261007/CLEANUP_REPORT.md)。

'@
$workspacePath=Join-Path $workspaceRoot 'WORKSPACE.md'
$oldText=[IO.File]::ReadAllText($workspacePath)
if(-not $oldText.Contains('## 2026-10-07 当前决策：回到原始 START，结束 CAPPED 探索')){
    $cut=$oldText.IndexOf("`n")+1
    if($cut -le 0){throw 'Workspace document has no header line'}
    [IO.File]::WriteAllText($workspacePath,($oldText.Substring(0,$cut)+$workspaceNote+$oldText.Substring($cut)),$utf8)
}
[IO.File]::WriteAllText((Join-Path $taskRoot 'workspace_note.md'),$workspaceNote,$utf8)
$result=[ordered]@{status='COMPLETE';baseline='Chair HYBRID START total5000';profile='chair-hybrid';deleted_weight_files=22;deleted_cache_files=16000;deleted_logical_bytes=$verify.logical_bytes;retained_files_unchanged=$verify.protected_files;retirement_markers=$markerPaths;workspace_sha256=(Get-FileHash -LiteralPath $workspacePath -Algorithm SHA256).Hash.ToLowerInvariant();real_model_forward_calls=0;training_updates=0;new_generation=0;automatic_git_commit=$false;stopped=$true}
[IO.File]::WriteAllText((Join-Path $taskRoot 'completion.json'),($result|ConvertTo-Json -Depth 5),$utf8)
$result|ConvertTo-Json -Depth 5
