# R2_SEAT_NOISE_TRANSFER_V1

固定三模型、两个分开汇总的面板。零训练/反向；48冻结请求x3模型=144 Euler50（7200生成器前向），每份RAW单次官方D0=144去噪前向。全部144 RAW保存校验后才开始D0。原权重/数据/评分实现/历史结果只读，不自动换默认、commit/push。

恢复总入口（现有完成单元hash复用，遇未完成模型预留拒绝隐式重放）：

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.seat_noise_transfer_v1.run --stage all
```

可分阶段 --stage generate / post / score / render。同一GPU全模型共享锁，不要并行启动两份总入口。约6GiB实验显存预算，PyTorch allocator cap5GiB预留约1GiB context估计，不等于WDDM无法测得的精确进程上限；单GPU worker，CPU评分最多2 worker。每次载入及调用边界检查全卡headroom，不关闭用户图形程序。工程恢复需要记录原因，禁止静默改变科学定义。

数据已冻结inputs/manifest.json，来源/种子/查询审计在inputs。匿名映射先于生成冻结于evaluation/blinding/model_key.json；实际图审提交前不要查看。图集generation并不代表已审阅。图审完成后analyse模块核验288行实际记录才解盲。旧噪声仅作为旧目录回归参考，不能并入新面板均值。

熟悉面板4父各8噪声；DEV面板4父各4噪声。DEV未进原67父条件训练池，但已有开发使用史、官方预训练曝光UNKNOWN；已知跨父近重复排除，不声称穷尽重复。新C的K/N与熟悉面板不同（含FULL K32与HALF K1），迁移结果不能单独归因记忆，见inputs/COVERAGE_LIMITS.md。

本文件是恢复合同，不是完成声明；完成状态以最终COMPLETE.json和REPORT.md为准。
后处理完毕后的离线入口（均不调用模型）：

```powershell
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.seat_noise_transfer_v1.analyse
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.seat_noise_transfer_v1.report
wsl -d Ubuntu --cd /mnt/e/MeshFlow -- /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.pilots.seat_noise_transfer_v1.package --stage check
```

分析按已保存的实际图审记录重聚合，不能以程序填表代替实际评阅。finalize是首次封存动作，COMPLETE已存在后不要重写完成时间。打包命令将check换成build；包已存在时仅校验source hash和zip hash，不重新生成模型或选择图片。小包是审阅/表格复算材料，恢复几何评分仍依赖本地NPZ及模型环境。