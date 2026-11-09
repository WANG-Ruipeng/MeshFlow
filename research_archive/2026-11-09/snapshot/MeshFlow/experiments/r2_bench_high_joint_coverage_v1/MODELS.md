# 模型与恢复身份

W为本轮唯一新增训练终点；B为上一轮已完成窄HIGH参照。两者从同一曝光分配B/global21000的live生成器、AdamW与RNG出发，各新增1500更新到global22500/native10500。本轮没有重训B。

|模型|文件用途|路径|SHA256|
|---|---|---|---|
|WIDE_HIGH|推理权重|/mnt/e/MeshFlow/experiments/r2_bench_high_joint_coverage_v1/training/WIDE_HIGH/generator_step1500.pt|`834d41c8996beb109e75322edf6280b1d4a35d01b18ba8bfb204bcb95778966f`|
|WIDE_HIGH|完整恢复状态|/mnt/e/MeshFlow/experiments/r2_bench_high_joint_coverage_v1/training/WIDE_HIGH/checkpoint_step1500.pt|`d93ccebac52901f6281a550232e3b411d403e85ee3828ab029bd7b48023ca957`|
|NATURAL_HIGH (read-only)|推理权重|/mnt/e/MeshFlow/experiments/r2_bench_natural_n_recipe_v1/training/NATURAL_HIGH/generator_step1500.pt|`b9a3979c84bb851489c49f8b1539618c6908d9f64b4b579cad4e51043264f2f8`|
|NATURAL_HIGH (read-only)|完整恢复状态|/mnt/e/MeshFlow/experiments/r2_bench_natural_n_recipe_v1/training/NATURAL_HIGH/checkpoint_step1500.pt|`111d2142780ddb21c9aa94cef9174571fbb22f0602d561c663c4af92b4a9e17d`|

W生成器状态SHA256：`24a14e8cca6817ddb724cf0dd199aa9a4951cbf5e3475f0cd97d0a6220e94da5`。
本轮固定W1500为结果，不挑中途点。rolling latest恢复点及各次保存hash有账本；最终完整状态含匹配AdamW和随机状态，旧源文件只读。
原Native/T1+role+Geo、OT后HYBRID0.25、batch8/micro1、原FM全batch自由坐标归一。无SEAM、占用头、额外loss或教师训练。
本表只声明资产身份，不声明整体质量晋级。现有Bench研究参考、Chair维护CLI默认和官方资产不变。没有cleanup、commit或push。
