# Readout 中间权重清理（2026-10-08）

已按用户要求删除三份已完成实验的 500 步中间权重和五个 pytest 临时缓存文件，共 **8 个文件、4,751,866,505 逻辑字节（约 4.75 GB / 4.43 GiB）**。保留 A/B/C 三份 1000 步主要终点。维护清理不代表退役 Readout 源码，也没有改变默认模型。

| 已删除权重 | 字节 | 删除前 SHA256 |
| --- | ---: | --- |
| A_step0500.pt | 1565067275 | 41e977333efa59ed0e4fb226a4036c84bbd5e83ea428dfa9aa9f3c25cbd7e628 |
| B_step0500.pt | 1593398219 | 8d59d0d8e6f627a51c73d3594c81ae266a8bf1ea181c584f51c57cbe3045c7fa |
| C_step0500.pt | 1593398219 | 1900f0f2320dfeaf55feb18c6212a5e09d26d192e081fd2181a249074e7e25f5 |

三个权重合计 4,751,863,713 字节；Windows hardlink 清单和独立 WSL 检查均确认各只有一条硬链接。全卷可用空间前后读数另见 plan.json、execution.json 和 verification.json，可能受其他程序影响，不把它当作严格归因的释放量。

## 保留与核验

START 两条硬链接、官方和历史指定保留权重、A/B/C step1000 均保留。全部 RAW/PLY、轨迹、冻结 C/噪声/查询点、训练日志、逐例评分、图集、诊断数组、报告、代码及差异不变。独立虚拟环境保留。

完整保留 8,000 个 OT NPZ 及 16,000 个 attempt/return JSON：NPZ 仅 19,308,023 字节，完整配对目录约 29 MB，保留精确匹配记录的价值高于释放这点空间。没有重算 OT。

删除前后 **30974 个普通文件内容 SHA256 一致**；另 4 个 WSL pytest 符号链接的 reparse 元数据哈希一致。总计 30978 项，意外缺失/变动 0。符号链接比较的是 fsutil reparsepoint query 的 UTF-8 输出，不能称作目标内容哈希。

核验覆盖独立制品树（除虚拟环境）、独立源码树（除 .git）、原仓库 canonical 权重、START 历史硬链接、Geo4000、WORKSPACE.md 和 AGENTS.md。其他原仓库数据不在任何删除路径内；未宣称重算全盘哈希。两个仓库 Git status 前后相同，未 commit/push。

## 清理后的复现边界

500 步模型状态已主动删除，不能从该步精确续训或重新采样。已有 500 步网格、评分、轨迹、图集仍可离线复核。1000 步权重和精确训练配对仍在。

旧 tools/run_experiment.py --resume 在检查 500 输出缓存前读取权重哈希，旧 tools/verify_delivery.py 固定核验六份权重，所以这两个旧入口将因缺少 500 权重而失败。保留原代码与历史 PASS，不改写执行当时的证据。

现有 outputs/all_completed.json 的 CPU 评分、渲染、汇总不需要 500 权重。最近精度路径研究用到的 B/C1000、全部后续诊断和预算账本保留。旧报告中的文件存在状态是历史快照，应结合本删除清单阅读。不得为了补回缺失路径自动重训或生成。

## 执行记录

使用 8 条绝对路径白名单，核对规范路径和逐级 reparse point，逐文件删除前再次核对大小与 SHA256。只用 PowerShell Remove-Item -LiteralPath 删除文件，无递归目录删除。

删除前检查 Windows 和 WSL 进程，没有活动实验模型进程；未终止桌面、Unreal Editor 等用户程序。初次准备遇到 4 个 WSL 测试符号链接的 Windows 内容读取错误，当时删除数为 0；保留原脚本后改为比较链接元数据，再完成准备。见 engineering/。

本轮模型加载、前向、反向、训练更新、新生成、重抽噪声均为 0，没有重算评分。仅新增维护账本和制品根目录清理说明。

账本：delete_manifest.csv、deletion_ledger.jsonl、preserved_before.csv、preserved_after.csv、plan.json、execution.json、verification.json、resource_snapshot.json、cleanup.ps1。维护入口按 Prepare / Execute / Verify 分阶段；只有账本确认删除的项目才会跳过，若删除与完成落账之间中断，需要人工依据记录核对，不会静默当作成功。