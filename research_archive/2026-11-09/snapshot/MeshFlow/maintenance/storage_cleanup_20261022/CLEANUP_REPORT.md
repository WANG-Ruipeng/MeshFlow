# 2026-10-22 本地冗余恢复点与缓存清理

已按用户要求实际删除159个文件，逻辑大小10,977,147,265字节（10.2233 GiB）。E盘两次可用空间读数增加10,977,435,648字节；期间其他进程活动可能影响可用空间差值，逻辑字节以逐文件账本为准。

删除范围：

- 已完成的Bench STATE两臂、COVERAGE、USER v2、EARLY_FM共5份local1250的`rolling.pt`。这是过期中间恢复点，不是终点相同副本。
- COVERAGE和USER v2两份`checkpoint_step500.pt`，仅作旧安全恢复点，无对应正式阶段评价或当前下游恢复依赖。
- 80文件的旧ZIP验包临时展开树；每个文件均与保留的主交付原件SHA256完全相同。
- 72文件的`r2/build`可重建构建树。

保留五分支全部step0、完整step1500、generator_step1500及日志/JSON。COMMON12000/A初始参考、官方模型、Chair正结果、旧RAW/POST、原数据、评测噪声/查询、原评分、图片和报告均未列入删除白名单。新26父/91条件合并提案及全部构造证据未动，删除后原封存恢复入口已核验PASS。

95个保护文件SHA256前后相同；五个受影响实验的28,381个非权重资产大小和修改时间核对相同。各删除路径均解析为E:\MeshFlow内的精确文件，拒绝reparse路径；使用PowerShell LiteralPath逐文件删除，没有递归跨目录删除。

这些中间点的直接权重重放正式退役；旧报告中的路径及原历史库存仍是历史记录，不能用旧全库存校验器要求自动补回。最终恢复和推理仍使用保留step1500。原科学结果未重写。

证据：`prepared_manifest.json`（路径/大小/hash与保留清单）、`deletion_ledger.jsonl`（159条实际删除）、`cleanup_receipt.json`、`cache_review.json`、`nonweight_assets_before.csv`、`disk_before.txt`与`disk_after.txt`。两次维护脚本准备阶段错误在`engineering_note.txt`记录，均发生在删除前；执行阶段完成且无失败。没有模型调用或训练。

用户随后授权复核数据并训练新增数据臂；该训练使用独立实验合同，清理不改变其模型/数据/预算。
