# 恢复与交付

本实验已经完成。优先读取 REPORT.md、decision_summary.csv 和 gallery/index.html；不需要重新运行模型。

科学入口位于 r2/tools/c_restore_coupling_v1/run.py，固定阶段为 prepare、geometry、denoise、score、gallery、all。WSL 环境：

```bash
cd /mnt/e/MeshFlow
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B r2/tools/c_restore_coupling_v1/run.py --stage prepare
```

prepare 校验固定注册/源码与已有资产，不产生模型调用。只对缺失的、尚未尝试的已授权阶段使用对应stage。若模型ledger已记START但缺合法receipt，禁止自动重放。已经完成的denoise阶段验证24个query并复用；不重新加载D0，不增加前向。score/gallery有逐输出完成receipt；本轮完成后不需要重复覆盖最终阶段耗时。

独立CPU核查入口：

```bash
/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B r2/tools/c_restore_coupling_v1/independent_audit.py --stage all
```

该核查只读取已保存模型输出/原面对分类，回算关系及hash，不重新运行D0、GEO或完整quality扫描。原旧数据与报告只读；新文档采用UTF-8 BOM/UTF-8明确编码。

真实预算：12同一RAW；12复用D0 proposal；24新增D0查询（2更新/请求）；24新GEO；24新POST完整quality，另24旧quality复用；48份PRE/POST匹配与拓扑；24两视角四组图；48逐输出AI_PROXY图审。CPU4几何正反控+3解析控不是实际D0合成能力测试。

主要证据：

- feedback_queries/：每次实际网络输入、proposal、误差、累计delta与receipt。
- pre_geo/：RBF和FEEDBACK各12份末端输出；RBF逐角点拟合细节。
- outputs/、scores/、diagnostics/：保持C的实际POST、原质量与连接证据。
- per_output.csv、quality_per_output.csv、topology_per_output.csv：逐输出完整主表。
- actual_edge_details.csv、per_boundary_vertex.csv、mechanism/：逐边、逐点与机制。
- gallery/visual_review.csv：最终48行图审。review_pending与原gallery_manifest的PENDING是渲染时历史；visual_review_completion.json是后续完成证明。
- scoring_code/：实际原评分器小文件；source_hashes.json、code.diff以打包阶段最终产物为准。
- independent_audit/：执行链和全评分独立核对。

审阅ZIP小于10MB，包含所有逐例CSV、24JPEG、实际源码、报告和账本。NPZ/PLY/完整PNG/权重保留本地，省略资产按路径/hash登记。ZIP不是可独立从零运行的大模型发布包；复核小表和图集不需要权重。

不得从结果挑参数、追加迭代、修改loss、自动切换模型或commit/push。当前固定任务已停止。