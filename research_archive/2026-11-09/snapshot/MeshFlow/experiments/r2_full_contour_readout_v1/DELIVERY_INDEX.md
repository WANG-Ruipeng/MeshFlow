# 完成交付：R2_FULL_CONTOUR_READOUT_V1

状态：COMPLETE。科学任务、实际图审、最终执行审计和小包校验均已完成。训练/生成进程已退出；不追加实验、不切默认、不commit/push。

结论：CONTOUR_S6000未优于同预算FULL_S6000。两臂均16/24明确采用、8/24有合法接口、6/24同时采用且有合法接口。原有六例全部保留，新增0；只在85的既有接口上增加两条边，DEV CF/I略升。完整结果见[REPORT.md](REPORT.md)。

- [9.60MB审阅ZIP](archive/snapshot_2335d68991bb0897/R2_FULL_CONTOUR_READOUT_V1_REVIEW.zip)：9,602,480 bytes，SHA256 `b0ad1d427e763a5ac960ee848f4f2e95e6bf23840bb7b5b0cfeb4f2c5b78f188`。
- [终点两模型图集](evaluation/gallery_FULL_S6000_CONTOUR_S6000/index.html)、[全部四时点图集](evaluation/gallery/index.html)、[完整剖面](evaluation/sections_gallery/index.html)。本地图保留原分辨率；ZIP的48张统一压缩预览只供导航，不足以替代原图审。
- [完整终点恢复权重](checkpoints/CONTOUR_step6000.pt)、[推理权重](inference/CONTOUR_S6000_generator.pt)。大权重与RAW/POST、OT均在本地，不放入小ZIP。
- [24请求联合成败](evaluation/comparison/snapshot_9f36c19ce0b272/final_joint_vs_FULL_S6000.csv)、[48个RAW/POST配对](evaluation/comparison/snapshot_9f36c19ce0b272/final_vs_FULL_S6000.csv)、[144条实际图审](evaluation/review/AI_PROXY_reviews.csv)。
- [四张学习曲线与148行底表](evaluation/result_plots/latest.json)、[训练完整日志汇总](training_analysis/latest.json)。
- [执行终态审计PASS](audits/execution_final_80a46e312e8c1c30.json)：1,545,992项检查，0错误/缺失，最终checkpoint一次CPU加载，旧AdamW时钟12000、新组6000。
- [ZIP校验PASS](archive/snapshot_2335d68991bb0897/verification.json)：470成员、469个成员hash加外部ZIP hash，包含实际评分源码、code.diff、源hash及完整CSV。

首次打包曾把 `00000.attempt.json` 当成完成回执并误查 `.attempt.npz`；6000个真实OT缓存全部存在。只修正归档文件名筛选，未新增模型、OT或评分。[保留的修正收据](audits/archive_ot_receipt_filter_fix/receipt.json)。原已通过审计的报告和科学源码未因此改变。压缩阶段统一缩小全部预览以达到10MB上限，没有删CSV、代码或挑图片。

只读入口：`r2/pilots/full_contour_readout_v1/resume.ps1 -Stage status`。控制器历史状态仍记录自动阶段终点 `COMPLETE_PENDING_ACTUAL_IMAGE_REVIEW_AND_REPORT`；其后的人工图审/报告/审计/归档由本目录[completion.json](completion.json)及上述实际收据确认完成，不覆盖控制器历史记录。