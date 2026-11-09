# 本轮论文图清理

按用户“先把这次生成的清理一下”的要求，删除整个 `E:\MeshFlow\deliverables\chair_paper_figures_v1`，包含四张主图的 PNG/PDF/SVG、预览、面板缓存、候选图板、ZIP、脚本和本轮来源副本。

实际删除 394 文件、32,675,398 逻辑字节（约 32.68 MB）。删除前核验绝对路径位于指定工作区且不存在重解析点；逐文件大小与 SHA256 记录在 `deleted_files.json`。旧论文图链接已失效。

为随后获准的“两三角形面片 control”小试复用一份通用只读网格渲染器，保存至 `native_t1/runs/chair_two_triangle_control_v1/scripts/raw_renderer.py`。没有保留旧论文图或 ZIP 副本。

原始实验 RAW、数据、C、评分、报告、权重均不在删除范围。509 个旧来源依赖文件在清理后重新计算 SHA256，全部与清理前记录一致；见 `protected_sources_before.json`、`protected_sources_after.json`。没有 Git commit/push。
