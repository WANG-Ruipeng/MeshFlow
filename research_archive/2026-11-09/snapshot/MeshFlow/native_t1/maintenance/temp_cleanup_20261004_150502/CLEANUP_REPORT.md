# 临时缓存清理与数据盘点

已删除主工作区13个可重建Python字节码文件，共145,763 bytes（约146 KB）。逐文件先验证绝对路径位于E:\MeshFlow、普通文件类型、SHA256与盘点记录一致，再按LiteralPath删除；没有递归删除。空的__pycache__目录随后移除。

当前Chair源数据：3,135个UID/NPZ，网格文件47.19 MB，加train/test划分文件47.45 MB；原始官方划分为训练2,822、测试313。2026-10-02既有几何去重盘点为3,125组，此轮没有重算几何去重。

已准备实验父对象48个：训练32、验证8、测试8，是上述Chair源数据的子集。此前的B宽度面板仅用了其中8父，不等于项目只有8个对象。

其他历史测试数据包括ss_overfit的128个文件（同一个cow几何的复制），以及历史worktree中的closed16及其512面派生版本；不能把这些文件数相加当作新增独立对象。shapenet-cls.tar.gz原归档25.48 MB继续保留，不从归档大小推断全部网格已下载。

项目盘点合计94.41 GB（逻辑文件长度，含Git/worktree，链接不展开）。其中.pt权重/检查点约57.75 GB，Git元数据约24.26 GB。大体积源于这些保留资产。未把raw、coverage_vectors、pullback_cache、检查点或报告当临时文件删除。

保留文件逐项核对大小/修改时间：65,622项一致，意外改动或缺失0项。刚交付的CHAIR_CONTROL_ATTRIBUTION_REVIEW.zip SHA256仍为`f97ae6d8674fbb7468a9ebc70899a4da11869c39a44ffd209e74dfb3febf619b`。原始数据、权重、raw和报告删除数0。

清理清单见cache_deletion_candidates.json；实际删除回执见deletion_receipts.jsonl；统计见cleanup_summary.json与directory_sizes_depth*.csv。完整文件盘点CSV仅作为本轮核验临时输入，核验完成后移除；保留小型摘要与删除证据。没有训练、模型调用、Git reset/clean/commit/push。
