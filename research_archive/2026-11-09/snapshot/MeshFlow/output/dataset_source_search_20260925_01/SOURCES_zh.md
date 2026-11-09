# 官方数据来源检索（2026-09-25）

本次查阅 MeshFlow 官方 README、Hugging Face 文件目录、论文数据说明，并复核本地归档类别元数据。只做来源调查，没有运行新对象网络、优化或几何资格筛选。

## 已确认的候选

| 来源 | 下载规模/本地情况 | 与下一轮实验的关系 |
|---|---|---|
| MeshFlow ShapeNet bottle (02876657) | 本地 shapenet-rebuttal2.tar.gz 已有225 train /249总对象 | 最省事的跨类别候选；尚未测合格数量 |
| MeshFlow ShapeNet display (03211117) | 同包528 train /586总对象 | 跨类别备选；尚未测合格数量 |
| MeshFlow ShapeNet cabinet (02933112) | 本地 shapenet-rebuttal.tar.gz 已有601 train /667总对象 | 跨类别备选；尚未测合格数量 |
| 官方上游 ShapeNetCore loudspeaker (03691459.zip) | HF目录显示529 MB，需要提供方批准访问 | 保持音箱类别的新来源候选；尚未下载/逐UID排重/几何筛选 |
| MeshFlow处理版 Objaverse | objaverse_occ_v5_ids.tar.gz 170 MB + split.tar.gz 2.19 MB | 多类别备用池；音箱数量及径向合格数量未知 |
| MeshFlow处理版 Sketchfab | sketchfab.tar.gz 669 MB | 更大的备用池；类别和合格数量未知 |

本地官方处理版音箱417父UID（376train/41test）已在主包和rebuttal2重合；再次下载同包不产生新父对象。上轮缺16的结论限定于原来源/原split/原筛选规则。41个官方test未做此次资格筛选，不能列为已确认合格的新16。

MeshFlow论文明确沿用MeshXL的数据划分，并使用边折叠简化与Hausdorff距离筛选进行数据增强。因此不同面数或不同网格版本不等于新父对象。MeshXL官方公开音箱train小包已下载（2,857,529字节）并用静态pickletools核对：376个唯一object_id与本地train376集合完全相同，新增0；未执行pickle、未解析网格张量、未下载val/test。因此它没有解决新增父对象的问题。

## 建议

若优先快速判断冻结3k是否有迁移信号，先盘点已在本地的225个bottle训练对象，保持原几何规则且不读取模型效果；只有通过几何资格并完成身份排重后，才冻结16个与错配关系。此结果必须标为跨类别验证，不替代新音箱同类验证。

若保持原来的新音箱验证范围，优先获取ShapeNet官方03691459.zip，核对与本地417的父UID和来源重合，再做几何资格。原始高面数网格若无法直接用于现有512细分流程，需要先明确预处理协议，不能默认为与当前处理版同分布。是否能得到16个合格对象仍未测。

优先bottle是根据单体形状可能更适配当前单中心径向表示的推断，不是已测筛选通过率。所有来源都仍需通过相同封闭性、定向、径向单交和精确几何检查。

## 直接来源

- [MeshFlow官方README](https://github.com/qiisun/MeshFlow#data)
- [MeshFlow官方HF数据目录](https://huggingface.co/datasets/qsun2001/meshflow/tree/main/obj_data)
- [MeshFlow论文数据说明](https://arxiv.org/html/2606.23489v2#S5)
- [ShapeNet官方数据页及访问要求](https://huggingface.co/datasets/ShapeNet/ShapeNetCore)
- [ShapeNet分类文件目录](https://huggingface.co/datasets/ShapeNet/ShapeNetCore/tree/main)
- [MeshXL官方代码与数据链接](https://github.com/OpenMeshLab/MeshXL)
- [MeshXL处理版数据目录](https://huggingface.co/datasets/CH3COOK/MeshXL-shapenet-data/tree/main)

目录显示体积为网站十进制近似值；225/528/601为本地只读split审计结果，不是网站承诺的几何合格计数。当前没有改变实验范围或修改已冻结协议。