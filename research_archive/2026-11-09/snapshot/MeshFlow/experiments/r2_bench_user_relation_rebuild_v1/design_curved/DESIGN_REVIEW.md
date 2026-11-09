# 两父曲面条件选择实录

本文件为条件构造子任务记录，最终训练资格由主任务生产打包与质量检查决定。没有修改完整目标 M，没有添加、移动、删掉或重新三角化任何目标面；只保存冻结 target_mesh 的面行号。两父 target_mesh.faces 与 target_case.source_faces、target_mesh.vertices 与 target_case.model_vertices 均逐数组完全相同，8 个源资产 hash 检查未变。

实际 AI_PROXY 查看了旧候选图、两张法向/顶点/面索引投影图及六张新 M/C/Y 双视角图。六条区域都为一个边连通分量，非流形边数为 0；这些基础统计不代替主任务完整 SourceAuditor 和 production.validate。

| 父 | 条件 | K | 实际观察 |
|---|---|---:|---|
| 9e429b | CURVED_THREE_BAY_SEAT_SURFACE | 26 | 三个座位湾及两条自然隆起分隔完整保留，确为上表面；两端在原网格进入端部竖向上升的折边处结束，不是高度截断。|
| 9e429b | COMPLETE_LEFT_END_OUTER_SURFACE | 20 | 从底脚沿外侧绕过扶手冠部，再沿内侧下降至座面连接的完整曲带；保留全深度。它是完整外侧曲面，不是封闭的整个端墙实体，明确不含前后端盖及内凹底侧。|
| 9e429b | CURVED_SEAT_PLUS_LEFT_END | 46 | 前两区域在原座面与端部转折边相接；曲面座壳与一端支撑共同输入。未把余下壳体删掉或隐藏成不存在。|
| 943dde | TRUE_CURVED_SEAT_UPPER_SURFACE | 24 | 中央跨全宽的略鼓起座面上表面，侧投影为弧形；区别于历史误选的平坦座底。|
| 943dde | COMPLETE_INNER_CURVED_BACK | 15 | 从座背连接边至背顶的完整内侧曲背，包含底部过渡；不含背后大平面与顶盖。|
| 943dde | CURVED_INNER_SEAT_BACK | 39 | 实际座面上表面与实际内曲背连续连接，侧投影清楚展示非平面座背关系。|

设计修正记录：943dde 初次构造法向容差过严，遗漏历史 QEM 后 x 法向约 0.000278 的原面 203，造成两连通片。检查面 ID 与侧轮廓后将中央挤出曲面容差改为 0.01，保留原面 203，内背及组合恢复为一个连通片。没有更改 M 或补造面。初版 M/C 灰蓝分组渲染有深度排序遮挡，改成同一 collection 的逐面色后重绘，并实际复看六张最终图。

`candidate_regions.json` 包含完整面 ID、源文件 hash、原几何资格收据、意义和限制。ID 为既有 QEM 后 target_case 的 source_faces 行号，不能误称原高面数网格的原面 ID；追溯更早来源需读取保留的 lineage.json。

恢复入口：

```text
python -B -m r2.data_tools.bench_user_relation_rebuild_v1.design_curved
python -B -m r2.data_tools.bench_user_relation_rebuild_v1.design_curved --make
```

该入口只重建本独立设计目录中的 CPU 派生图与选择清单，不运行模型，不改变生产评分或旧条件池。
