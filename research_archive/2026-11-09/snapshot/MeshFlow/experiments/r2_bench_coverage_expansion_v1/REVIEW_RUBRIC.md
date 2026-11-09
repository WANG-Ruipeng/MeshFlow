# 配对图审约定

在终点评测生成之前写定。评阅者均标识为 `AI_PROXY`，非盲；不是人类用户评分，也不认证无交或水密。保留完整144页，不能只选改善图片。每页同一请求、同一阶段，A与COVERAGE各一行记录，共288行。

每行记录 `page_id, request_id, condition_id, parent_id, split, part_type, stage, label, reviewer, image_path, image_sha256`，以及以下字段：

- `C_use`: YES / NO / UNCERTAIN。C在成品中有合理部件用途，而非被一套灰色结构替代、夹在内部或明显脱离；无需与来源椅子的款式相同。看不到、用途多解或遮挡时选UNCERTAIN。
- `whole_shape`: COMPLETE / PARTIAL / UNCERTAIN。是否能辨认完整合理的长椅主体和支承；局部缺面/接缝不自动等于主体缺失。
- `layered_or_conflicting`: YES / NO / UNCERTAIN。图中是否有明显替代层、穿插层或折叠；定量冲突由原评分器另报。
- `visible_fragment`: YES / NO / UNCERTAIN。三维图与截面联合可见孤立小片、悬片或明显破口。只有一条截面断线时不能据此确认拓扑碎片。
- `note`: 简短具体的图像观察，注明对图和剖面是否一致。不得抄指标代替实际看图。

同页再记录 `pair_preference`: COVERAGE / A / TIE / UNCERTAIN，针对整体可用性而非单一接缝数量；说明哪项改善、哪项代价。相近时记TIE，不强分胜负。RAW、BASE_POST、CLEAN_FINAL_POST分别评阅；不把后处理变化算成生成器RAW收益。

图审记录写入 `visual_review/review_<reviewer>.json`，格式为 `{reviewer, rubric_sha256, rows:[...]}`。以实际打开的图像为凭证，每个模型的行都保存该页hash。最终先检查覆盖与重复，再汇总逐请求和逐父变化。闭合C的接口NA保留，不能因接口为0覆盖合理部件用途判断。
