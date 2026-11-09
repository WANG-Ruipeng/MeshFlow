# 本轮实际图审

按 assignments.json 分三份，每份 4 请求 × 3 阶段 × 7 模型＝84 cells。
真实打开 gallery/<stage>__<request_id>/comparison.png（detail=original）；七行可能显示过小，
可以用本工具 crops 生成原图无缩放裁切，逐组查看。裁切仅帮助看清，不改变相机或截面。
每个 cell 独立填写五轴和中文可见证据，完整三阶段由同一 reviewer 评阅。
完成文件放 completed/review_part_01.json、02、03；不得修改空模板来冒充历史查看。
填写原图与 comparison.json SHA256、真实 view 事件、UTC、actually_viewed 和 views_and_sections_checked。
若使用裁切，额外记录 crop_paths/crop_sha256/crop_view_events；仍保留源原图哈希。
未查看保持 null；不确定标签仅表示真实观察仍无法判断。
不要读取数值评分或旧标签反推图审。AI_PROXY，非盲，非独立人类认证。
