# 本轮实际图审记录格式

固定请求、分工和五轴定义分别见 `assignments.json`、`protocol.json`。三份完成记录分别保存为 `completed/review_part_01.json`、`review_part_02.json`、`review_part_03.json`，各有 **144、135、135** 个 model × request × stage 记录。不要改空模板；不要把旧 anchor 的已查看状态复制为本轮状态。

推荐使用下列扁平 `rows` 格式。以下只是待填写示例，不是实际观察。只有真实打开图像、查看三行模型各自两个视角和两个固定截面以后，才能填写标签、时间和事件，并将顶层 `status` 改为 `COMPLETE`。

```json
{
  "schema": "bench_multilod_actual_review_rows_v1",
  "status": "PENDING_ACTUAL_VIEW",
  "assignment_id": "review_part_01",
  "protocol_sha256": "复制 protocol.json 的实际 SHA256",
  "reviewer": "实际评阅代理名称",
  "reviewer_kind": "AI_PROXY",
  "blind": false,
  "rows": [
    {
      "request_id": "从分工复制原 request_id",
      "label": "EXPOSURE",
      "stage": "RAW",
      "page_id": "RAW__原request_id",
      "tool": "view_image",
      "view_event_id": null,
      "viewed_at_utc": null,
      "actually_viewed": false,
      "views_and_sections_checked": false,
      "image_sha256": null,
      "image_receipt_sha256": null,
      "overall_shape": null,
      "C_use": null,
      "support_retention": null,
      "visible_fragment_or_gap": null,
      "layering_or_fold": null,
      "note": null
    }
  ]
}
```

- 一个 comparison.png 包含三个模型，所以实际打开同一张图后，三个 model 行可以引用同一实际 view 事件和时间，但每行五轴和可见证据说明必须独立填写。
- `image_sha256` 是实际打开的 `gallery/<page_id>/comparison.png`；`image_receipt_sha256` 是同目录 `comparison.json`。汇总器同时验证图片、图集回执与模型/请求/阶段身份。
- UTC 时间使用带 `+00:00` 或 `Z` 的 ISO 字符串。所有本轮查看时间必须晚于图审协议冻结时刻。
- 可选的 `parent_id`、`condition_id`、`noise_index`、`report_group` 若填写，必须与冻结 manifest 一致；不填时汇总器由注册清单补齐。
- 同一请求全部阶段、全部模型必须由同一实际 reviewer 审。未查看记录保持 `null`，不能用 `UNCERTAIN` 代替未查看；`UNCERTAIN` 只用于实际图像证据不足。
- 也兼容 `templates/review_part_*.json` 的 `pages` 格式：填写页级实际查看身份与 hash、模型级五轴与 note，顶层补 `status: COMPLETE`、实际 reviewer。不要同时提供 `pages` 和 `rows`。

完整后三份文件一起运行：

```text
python -B experiments/r2_bench_multilod_training_v1/control/summarize_visual.py
```

未齐只返回 `PENDING`，不生成完成结果。完整且校验通过才写 `visual_review/summary_v1/`：五轴计数、三轴交集、逐请求/父/组的三组配对转移，以及独立的 `baseline_review_differences.csv`。原 EXPOSURE 标签仅在完整图审校验通过后读取作历史比较，不用于推导或纠正本轮标签。

三轴交集严格表示 `REASONABLE ∩ PLAUSIBLE ∩ PRESENT`。这不等于无裂缝、无相交或水密。所有 UNCERTAIN 保留在完整分类计数和转移矩阵中；任一三轴明确失败则交集为 NO，否则有 UNCERTAIN 则交集为 UNCERTAIN。严格正例只计 YES，分母包含全部已评阅请求。
