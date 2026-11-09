# 本轮实际图审格式

请实际查看本轮 `gallery/<STAGE>__<request_id>/comparison.png`，按固定原图记录。三位阅图者各负责四个完整请求，每请求的三个阶段均由同一位阅图者完成：4 请求 × 3 阶段 × 4 方法 = 48 条；合计 144 条。不要按数值指标或旧实验标签倒推图审。

方法固定为 `EXPOSURE`、`PLAIN1000`、`LAYOUT1000`、`LAYOUT_NULL`。阶段固定为 `RAW`、`BASE_POST`、`CLEAN_FINAL_POST`。`LAYOUT1000` 接受来源粗区域 P，是特权信息诊断，不能称为 C-only 部署方法；`LAYOUT_NULL` 与它是同一权重、显式遮蔽 P。

每页四行，分别是上述四方法；每行包含两幅斜视图及固定位置的 front/side 截面。需要一起看，截面线段端点不能单独证明破洞。允许使用不改变相机和尺度的原图裁剪辅助观看；仍需记录完整原图和 `comparison.json` 的 hash。如视角不能确定，真实查看后填写 `UNCERTAIN`；未查看项保持 null，不伪装为不确定。

文件命名为：

- `visual_review/completed/review_part_01.json`
- `visual_review/completed/review_part_02.json`
- `visual_review/completed/review_part_03.json`

阅图者身份应为真实代理名称，`reviewer_kind` 固定 `AI_PROXY`，`blind` 为 `false`。它不等于独立人类认证。

五轴及允许值：

| 字段 | 允许值 | 判断口径 |
| --- | --- | --- |
| `overall_shape` | `REASONABLE` / `INCOMPLETE` / `UNCERTAIN` | 是否构成可辨认、主体结构完整的长椅；不要求复制唯一来源款式或水密 |
| `C_use` | `PLAUSIBLE` / `REPLACED_OR_IGNORED` / `UNCERTAIN` | 蓝色 C 是否合理参与成品；灰色替代座面、漂浮插片不得仅因 C 保留判成功 |
| `support_retention` | `PRESENT` / `MISSING_OR_DAMAGED` / `UNCERTAIN` | 主要支撑是否保留连贯、有解释的表面或正常杆架；碎壳残丝不能自动当支撑 |
| `visible_fragment_or_gap` | `YES` / `NO_OBVIOUS` / `UNCERTAIN` | 是否有明确碎片、异常撕裂或分离；正常板条间隔与装配间隙不自动算缺陷 |
| `layering_or_fold` | `YES` / `NO_OBVIOUS` / `UNCERTAIN` | 是否有可见重复层、明显穿插、折片或竞争座面；不得用 CF/FF/I 代填 |

最小格式示例（以下是空模板，不是一次图审；占位符和 null 必须由实际查看替换）：

```json
{
  "schema": "bench_layout_oracle_actual_review_rows_v1",
  "status": "COMPLETE",
  "assignment_id": "review_part_01",
  "reviewer": "填写实际代理身份",
  "reviewer_kind": "AI_PROXY",
  "blind": false,
  "rows": [
    {
      "request_id": "填写冻结请求的完整 ID",
      "label": "LAYOUT1000",
      "stage": "RAW",
      "actually_viewed": null,
      "views_and_sections_checked": null,
      "image_sha256": "填写 comparison.png 的实际 SHA256",
      "image_receipt_sha256": "填写 comparison.json 的实际 SHA256",
      "tool": "view_image",
      "view_event_id": "填写真实 view_image 工具事件或明确可追溯的调用标识",
      "viewed_at_utc": "填写实际查看 UTC 时间",
      "overall_shape": null,
      "C_use": null,
      "support_retention": null,
      "visible_fragment_or_gap": null,
      "layering_or_fold": null,
      "note": "填写从两视角和截面实际观察到的证据，不按数值推测"
    }
  ]
}
```

如果使用裁剪，可附 `crop_paths`、`crop_sha256`、`crop_view_events`。不得把模板写成完成文件；只有 48 个实际查看的单元齐全后，才置 `status=COMPLETE`。

三轴联合只表示同一个输出同时为 `REASONABLE ∩ PLAUSIBLE ∩ PRESENT`，不代表没有碎片或冲突。完整五轴必须同时保留。每个阶段独立判断，不能沿用 RAW 标签冒充 POST 图审。

P 与 NULL 的 RAW 位移、轨迹变化只证明同权重对 P 有响应。位移更大、响应更稳定不自动意味着采用更好；采用和整体质量以本图审及完整原指标对照判断。

结束后主控执行（这些命令不会生成标签）：

```text
python -B experiments/r2_bench_layout_oracle_v1/control/summarize.py
python -B experiments/r2_bench_layout_oracle_v1/control/final_verify.py
```

`final_verify.py` 默认读取三个 `completed/*.json`，也支持重复指定 `--review PATH`。它发现缺失或未查看标签时不会补值，也不会启动任何训练、生成、评分或渲染。
