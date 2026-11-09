# PATCH 与普通对照：实际图审记录约定

本分工由 `/root/official_augmentation_audit` 评阅 `PLAIN200`、`PLAIN267`、`PATCH200`。固定 12 个请求，逐一评阅 `RAW`、`BASE_POST`、`CLEAN_FINAL_POST`，共 **36 张七行对照页、108 个模型单元**。目前只准备模板，尚无本轮图审结果。

沿用 `BENCH_VISUAL_CRITERIA_V2`，来源为 `experiments/r2_bench_four_routes_v1/exposure/visual_review/v2_definition.json`，五轴字段及枚举与上一轮 `bench_multilod_training_v1/visual_review/protocol.json` 一致。新实验不继承任何历史逐例标签。

## 五轴独立判读

| 字段 | 可选标签 | 判读规则 |
| --- | --- | --- |
| `overall_shape` | `REASONABLE / INCOMPLETE / UNCERTAIN` | 是否有可辨认、组织合理的完整长椅主体与主要支撑。允许不同款式、装配分离和非水密薄板；主体或主要支撑明显缺失、破裂或彼此不相容才判不完整。 |
| `C_use` | `PLAUSIBLE / REPLACED_OR_IGNORED / UNCERTAIN` | 蓝色 C 是否在灰色主体中承担合理部位关系。蓝色可见或存在精确共边均不足以证明采用；另画灰色替代座面、把 C 留成无关插片等计为替代或忽略。 |
| `support_retention` | `PRESENT / MISSING_OR_DAMAGED / UNCERTAIN` | 保留有面积、结构连贯的支撑表面，或合理的细杆/框架。不能仅因仍有少数腿端残丝就判保留；也不能仅因局部小缝就判主要支撑消失。 |
| `visible_fragment_or_gap` | `YES / NO_OBVIOUS / UNCERTAIN` | 可见悬浮碎片、异常撕裂或结构间断。真实板条间隙、合理独立零件不自动判坏；截面端点不自动等于裂缝，必须结合两张整体视图。 |
| `layering_or_fold` | `YES / NO_OBVIOUS / UNCERTAIN` | 可见多余重层、相互穿过或折叠的面皮、竞争座面。不能由 FF/I 数值反推标签。遮挡和薄层歧义需保留不确定。 |

三个正向轴的交集仍是 `REASONABLE ∩ PLAUSIBLE ∩ PRESENT`。它不认证无自交、无裂缝、拓扑接合或水密。两个缺陷轴单独保留，不能被三轴交集遮盖。

## 实际查看与证据

每张 `gallery/<stage>__<request_id>/comparison.png` 从上到下的模型行是：EXPOSURE、PLAIN200、PLAIN267、PATCH200、LOOP200、RL_ZERO、RL16。本分工只填写第 2、3、4 行；每行必须同时检查两张斜视图和原 front/side 截面。

评阅是 **AI_PROXY、非盲**，不能表述为独立人类认证。图中可能看到其他模型行，但不代替其他分工填写结论。原源码保证的 PATCH 共边也不能作为视觉成功证据。

真实打开图像后记录 `view_image` 事件、UTC 时间、实际 PNG SHA256、同目录 `comparison.json` SHA256；所有标签必须附具体可见位置说明。存在裁切或遮挡时注明，必要时使用 `UNCERTAIN`。未查看保持 `null`，不能用 `UNCERTAIN` 冒充完成。

不读取本轮数值分数、PATCH 结构诊断或旧逐例标签来生成判断。不按唯一来源款式评分，不因看到预期机制而给正标签。RAW、BASE、FINAL 各自判读，不能由前后阶段自动复制。

## 保存格式

待填写模板是 `templates/official_augmentation_audit.json`，保留原件。完成记录写到 `completed/official_augmentation_audit.json`。每查看 3—6 张图后保存独立工作片段，避免丢失真实观察。尚未完整的记录保持 `IN_PROGRESS`，108 个单元全部实际查看、身份及图像 hash 校验后才标 `COMPLETE`。

每行字段如下（这里不包含实际标签）：

```json
{
  "request_id": "原清单身份",
  "label": "PATCH200",
  "stage": "RAW",
  "page_id": "RAW__原清单身份",
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
  "note": null,
  "uncertainty_reason": null
}
```

这只是记录合同，不执行评分、渲染、模型调用或自动判断。
