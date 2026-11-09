# 端点图审汇总：已有记录的只读核账

完整合并四位原评阅者对四条已见 TRAIN C 的记录：16 请求 × 4 模型 × RAW/POST = 128 行，32 页每页四列全部登记为 ACTUALLY_VIEWED。这里核验原图审声明、身份与图像 hash；不是聚合程序重新看图。没有新增视觉判断、几何评分或模型调用。

[逐行 CSV](visual_review.csv)；[完整 JSON、逐页账本与所有不确定记录](visual_review_summary.json)；[校验回执](visual_review_receipt.json)。

以下每格为 **YES / NO / UNCERTAIN** 原始计数，每组分母 16。未将 UNCERTAIN 当作 NO；缺陷字段的 YES 是发现可见问题，正向字段的 YES 是观察到对应性质，不能把各列直接相加排名。

| Stage | Model | C_adoption | whole_body_complete | support_missing |
| --- | --- | --- | --- | --- |
| RAW | SEAM200 | 16 / 0 / 0 | 16 / 0 / 0 | 0 / 16 / 0 |
| RAW | LONG | 16 / 0 / 0 | 16 / 0 / 0 | 0 / 16 / 0 |
| RAW | ANTI_COVER | 16 / 0 / 0 | 16 / 0 / 0 | 0 / 16 / 0 |
| RAW | LOCAL_NOISE | 16 / 0 / 0 | 16 / 0 / 0 | 0 / 16 / 0 |
| POST | SEAM200 | 16 / 0 / 0 | 16 / 0 / 0 | 0 / 16 / 0 |
| POST | LONG | 16 / 0 / 0 | 16 / 0 / 0 | 0 / 16 / 0 |
| POST | ANTI_COVER | 16 / 0 / 0 | 16 / 0 / 0 | 0 / 16 / 0 |
| POST | LOCAL_NOISE | 16 / 0 / 0 | 16 / 0 / 0 | 0 / 16 / 0 |

| Stage | Model | fragments_or_folded_layers | visible_cracks | visible_join |
| --- | --- | --- | --- | --- |
| RAW | SEAM200 | 2 / 8 / 6 | 1 / 4 / 11 | 10 / 0 / 6 |
| RAW | LONG | 1 / 11 / 4 | 1 / 3 / 12 | 9 / 0 / 7 |
| RAW | ANTI_COVER | 1 / 11 / 4 | 1 / 3 / 12 | 9 / 0 / 7 |
| RAW | LOCAL_NOISE | 1 / 12 / 3 | 1 / 4 / 11 | 9 / 0 / 7 |
| POST | SEAM200 | 1 / 11 / 4 | 1 / 4 / 11 | 10 / 0 / 6 |
| POST | LONG | 1 / 14 / 1 | 1 / 4 / 11 | 9 / 0 / 7 |
| POST | ANTI_COVER | 1 / 13 / 2 | 1 / 4 / 11 | 9 / 0 / 7 |
| POST | LOCAL_NOISE | 1 / 15 / 0 | 1 / 4 / 11 | 9 / 0 / 7 |

## 具体可见问题索引（原标签原文）

下面只索引原表中正向性质为 NO 或缺陷性质为 YES 的记录，不建立新的复合失败定义。全部 UNCERTAIN 原文单独保存在 JSON 的 uncertainty_records，原始逐行结果没有删减。

| 请求 | 阶段 / 模型 | 原不利标签 | 原评阅者具体观察 |
| --- | --- | --- | --- |
| 73b369b9_n2 | RAW / SEAM200 | visible_cracks=YES | 多边形上靠背出现明显三角形开缝，另一视角左侧也见亮缝；蓝座面、下部支撑板和底板齐全。 |
| 73b369b9_n2 | POST / SEAM200 | visible_cracks=YES | 上靠背三角形亮缝在 POST 仍明显，侧视角也有开口；座面和下部椅身存在。 |
| 3f741759_n1 | RAW / SEAM200 | fragments_or_folded_layers=YES | 蓝色C处于座面，圆弧靠背、竖向背条与四腿轮廓可辨；未见整块主体或支撑消失。正面剖面有从C左端向内斜伸的灰线，侧剖有独立内层短线；整体视图该层大部分被座面遮挡。 |
| 3f741759_n1 | POST / SEAM200 | fragments_or_folded_layers=YES | 蓝色C处于座面，圆弧靠背、竖向背条与四腿轮廓可辨；未见整块主体或支撑消失。正面剖面有从C左端向内斜伸的灰线，侧剖有独立内层短线；整体视图该层大部分被座面遮挡。 |
| 3f741759_n2 | RAW / SEAM200 | fragments_or_folded_layers=YES | 蓝色C处于座面，圆弧靠背、竖向背条与四腿轮廓可辨；未见整块主体或支撑消失。两剖面中C下方可见近邻灰色额外线段，整体外观将其遮住。 |
| 85b16941_n1 | RAW / LONG | fragments_or_folded_layers=YES | 蓝色座面和四条细腿保留；座后中央有一张宽薄长片向下垂到接近腿底高度。 |
| 85b16941_n1 | RAW / ANTI_COVER | fragments_or_folded_layers=YES | 蓝色座面、背板与四腿保留；座后中央出现宽薄下垂片，在两视角都可见。 |
| 85b16941_n1 | RAW / LOCAL_NOISE | fragments_or_folded_layers=YES | 座面、背板、四条细腿保留；后侧有从座后向下延伸的大薄片。 |
| 85b16941_n1 | POST / LONG | fragments_or_folded_layers=YES | 蓝色座面与四腿保留；向下垂到腿部的大薄片仍在，POST未将其消除。 |
| 85b16941_n1 | POST / ANTI_COVER | fragments_or_folded_layers=YES | 矩形背板、蓝色座面和四腿保留；后侧大薄片继续延伸到腿部。 |
| 85b16941_n1 | POST / LOCAL_NOISE | fragments_or_folded_layers=YES | 蓝色座面与四腿保留；座后下垂薄片保留，背板外轮廓近似其他续训臂。 |
| 85b16941_n2 | RAW / LONG | visible_cracks=YES | 主体与蓝色座面保留；后侧裙靠近座面处可见细小三角形白色开隙。 |
| 85b16941_n2 | RAW / ANTI_COVER | visible_cracks=YES | 蓝色座面、背板和四腿保留；后侧裙上缘可见与LONG相近的局部开隙。 |
| 85b16941_n2 | RAW / LOCAL_NOISE | visible_cracks=YES | 蓝色C仍作座面；正面斜视的左侧裙面有明显白色长缺口，背面侧裙也有小开口；四腿保留。 |
| 85b16941_n2 | POST / LONG | visible_cracks=YES | 主体与C用途保留；后侧裙上缘的小三角形白色开隙仍可见。 |
| 85b16941_n2 | POST / ANTI_COVER | visible_cracks=YES | 主体和四腿保留，后侧裙靠座面处的细小开口仍在。 |
| 85b16941_n2 | POST / LOCAL_NOISE | visible_cracks=YES | 蓝色座面与四腿保留；正面斜视左侧裙面的大白色开口仍在，背侧也见小开隙。 |

## 各父图审文字结论

以下概括沿用原评阅文字，不据评分改判。

- **3f**：主体与 C 座面用途保留；n1 旧的内向灰色剖面线在新臂中不明显，n2 RAW 的近座层线也更少见。但 n3 新侧剖存在含义不明的下表面断段；相似 POST 外观不证明冲突尾部已消失。见[原记录](reviews/3f.md)。
- **73**：n2 旧模型靠背的大三角开缝在新臂更连续；总体差别不大。座面下平行线可能是正常厚度或重层，保留不确定；没有 ANTI_COVER/LOCAL_NOISE 独有整体优势。见[原记录](reviews/73.md)。
- **85**：n1 三个续训臂均出现后侧长片，n2 LOCAL 的侧裙开口最明显；n0 LONG/ANTI 的背板开口保留款式歧义。n3 四臂接近。见[原记录](reviews/85.md)。
- **8c**：四种噪声均保留座面、梯形背条和支撑；新臂部分座箱剖面更长、更方，但背条斜线、局部小不齐与开口仍难解释为厚度还是折层。没有 ANTI_COVER/LOCAL_NOISE 相对 LONG 的明确整体视觉胜出。见[原记录](reviews/8c.md)。

这不是统一人工标尺下的盲评。各父由不同 AI 评阅者负责，标签频次仅是已记录观察的描述统计，不估计评阅一致性。可见接近不认证合法拓扑；蓝色 C 可遮挡冲突层。原源款式不是唯一正确答案。

## 全部 32 页查看账本

每页四模型均由下列原评阅者登记实际查看；原 CPU antiflatten/redirection 图审、历史 C200 图审不计入本轮 128 行。

| 请求 | 阶段 | 原评阅者 | 模型列数 | 页面 |
| --- | --- | --- | --- | --- |
| 73b369b9_n0 | RAW | AI_PROXY_SEAM_PORTABLE_CORE | 4 | [73b369b9_n0_RAW.png](gallery/73b369b9_n0_RAW.png) |
| 73b369b9_n0 | POST | AI_PROXY_SEAM_PORTABLE_CORE | 4 | [73b369b9_n0_POST.png](gallery/73b369b9_n0_POST.png) |
| 73b369b9_n1 | RAW | AI_PROXY_SEAM_PORTABLE_CORE | 4 | [73b369b9_n1_RAW.png](gallery/73b369b9_n1_RAW.png) |
| 73b369b9_n1 | POST | AI_PROXY_SEAM_PORTABLE_CORE | 4 | [73b369b9_n1_POST.png](gallery/73b369b9_n1_POST.png) |
| 73b369b9_n2 | RAW | AI_PROXY_SEAM_PORTABLE_CORE | 4 | [73b369b9_n2_RAW.png](gallery/73b369b9_n2_RAW.png) |
| 73b369b9_n2 | POST | AI_PROXY_SEAM_PORTABLE_CORE | 4 | [73b369b9_n2_POST.png](gallery/73b369b9_n2_POST.png) |
| 73b369b9_n3 | RAW | AI_PROXY_SEAM_PORTABLE_CORE | 4 | [73b369b9_n3_RAW.png](gallery/73b369b9_n3_RAW.png) |
| 73b369b9_n3 | POST | AI_PROXY_SEAM_PORTABLE_CORE | 4 | [73b369b9_n3_POST.png](gallery/73b369b9_n3_POST.png) |
| 3f741759_n0 | RAW | AI_PROXY_ROOT | 4 | [3f741759_n0_RAW.png](gallery/3f741759_n0_RAW.png) |
| 3f741759_n0 | POST | AI_PROXY_ROOT | 4 | [3f741759_n0_POST.png](gallery/3f741759_n0_POST.png) |
| 3f741759_n1 | RAW | AI_PROXY_ROOT | 4 | [3f741759_n1_RAW.png](gallery/3f741759_n1_RAW.png) |
| 3f741759_n1 | POST | AI_PROXY_ROOT | 4 | [3f741759_n1_POST.png](gallery/3f741759_n1_POST.png) |
| 3f741759_n2 | RAW | AI_PROXY_ROOT | 4 | [3f741759_n2_RAW.png](gallery/3f741759_n2_RAW.png) |
| 3f741759_n2 | POST | AI_PROXY_ROOT | 4 | [3f741759_n2_POST.png](gallery/3f741759_n2_POST.png) |
| 3f741759_n3 | RAW | AI_PROXY_ROOT | 4 | [3f741759_n3_RAW.png](gallery/3f741759_n3_RAW.png) |
| 3f741759_n3 | POST | AI_PROXY_ROOT | 4 | [3f741759_n3_POST.png](gallery/3f741759_n3_POST.png) |
| 85b16941_n0 | RAW | AI_PROXY_SEAM_EVIDENCE | 4 | [85b16941_n0_RAW.png](gallery/85b16941_n0_RAW.png) |
| 85b16941_n0 | POST | AI_PROXY_SEAM_EVIDENCE | 4 | [85b16941_n0_POST.png](gallery/85b16941_n0_POST.png) |
| 85b16941_n1 | RAW | AI_PROXY_SEAM_EVIDENCE | 4 | [85b16941_n1_RAW.png](gallery/85b16941_n1_RAW.png) |
| 85b16941_n1 | POST | AI_PROXY_SEAM_EVIDENCE | 4 | [85b16941_n1_POST.png](gallery/85b16941_n1_POST.png) |
| 85b16941_n2 | RAW | AI_PROXY_SEAM_EVIDENCE | 4 | [85b16941_n2_RAW.png](gallery/85b16941_n2_RAW.png) |
| 85b16941_n2 | POST | AI_PROXY_SEAM_EVIDENCE | 4 | [85b16941_n2_POST.png](gallery/85b16941_n2_POST.png) |
| 85b16941_n3 | RAW | AI_PROXY_SEAM_EVIDENCE | 4 | [85b16941_n3_RAW.png](gallery/85b16941_n3_RAW.png) |
| 85b16941_n3 | POST | AI_PROXY_SEAM_EVIDENCE | 4 | [85b16941_n3_POST.png](gallery/85b16941_n3_POST.png) |
| 8c9b2023_n0 | RAW | AI_PROXY_SEAM_CODE_AUDIT | 4 | [8c9b2023_n0_RAW.png](gallery/8c9b2023_n0_RAW.png) |
| 8c9b2023_n0 | POST | AI_PROXY_SEAM_CODE_AUDIT | 4 | [8c9b2023_n0_POST.png](gallery/8c9b2023_n0_POST.png) |
| 8c9b2023_n1 | RAW | AI_PROXY_SEAM_CODE_AUDIT | 4 | [8c9b2023_n1_RAW.png](gallery/8c9b2023_n1_RAW.png) |
| 8c9b2023_n1 | POST | AI_PROXY_SEAM_CODE_AUDIT | 4 | [8c9b2023_n1_POST.png](gallery/8c9b2023_n1_POST.png) |
| 8c9b2023_n2 | RAW | AI_PROXY_SEAM_CODE_AUDIT | 4 | [8c9b2023_n2_RAW.png](gallery/8c9b2023_n2_RAW.png) |
| 8c9b2023_n2 | POST | AI_PROXY_SEAM_CODE_AUDIT | 4 | [8c9b2023_n2_POST.png](gallery/8c9b2023_n2_POST.png) |
| 8c9b2023_n3 | RAW | AI_PROXY_SEAM_CODE_AUDIT | 4 | [8c9b2023_n3_RAW.png](gallery/8c9b2023_n3_RAW.png) |
| 8c9b2023_n3 | POST | AI_PROXY_SEAM_CODE_AUDIT | 4 | [8c9b2023_n3_POST.png](gallery/8c9b2023_n3_POST.png) |
