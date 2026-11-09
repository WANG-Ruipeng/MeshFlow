# Main initial visual review / 主阶段初始视觉审阅

完整覆盖：8父 × 2比例 × 4噪声 × 6方法 = 384唯一输出；64请求、128原PNG（彩色/中性）。四位审阅者每人实际查看分配的32图，每图有四个实心视角及线框。合并者复核记录与哈希，不声称重新独自查看全部128图。

Complete coverage: 8 parents × 2 ratios × 4 noises × 6 methods = 384 unique outputs; 64 requests and 128 original colored/neutral PNGs. Each reviewer actually inspected its assigned 32 pages and five views per page. The merger verifies records/hashes, not a new personal inspection of all 128 pages.

初始记录先保存，再在本次合并中解盲。Method 1–6是规范显示行名；原文A–F保持原样，仅对应行1–6。JSON及CSV的initial_* / original_*保留全部原字段和措辞。未设分数、综合排名或显著性。开发背景已保留，本材料不是独立盲法人类研究。

Initial notes precede mapping access. Displayed Method 1–6 is canonical; original A–F text is retained as positional spelling only. JSON and initial_*/original_* CSV fields preserve every original field and text. No scores, composite ranks or significance claims were added. Prior implementation context is disclosed; this is not an independent blinded human study.

| Display | Model |
|---|---|
| Method 1 | HIST_FM6000 |
| Method 2 | R3_40_H_LATE |
| Method 3 | R2_MIX_H_ALL |
| Method 4 | R4_MIX_H_LATE |
| Method 5 | R1_40_H_ALL |
| Method 6 | START5000 |

Files / 文件：

- per_output.csv / .json: all 384 records; JSON also includes complete original request/support documents.
- per_parent_noise_summary.csv / .json and PARENT_NOISE_SUMMARY.md: all 16 parent/ratio descriptions.
- per_parent_noise_original_notes.json: all four noises for each method/condition, with original text.
- coverage.json, pre_unblind_verification.json, image_hash_verification.json, source_hashes.json: provenance and coverage.

排除：root/initial_notes.json只含初始5请求，不重复计入；secondary复查不计初始覆盖。schedule40_train/review_manifest.json保留旧分配状态，以实际逐请求记录及observations.json COMPLETE认证完成。gallery旧PENDING不改写，由本目录独立收据记录完成。

Excluded: root/initial_notes.json is an obsolete five-request partial; secondary rechecks are not counted. The train review_manifest retains its old assignment status, while completed request records and observations.json establish completion. Original gallery PENDING flags remain unchanged; this separate receipt records actual review completion.

限制：图片不能认证接缝、无相交/重复、精确C保持或语义用途。C自带尖片/缺口不自动归因于生成器。开放背板或不同椅型不按GT重建误差判坏。所有噪声和不利尾部保留。

Limits: pictures do not certify attachment, absence of intersections/duplicates, exact C retention or semantic use. Pre-existing input spikes/gaps are not automatically generator defects. Open backs and different chair styles are not penalized by source reconstruction. All noises and unfavorable tails remain included.
