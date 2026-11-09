R2_32_67_ANALYSIS_MINI_V1：仅由既有证据投影、配对、算术复算形成的小包。没有读取权重、加载模型、前向、生成、几何重评分、渲染或重新图审。旧实验及完整审阅包只读。

先读 `ANSWERS.md`；指标单位、原定义、分母、阈值和缺失状态见 `metric_dictionary.json`。`MANIFEST.json`、`FILE_INDEX.csv` 列出每个文件的大小、SHA256、CSV行数、字段和DERIVED标识；完整来源关系与已有/本次读取哈希在 `provenance.json`。其中来源ID使用各提取器独立命名空间（如training的T001），不得跨命名空间混用。

| 文件 | 行数/覆盖 |
|---|---:|
| metrics_per_output.csv | 840，完整96项原标量；NEW32 768、OLD6 72 |
| AVMR_PMR_all8.csv | 6720，840×8档，原表字节复制 |
| visual_review_840.csv | 840，全部原标签/备注；图页路径及已有哈希 |
| parents.csv | 105：共同32、新增35、DEV32、旧6身份单列 |
| conditions.csv | 264：训练194（其中SMALL91）、DEV64、旧6 |
| source_quality_and_interfaces.csv | 264，源C/Y质量与明确缺项 |
| source_events.json | 去重的已有源问题面对与逐面小角记录，非坐标/新评分 |
| exposure_by_condition.csv | 285，实际两臂曝光合计96000次 |
| training_steps.csv | 12000，两臂各6000；在线FM不是终点全池误差 |
| fixed_fixtures.csv / fixed_FM_diagnostics.csv | 16 / 128 |

`derived/` 的聚合、状态转移、接口配对、I分解及质量曝光全部为新算术结果；原表数值字符串保留原精度，不将UNKNOWN、UNCERTAIN、PARTIAL或缺失改为0。CSV中嵌入的小数组为JSON，应用标准CSV解析器读取。fraction=比例，乘100才是percent；两比例相减再乘100才是百分点。

在解压目录中，以Python 3.10+标准库运行（不需要原仓库、模型或绝对路径）：

```text
python protocol_and_code/recompute.py --out ../mini_recomputed
python protocol_and_code/summarize_quality_exposure.py --out ../mini_exposure_recomputed
```

这两个入口只读取包内CSV、写指定新目录。`protocol_and_code/original/` 为原实验实际代码快照的相关小文件，供定义审查；不要运行其中可能加载模型或写旧目录的原实验入口。`extract_*.py` 记录本次从本机资产提取的过程，需要原文件，仅属提取溯源；不是便携复算的前置条件。

输出主键是 `(request_id,label,stage)`；request_id绑定父、条件、噪声。来源数据按 `(parent_uid,condition_id)`，实际训练曝光再加arm。`USER_SMALL`/`USER_EXPANDED_67`/`HIST_USER_MIX_FROZEN`身份不合并。旧六父与新32父、RAW与POST均分开。

几何、图片及大权重有意不嵌入；它们的已有路径/哈希用于定位，不是本包复算依赖。只有原实验曾保存的边界、分母和状态才能继承，真实缺项见 `MISSING.md`。来源Y只是参考完成，不用于唯一GT款式排名。
