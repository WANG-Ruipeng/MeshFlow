**Control MVP48：结果入口与交接核对（Session A）**

A 的数据准备和交接已完成。要看最终结果、对比图或讨论实验结论，在 Session C 继续；训练进度和运行故障由 Session B 处理；数据划分、条件片、源面映射和坐标由 Session A 解释。

| 要查看的内容 | 文件入口 | 如何解读 |
|---|---|---|
| 数据与修复范围 | [DATA_REPORT.md](../data/DATA_REPORT.md) | 48对象、32/8/8、128训练条件、16验证条件、16测试条件及限制 |
| 冻结数据身份 | [DATA_READY.json](../DATA_READY.json) | repair_v2已冻结；这只代表数据准备完成 |
| 当前训练进度 | [training/status.json](../training/status.json) | 看status、phase、logical_step与actual_optimizer_updates |
| 训练交付说明 | [TRAIN_REPORT.md](../training/TRAIN_REPORT.md) | B记录训练、检查点与重载检查 |
| 最终结论和对比图入口 | [MVP48_REPORT.md](../evaluation/MVP48_REPORT.md) | 等C明确正式评价完成；等待状态下不是最终结果 |
| 每个测试对象的汇总 | [per_parent.csv](../evaluation/per_parent.csv) | 8个父对象等权比较；先核对执行状态和有效数量 |
| 逐case执行与指标 | [per_output.csv](../evaluation/per_output.csv) | 96条预登记请求不等于96条生成已完成 |
| 固定验证检查 | [teacher_validation.csv](../evaluation/teacher_validation.csv) | 三个固定终点共24次teacher；不用于选checkpoint |

完成顺序：A的DATA_READY → B发布根目录TRAIN_READY.json及T1/Geo/Full三个step1000终点，完成严格重载 → C按冻结协议执行32个case在三模型上的96条生成和24次teacher，并更新正式报告、有效数量及失败记录。不能仅凭报告文件或CSV已经存在判断完成。

本次只读检查时间：2026-10-02T09:40:42.574635+00:00。读取时B状态为TRAINING，阶段T1，logical_step=353，累计实际更新=353；真实数据预检查=PASS。C状态=CPU_READY_WAITING_TRAIN，CPU准备=PASS，正式评价完成标志=False。本段为时间快照；此后的进度以B/C各自状态文件为准。

A、B、C绑定的DATA_READY SHA256一致：b5cd88c3a7488c86c567324845cff63fb1757d7b37993c20bfd5a6524e20145e。B/C声明的81份冻结文件hash清单均与READY相同，本次额外读取并核验manifest、split、契约、三份任务计划、加载器、条件清单和数据报告，共9份关键文件；未重跑完整几何/近重复审计。

旧DATA_BLOCKED.json是严格原始版供给不足的历史证据，已由repair_v2 READY明确取代。早期data_to_B/C.md和train_dependency_audit.json仍包含当时的权重缺失描述；当前权重状态以[train_official_asset.json](train_official_asset.json)为准，运行状态以training/status.json为准。历史记录不改写。

结论的样本单位是8个独立测试父对象。同对象的两块条件和两份噪声是重复条件，不能当成32个独立对象。32训练与8验证来自官方train，可能已被官方无条件预训练见过；验证“未见”只指本轮条件微调。8测试来自官方test，但官方checkpoint具体预训练内容未独立认证。完整边界以冻结DATA_REPORT为准。

本说明属于冻结后的交接材料，不进入科学输入清单，不改变数据、模型、评价预算或执行规则。A本次未运行模型、OT、训练或评分，也未修改B/C代码。
