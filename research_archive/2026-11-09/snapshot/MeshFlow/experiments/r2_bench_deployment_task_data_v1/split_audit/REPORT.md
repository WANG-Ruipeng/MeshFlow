# Bench 父级来源与三组评价准备

状态：只读盘点完成；没有冻结新的训练/评价数量，没有生成噪声或调用模型。

- 当前 RELATION/SHIFT 为26父91条C；其中7父属于 COMMON12000的64父，19父为该次后续适配新增。
- 旧48请求中的历史TRAIN4父与当前26父交集为0。旧面板不能说明本轮真实训练条件上的拟合。
- 旧DEV8父39条件保持 REGRESSION；其中7父官方test、1父官方train。它们是反复使用的开发证据，不重新命名独立测试集。
- 官方档案split为504 train /56 test；本轮只读两个split数组并核验档案hash。实际官方权重逐父曝光日志不可得，记UNKNOWN，不将test成员直接宣称为对所有预训练完全未见。
- 已有实际适配清单父并集为96；剩余456个UID可列入“未出现在已核对适配清单”的候选，尚未通过目标几何、人类输入语义或近重复审核，不是已就绪数据。
- 复用170父缓存精确几何hash。未缓存父的几何重复状态UNKNOWN；不同UID不等于独立款式。

后续使用 evaluation_contract.json 与 manifest_template.json：先定父级/精确重复组边界，再造C；真实训练C、同父预留C、适配未见父分别报告。预留C不能只是换ID或角序的训练C；新父需同时避开所选初始化的适配祖先。旧DEV另列REGRESSION。来源完整目标只用于监督和检查，不能作唯一生成成功答案。

冻结前运行 `python -B -m r2.data_tools.bench_deployment_tasks_v1.split_audit validate <manifest.json>`。该验证器只核对合同与身份，不能代替完整目标/条件图审。
