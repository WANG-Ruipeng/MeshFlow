# Denoise 发布验证

本次维护入口已通过四项独立验证：**115 项 CPU 测试、60 例缓存迁移回归、一次真实官方 D0 前向、隔离 wheel 安装**。下面分别列出实际调用与证据，避免把工程回归计入历史研究预算。使用方式见[去噪接口](denoising.md)。

## 缓存迁移回归

[原完成收据](evidence/denoise_migration_completion.json)与[全部 60 例](evidence/denoise_migration_per_case.csv)记录：

| 范围 | 实际完成 |
|---|---:|
| 既有 POST → 来源 C 恢复 → free 定向 | 36 例，D0/RBF/BLEND 各 12 例 |
| 缓存 D0 proposal → 可选 RBF → GEO_C → 来源恢复/定向 | 24 例，D0 与 RBF 各 12 例 |
| 数组比较 | 4320 项，全部逐字节一致 |
| 受保护旧文件 | 199 份，哈希不变 |
| 实际 CPU 运算 | 60 次定向、24 次 GEO、12 次 RBF |
| 模型导入/前向、质量评分、新噪声 | 均为 0 |

比较包含原 32 个导出字段在恢复与最终阶段的结果；完整链另外比较 18 个 MODEL_POST 和 2 个 PRE_GEO 数组。旧单数键 `C_source_corner_permutation` 与新复数键之间只作登记过的名称映射；新增元数据不冒充旧字段。公开 CSV 保留全部行，只省去重复 condition ID 与机器资源/耗时列；request ID 仍标识条件和噪声。

这次迁移不加载父网格或 GT。旧参考此前已独立核对来源父及原顶点身份；新接口读取保存的映射，可信度仍为 `SAVED_MAPPING_ONLY`，不能声称每次都重新认证外部来源。16 个实际实现文件的 SHA256 与收据及 wheel 检查一致。

## CPU 测试与真实 D0

[CPU 收据](evidence/denoise_cpu_tests.json)：115 tests，0 failure、0 error、0 skip，JUnit suite 时间 20.670 秒。首轮 109 项通过、6 项失败，原因是临时目录放在仓库内触发既有训练入口的外部路径保护；换用仓库外临时目录后全部通过，未因此修改核心实现。

[真实 D0 收据](evidence/denoise_real_D0.json)：固定 USER_MIX 的 33ec 座面 medium、n0，N=218、K=6。真实官方 D0 前向开始和返回各 1 次，生成器前向、训练更新、新噪声、完整质量评分均为 0；另执行 1 次 GEO 和 1 次来源定向。

- 全 proposal 为 FP32，逐字节等于旧缓存，最大绝对误差 0。
- canonical GEO 的 18 个数组与旧 D0 结果一致。
- 最终来源恢复/定向的 46 个数组与缓存迁移结果一致。
- 受保护输入与实现哈希不变。

48.869 秒包括权重哈希、加载与 GEO，**不是模型延迟基准**。前向前重置后的 PyTorch 峰值为 allocated 1,879,434,240 bytes、reserved 1,910,505,472 bytes；不代表进程显存峰值。WSL 进程显存为 N/A；GPU 总使用量 4690 MiB 也不等于该 worker 峰值。只读 NumPy 转 tensor 产生过 warning；没有 tensor 原地写入，输入哈希不变，未重试。

## 隔离 wheel 安装

[wheel 收据](evidence/denoise_wheel_validation.json)与[全部 128 项数组检查](evidence/denoise_wheel_arrays.csv)均 PASS。wheel 为 181,038 bytes，SHA256：

```text
fb99d2c56af80c586a55c3d6ca271c193267f1c7a2b0615407524d89b05e94ce
```

通过 `pip --no-deps --target` 安装到独立目录，在仓库外临时工作目录、仅指向安装目录的 PYTHONPATH 下执行 help、recipe、prepare-condition、orient、缓存 D0 全链。没有从原仓库、native_t1 或历史 tools 导入运行模块；许可证、来源说明和源码清单均随包安装。

128 个恢复/最终数组逐字节一致，另有 PRE_GEO output/C 两项一致。该安装验证额外执行 **2 次定向、1 次 GEO、0 次 RBF、0 次模型调用**，与上面的 60 例迁移预算分列。五个 CPU CLI 不导入 torch；单独的源码身份检查导入 torch，但未构造模型或初始化 CUDA。16 个实现文件哈希与迁移版本一致，原输入和默认环境不变。

为保留已验证的上游源码字节，vendored denoiser 的既有行尾空白及许可证 CRLF 未格式化；不能宣称包含这些来源文件的完整 diff-check 没有空白提示。

## 来源、保持与指标合同

模型内部规范化角序和外部输入角序是不同合同。恢复来源顺序后固定 C 的该顺序与 FP32 坐标；free 可以翻转角序，逐角属性和源角点映射同步置换。顶点坐标、每面无序顶点集合、面数、实际无向连接不因此改变。

方向约束覆盖全部符合条件的双面、非退化索引邻接，包括 C/C；不会先按几何合法性挑选子图。矛盾分量整体不翻；无 C 锚分量只有确定性参考朝向，不能认证向外。非流形边及未决分量不获得全局流形、水密或朝外证书。

GEO_C 可能合并或拆分来源 C 的顶点关系，必须另报 `CHANGED_BY_GEO_C`；身份未知报 UNKNOWN。缺少可信来源排列时不得猜测原绕序，未知逐角属性不得静默丢弃或错误换序。

本轮不重新计算研究质量。数值见[研究结果](research_results.md)及[逐表 CF 合同](evidence/CF_metric_contracts.csv)：

- `CF_proper_affected_free_fraction` 只计 proper 穿越；`CF_illegal_affected_free_fraction` 计 proper 穿越或正面积共面非法重叠影响的 free 面并集。当前面板数值相等不表示定义相同。
- `FF_illegal_affected_free_fraction` 为非法 FF 面对影响的 free 面并集比例。RAW/POST 各用实际 free 面数，删面后分母可能改变。
- `ConflictCov_C` 为冻结 C 查询上的穿越影响带与近重复覆盖并集，保留 cross、dup 与交集；近重复检测可能包含合法薄壁。
- 小角率为 free 最小角严格小于 5°；不能只凭此数值判定细片多余。
- A-VMR/PMR 是原 C 开放边界顶点到 free 顶点的阈值匹配；严格模型阈值 1e-6，其余档用固定 L=9.208138264587333。匹配不等于实际共边。
- 实际共边、过滤非法/未决后的合法共边、方向一致共边，以及沿合法邻接可达的 free 面积分别报告。定向不创造接缝。
- 仅重排角序时，逐面几何等价核验后继承 CF/FF/I、距离、A-VMR/PMR 和角度值，并明确标注继承；方向统计重新检查。继承不记作新评分。

以上验证证明已维护入口复现既有算法与映射；不能单独证明新用户输入泛化、RBF 无害、接合成功或整体生成质量更好。原始收据和实现身份由[来源清单](evidence/source_manifest.json)追溯；公开收据省去机器路径、主机名和进程 ID。
