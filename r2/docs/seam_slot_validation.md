# SEAM 代码整理验证（2026-10-17）

本次验证的是源码整理、CPU 合同、独立安装和已保存证据的身份。没有加载权重、运行占用头/生成器/D0 前向、启动优化器更新、生成新网格或重新评分网格。CPU FM 单测仅对 12 个合成自由标量执行一次 autograd，不能据此声称完成新的训练或 BF16 神经计算验证。

## 实际完成的检查

| 检查 | 结果与范围 |
|---|---|
| 核心迁移 | 5 个模块与原 prototype 逐字节一致，SHA256 见包内 `source_manifest.json` |
| 本地 CPU 回归 | SEAM、原 D0 几何合同与来源绕序测试，**45 passed** |
| 独立 wheel | 离线 `pip wheel --no-deps --no-build-isolation ./r2` 成功；安装到独立目标目录，无覆盖现有环境 |
| 仓库外导入 | 在 `/tmp` 工作目录，仅将 wheel 安装目录放进 PYTHONPATH，SEAM **13 passed**；不是额外 13 个独立科学检验 |
| 安装内容身份 | 安装后的 5 个核心源码与原型 SHA256 一致，README 与来源 JSON 都被 wheel 包含 |
| 历史入口 | `train / generate / postprocess --help` 均成功；只解析帮助，不调用任务函数 |
| 原 NumPy 合同 | `check_corner_contract` 的 **38 项检查通过**，零神经调用 |
| 证据归档 | **45 份复制文件** SHA256 通过，含全部 16 张既有配对图；复制字节总量 6,855,881 |
| 原表复算 | 64 行主实验、192 行占用诊断和 32 行容量表通过；缓存 PREDICTED 与原 SEAM 数值一致 |
| 待提交树独立核验 | 仅从 Git 暂存树导出到新目录，13 项 SEAM 测试、证据哈希及三个历史帮助入口全部通过，未使用未提交实验源码 |
| Git 换行保护 | 原源码及原字节快照使用 `-text`，防止 Windows 自动换行转换破坏冻结身份；62 份源码过滤前后 Git 对象 hash 一致 |

测试环境是已有 Linux/WSL Python 3.10.21、NumPy 1.26.4；设置 `CUDA_VISIBLE_DEVICES=''`。不下载依赖或权重。原始 GPU 实验的实际精度、预算和执行审计仍以归档原报告为准。

复查命令：

```bash
PYTHONPATH=r2/src CUDA_VISIBLE_DEVICES='' python -B -m pytest \
  r2/tests/test_seam_contract.py r2/tests/test_denoise.py \
  r2/tests/test_source_winding.py -q
python -B r2/docs/evidence/seam_slot_c200_v1/verify_evidence.py
```

独立 wheel 中可用的是 `meshflow_control.seam` 机制 API；历史 `r2.pilots...` 是仓库源码入口，不属于 wheel。历史训练仍需本地冻结数据和匹配状态，评分快照仅供审阅。此次没有重新执行完整 GPU 链来验证新的部署路径；不声称任意新机器已具备一键实验条件。

原型、历史权重、数据、旧报告与评分定义保持不变。维护默认仍为 START；SEAM 未自动接入公开 R2 的默认采样/去噪命令。此次上传只包含代码、文档、小型逐例证据与现成对照图。
