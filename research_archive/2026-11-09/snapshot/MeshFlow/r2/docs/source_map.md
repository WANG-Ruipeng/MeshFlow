# R2 源码来源

[迁入清单](source_manifest.json) 逐文件记录原独立工程路径、字节数及 SHA256；[骨干来源](model_source_map.json) 和 [数据/训练来源](data_training_source_map.json) 保留其上游来源记录。来源记录中的旧路径是溯源身份，不是新项目运行时导入依赖。

R2 新增的入口和维护层为：

- `__main__.py` 与 `src/meshflow_control/__main__.py`
- `src/meshflow_control/r2_cli.py`
- `src/meshflow_control/r2_spec.py`
- `src/meshflow_control/r2_runs.py`
- `src/meshflow_control/r2_checkpoints.py`
- `src/meshflow_control/r2_sampling.py`

核心模型、混噪、输入流、更新与 checkpoint 计算代码逐字节保留。若干历史 Python 文件本身混有少量 CRLF，不能通过统一换行“整理”。本目录 `.gitattributes` 禁止 Git 自动换行转换，CPU 测试也检查源文件哈希，防止 Windows checkout 后冻结输入身份失效。

测试只保留 R2 计算依赖涉及的协议测试与新增入口测试。旧通用 A/B/C trainer 的恢复测试未迁入；原工程对应文件未修改。此选择在迁入清单的 `omitted_tests` 中记录。

所有文件的初始迁移、真实权重与冻结输入验证证据保存在本地 `native_t1/maintenance/r2_port_20261009/`。维护文档中的通过结论只覆盖 [验证范围](validation.md)，不代表本轮重新执行了历史 GPU 实验。
