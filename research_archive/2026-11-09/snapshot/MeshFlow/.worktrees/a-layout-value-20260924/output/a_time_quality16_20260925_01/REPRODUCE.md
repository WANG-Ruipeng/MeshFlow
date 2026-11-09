# 本次运行记录与复核入口

环境：Ubuntu WSL，/home/warpwang/miniforge3/envs/meshflow/bin/python，Python 3.10.21，Torch 2.7.1+cu128。当前实验只用 CPU，torch 4 线程 / interop 1 线程；不把 RTX 5080 性能算入本结果。

工作树：/mnt/e/MeshFlow/.worktrees/a-layout-value-20260924，HEAD bd7d8a138465c583694f9f86083ddd2979d10f87。官方保护源码保持原样；新增代码在 experiments/a_radial/time_quality16/。

已执行的命令（均从工作树根目录运行）：

```bash
PY=/home/warpwang/miniforge3/envs/meshflow/bin/python
$PY -m pytest experiments/a_radial/time_quality16 -q -p no:cacheprovider --junitxml=output/a_ttq_test_gate_20260925_01/pytest.xml
$PY -m experiments.a_radial.time_quality16.prepare --repo . --out output/a_time_quality16_20260925_01
$PY -m output.a_ttq_test_gate_20260925_01.integration_gate
$PY -u -m experiments.a_radial.time_quality16.launch --protocol output/a_time_quality16_20260925_01/RUN_PROTOCOL.json --gate output/a_time_quality16_20260925_01/GATE.json
```

29 项测试通过，另有一个真实 Fixed case 验证端到端接入。真实接入检查在独立 gate 目录中，不进入正式144次计时账本。首次增加的一个契约测试曾因测试夹具缺字段失败，修正夹具后全套通过；未据此改变几何算法。测试 XML 与真实接入 TASK 由 GATE.json 绑定。

prepare 生成独立的模型纯权重文件和每对象输入文件，核对原权重张量哈希与同主场参考数组；不含教师布局或缓存的代理点。RUN_PROTOCOL.json 冻结源代码、支持模板、16 个输入、原始和便携权重的哈希。主运行前后都验证绑定未变。

这是一条不覆盖已有尝试的运行链。再次启动现有目录会被拒绝；新一轮必须使用新目录，重新 prepare、完成对应测试及真实接入 gate，再运行 launch。当前 integration_gate.py 记录的是本次固定目录，重现到其他目录时应保留原脚本并复制修改新脚本的 run/gate 路径，不能覆盖本次证据。

质量目标：实际恰好512面、精确端点有效性、三角形最小角和三角形最小角p05均不低于原始 Fixed（容差1e-6度），两次独立面积MC的同主场RMS均比未经优化的Fixed降低至少2%。参考是已冻结2050节点/4096面径向主场，不是原始资产表面。所有方法使用相同原优化器、种子和16步上限；独立MC只观察，不选择优化器候选或改变停机。

审计网格0/1/2/4/8/16。主TTQ记录首次完成达标审核的实测累计时间，包括前面所有审核。完整时长包含本协议的全部16步、基准审核与六次观察审核、最终重新查询核对；磁盘证据写入和源码预检在每case计时外，并包含在driver总运行时间中。启动记录是Python入口到ready，不包含OS创建进程的时间；不作冷启动声明。

COMPLETE.json 只表示144次尝试都完成。必须结合 SUMMARY.json 的失败、技术问题、FINAL_VERIFICATION.json 与 RESOURCES.json 判断账本有效性。重复是计时重复，不增加父对象和训练种子数量；不能把成功子集的平均速度作为总体加速。