# HINGE V2：有限位置合并与真实三边环补面

独立 CPU 实验，读取28份 `PLAIN_S2000 + HINGE_OPP015`，不运行生成器、D0、训练或新噪声。结果位于 `experiments/r2_hinge_v2_cpu_v1/`。原数据、源码、评分器和结果只读。

冻结四臂：原中点再扫一遍、多α合并、仅三边环补面、多α后补面。多α只在原两自由簇代表点连线上取有限位置，原位置优先；C簇固定。位移上限仍为相对原PRE_GEO全角点的1.5%L。没有法向偏移、放大半径、edge flip或子段分裂。

补面只接受已有索引构成的真实三开放边环，三边均属于free且至少邻接一张SEAM，统一反向绕序，无新非法/未决接触。新增面标记BRIDGE；不伪造来源RAW身份，不计入原bulk面积，但可作为合法图通路。实际自由面分母、固定原父分母及BRIDGE统计并存。

恢复入口（WSL，仓库根目录）：

```bash
PYTHONPATH=/mnt/e/MeshFlow/r2/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 /mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B -m r2.tools.hinge_v2_cpu_v1.resume
```

完成后入口只验hash并退出，不追加扫描、处理或模型调用。未完成目录拒绝静默覆盖；工程恢复需显式记录。正式矩阵、图审、评分定义、资源账本、全部逐输出与逐父表、代码快照与hash均保留。

结果是有代价边界的局部正结果，不是普遍DEV接续解决方案；没有更改默认模型、正式后处理入口或Git提交。
