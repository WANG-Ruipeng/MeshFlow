# GEO_C 后处理正式入库完成

提交并推送：8961b7a41fc153b5b45f0c8903370c6bfd1cf04f，origin/master已查询核对一致。

仅提交13份源码、测试、文档。固定几何算法的7个函数AST与原实现一致；32份既有输入的416个数组逐字节相同，逐面记录及除耗时统计相同；97个旧资产hash不变。公共CPU单元测试130/130通过，其中21项为新增几何和IO测试。封锁torch/models/experiments导入的真实CLI子进程完整导出成功，错误输入和既存输出返回非零。

新增命令：python -B -m native_t1 postprocess --input RAW.npz --known-faces K --out NEW_DIR。只处理既有FP32 RAW，K须为实际已知面数；另存NPZ/PLY及可追溯记录。采样默认RAW不变。未训练、未生成、未加载真实模型或重评分。

旧实验与报告仍只读保留；未提交任何权重、网格或实验数据。首次隔离测试fixture遗漏新轻量helper的失败记录保存在engineering，修正fixture后已通过，并非几何算法失败。详见DELIVERY_MANIFEST.json与各验证收据。
