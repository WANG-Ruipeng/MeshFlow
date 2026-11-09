# B TRAIN47 数据实核

已实际构造47 TRAIN及3 DEV来源包，旧4C原有case数组全部相等。无删坏父、无来源几何修复、无新评测噪声、无模型调用。

TRAIN共有400条C开放边、366条来源合法接口边、360个独立owner槽、6个alias返回bulk。third-in-C合法边18条，其中owner12条仍作为自由tip，不加GT锚。每父原始目标面恰好一次，N不变。

原SEAM最小代码不支持这些情况；本轮显式扩展label为owner-slot，不能称仅扩大父数而任何代码/标签都没变。所有来源对照和别名明细在B_SOURCE_CONDITIONS.csv、B_SOURCE_ALIAS_THIRD_C.csv。来源31/47有旧非法FF及3父UNKNOWN仍原样保留，不把数据宣称无交。

固定6400曝光，每父136或138；clean与x-width U(.9,1.1)各3200；恢复旧200步data RNG后按47父排列循环；新seed步距1000避免跨step重号，epoch边界同父重复用500+batch位置独立命名。t、z、epsilon、permutation、OT后HYBRID.25沿原算法。不能与LONG声称逐实例随机数组一一相同。

DEV资产在B_EVAL_DEV_ASSETS.json：3父×4既有噪声，直接取旧FULL请求z[K:]，C数组与原请求逐位一致；仅C和N进入预测占用布局。source标签单独文件，只用于训练/离线诊断。

OT缓存尚需data阶段实际完成；本文件是数据合同核验，不是训练结果。当前scope与先前R31四TRAIN接续信号不同，检验更多训练C能否在原4C及历史DEV3上兼顾接续与质量，不预设更广池必然防记忆或获胜。
