# 双对象拟合探针最终训练只读审计

结论：PASS。独立 T1_fit2_probe 已完成 probe0→1000 的1000次新增更新、8000个微批前向及反向；与三主臂审计合并，真实训练总计13000次 optimizer 更新和104000个微批前向/反向，没有重试或重复更新。这里只验收训练，不把留出评价或最终交付提前标为完成。

审计时刻：2026-10-02T23:06:01.443830+00:00。仅CPU读取状态/张量/文件哈希，数值线程2；没有GPU初始化、模型前向/反向、OT或optimizer调用，未修改任何冻结 .py。

## 输入与逐更新账本

- 固定输入仅为原训练集按UID排序前两个父对象：103b75dfd146976563ed57e35c972b4b、12c2b7080b07146a9d4d5dcd2cc33826；均使用已注册 p0，每个 effective batch 各4例，microbatch=1。没有验证或留出对象进入probe训练。
- fit_plan 恰好8000条，sample_index=0–7999、effective_batch_index=0–999、logical_step=1–1000，每条Gaussian/t/permutation种子均为冻结整数。保存的stream offset=0、end=1000；probe500对应sample_index=4000，probe1000对应8000，计划SHA一致。
- CSV、UPDATE_BEGIN、UPDATE_VALIDATED各1000条，step1–1000连续且唯一；逐行batch SHA相同，每步optimizer attempted/returned增加1，forward/backward attempted/returned均增加8。loss均有限，geometry=0且loss=FM。没有FAILURE、重复或同target恢复事件。
- 为遵守零额外OT，本审计没有重新生成训练样本；独立核对的是冻结计划记录、实际持久stream游标/哈希，以及CSV与事件中的实际batch哈希一致性。

## Fresh AdamW 与 checkpoint 身份

初始模型来自受保护的 MVP48 T1_step1000，文件身份与源清单一致；probe 的逻辑步数独立从0计起。冻结 _restore_source(probe=True) 建立make_optimizer并跳过源 optimizer.load_state_dict；元数据为FRESH_PROBE、optimizer_reset_at_1000=true，这个“1000”指源模型起点，不能误解为在probe终点再次重置。

实际保留的probe800完整latest中，全部AdamW状态step为800；最终probe1000全部AdamW状态step为1000。结合probe1–1000连续计数、500边界游标及两个分段summary，支持“源T1模型上fresh AdamW，probe内部继续恢复”，没有证据表明500步重新reset。超参数lr=1e-5、betas=(.9,.95)、weight_decay=0，loss=fm且无Geo，保持源定义。

| 独立probe步数 | 完整恢复状态 | 实际文件SHA256 |
|---:|---|---|
| 500 | False | 00ac2663ccab5ec136118b79f9e38f5d11fe5cff8f647af50151fb6da6f5c4c1 |
| 1000 | True | 6940d498f5a02c0148cf72d54941a998d3919ced2c8ad30884fbcbbd5dd83457 |

两份固定checkpoint均已实际CPU读取，模型张量哈希、variant/step/source/protocol/counts、RNG及stream哈希与sidecar一致。probe500为固定模型状态；probe1000包含完整AdamW/RNG/stream，latest指针准确指向该1000端点。CPU没有运行模型或optimizer恢复更新。

## 两个分段成本

两个 SEGMENT_END 与各自 summary 完全一致；每段500更新、4000微批及4000次CPU OT attempted/returned。CSV训练和输入准备时间分别求和等于summary，无同target恢复覆盖造成的成本缺口。

| 范围 | 新增更新 | 训练更新段秒 | 输入准备秒 | 保存校验秒 | 加载秒 | worker墙钟秒 |
|---|---:|---:|---:|---:|---:|---:|
| 0→500 | 500 | 543.218 | 11.483 | 54.602 | 13.594 | 635.094 |
| 500→1000 | 500 | 544.472 | 11.510 | 48.322 | 17.412 | 634.006 |

合计训练更新段1087.689秒、输入准备22.993秒、保存校验102.924秒、加载31.006秒、worker墙钟1269.100秒。计时范围沿用 MAIN_TRAINING_FINAL_AUDIT.md：训练更新段包含loss/反向/clip/AdamW/同步等；OT_seconds包含整个输入准备，均不能被当作纯forward或纯OT算法耗时。

## 合并训练预算

重新核对三主臂最终status、全部主更新事件，并重新计算三个主5000checkpoint的文件SHA，均与既有主训练审计一致：主12000更新+probe1000=13000；主96000微批+probe8000=104000，attempted和returned的forward/backward计数一致，optimizer_step_attempts=optimizer_updates=13000。所有训练已达到固定预算，不授权任何额外更新。

证据：PROBE_TRAINING_FINAL_AUDIT.json、probe_training_segments.csv，以及 MAIN_TRAINING_FINAL_AUDIT.json。剩余生成/teacher/条件替换/留出回归及交付仍由独立评价账本验收。
