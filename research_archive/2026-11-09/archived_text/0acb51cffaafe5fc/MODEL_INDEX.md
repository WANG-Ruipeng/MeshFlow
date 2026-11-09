# 本轮模型与检查点索引

默认仍为 START。下列是10个被评价的生成器实例；同名配方出现在 confirm 时是新的独立训练终点。

新矩阵、课程和确认都从同一官方权重起步，各自训练到6000步，使用新AdamW。课程并非从R2续训，确认也并非从旧课程续训。历史START/HIST的训练历史不同。

|实例|训练关系|实际绑定与完整SHA|
|---|---|---|
|main/START5000|历史条件训练累计5000；保留的默认参照|[绑定](phases/main/checkpoints/START5000.json)；d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80|
|main/HIST_FM6000|历史START继续FM至6000；实用参照|[绑定](phases/main/checkpoints/HIST_FM6000.json)；0d5adc6295bce68de6447e89db63959d36276e89561acc8b79e1c699fd2c5501|
|main/R1_40_H_ALL|官方起点，全程40%/H_ALL至6000；复用既有匹配FM_ONLY|[绑定](phases/main/checkpoints/R1_40_H_ALL.json)；4677a3f76e90f4f1934b45a91721312f6df709d683160de85b668a4fc47f654c|
|main/R2_MIX_H_ALL|官方起点，全程混合/H_ALL至6000|[绑定](phases/main/checkpoints/R2_MIX_H_ALL.json)；6770a575e03e7845a7b0f491d201dc61c3fc42a254fa949b84c498eb8ea149e8|
|main/R3_40_H_LATE|官方起点，全程40%/H_LATE至6000|[绑定](phases/main/checkpoints/R3_40_H_LATE.json)；b58f4637afd4341fb3c9a29e4bab4769bce5508485637d92e60e018882bdcd19|
|main/R4_MIX_H_LATE|官方起点，全程混合/H_LATE至6000|[绑定](phases/main/checkpoints/R4_MIX_H_LATE.json)；3ef70d83afb9287be2e633d1182feff1afa09ffa18f656cdee760acbd6a95159|
|course/C20_40_MIX|官方起点，20→40→混合/H_ALL至6000|[绑定](phases/course/checkpoints/C20_40_MIX.json)；2b5dd0a2c729fb2193ce5fb5385d60b12059e6a4a6a76664d910955c47fa104b|
|course/C40_20_MIX|官方起点，40→20→混合/H_ALL至6000|[绑定](phases/course/checkpoints/C40_20_MIX.json)；98e30126338e20bbacd2a37850eff18e4e9087c6aeaf151b6dc462aa83d01df7|
|confirm/C20_40_MIX|官方起点，新独立训练输入流，20→40→混合至6000|[绑定](phases/confirm/checkpoints/C20_40_MIX.json)；3c6cd7ed09e0a9db1c069acad3f31ba376b2b51373af042c05f76aabc70b733d|
|confirm/R2_MIX_H_ALL|官方起点，新独立训练输入流，全程混合至6000|[绑定](phases/confirm/checkpoints/R2_MIX_H_ALL.json)；9dc93885043121ea82287a105cb9ad3ab373827ce4c3b41802fcfe3b746f59a8|

完整路径和绑定来源见 [model_index.json](analysis/model_index.json)。权重留在本地，不进入审阅ZIP。
本文件只整理已完成评价的身份，不选择新默认，不加载或转换模型。
