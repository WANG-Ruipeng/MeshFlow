# Control MVP48 — Session A 数据报告（repair_v2）

**READY：48对象、32/8/8划分、128训练C＋16验证C＋16测试C全部完成。** 用户授权修复后，本修订只清理精确零面积面与完全重复面；官方原文件只读，上一轮严格筛选阻塞证据完整保留。

| 划分 | 官方split | 对象 | 原面数 min/median/max | 保留面数 min/median/max | 条件数 |
|---|---|---:|---|---|---:|
|train|train|32|140/208/249|128/193.5/245|128|
|validation|train|8|136/212/256|129/191/246|16|
|test|test|8|154/204/212|132/184.5/206|16|

48对象中40个进行了清理，合计删除109个精确零面积面、571份完全重复三角形。没有移动/焊接/删除顶点，没有创造面、补洞、简化独立正面积面、细分或重网格。重复面保留最小原始face ID；所有有效三角形的位置不变，正面积三角形的几何并集保持。原始N和保留N均在128–256，N/K/free按保留面数定义。每个NPZ保留原始faces/vertices、删除原因、保留↔原始ID映射；这批数据应称“经授权最小清理的官方源派生数据”，不再称完全未经清理。

test供给已从严格版6对象恢复：本轮官方test原范围50个，保留面数合格41个，2条件适用36个，未曝光合格组34个。train相应为462/396/343，4条件适用329个。按预登记seed2026100201和组/UID哈希选择，不以模型表现或人工外观选择。

所有排除原因逐UID、逐起点记录于repair_candidate_audit.json；对象过滤计数（可重叠）为 `{"test": {"fewer_than_2_applicable_connected_patches": 5, "original_face_count_outside_128_256": 263, "retained_face_count_outside_128_256": 251}, "train": {"fewer_than_2_applicable_connected_patches": 53, "original_face_count_outside_128_256": 2360, "retained_face_count_outside_128_256": 2280}}`。原始/保留面数出界、极小非零面和缺少足量适用C均拒绝；已合格但未抽中的只是未选。异常跨known/free边拒绝patch而非无条件排除对象。严格水密、自交仍不是筛选门槛；自交未核验，未声称完美拓扑。

K=floor(.2N+.5)，共享完整源索引边FIFO BFS、几何分散起点。160块C在同对象内不同源面集合、不同起点位置，保存补集、坐标、面积比例、接口长度/边数和重合率。每条接口在清理后的拓扑及清理前原始拓扑中都恰好一known一free两面；删除重复/退化面不用于把异常原接口伪装为正常监督。

分组在划分前完成：继承全部严格版同源/精确/近重复分组，再添加所有3135个清理后几何的精确/规范化重复和528个范围/历史对象的139128对近重复检查。48对象来自48个不同最终组；历史20个开发父对象及最终组闭包不进入验证/test，P0/P1均不入选。近重复阈值为规范化顶点和面质心双向Hausdorff均≤.01；未旋转/镜像对齐或穷举语义家族。UID后缀仅是保守代理，不能声称完全无潜在近重复泄漏。

一次完整对象官方预处理，保留全部原始顶点保证bbox尺度约定；C/free共用变换。实际捕获pre_OT，严格核验`full_target=(pre_OT*2).astype(float32)`，禁止再次乘2或对C单独规范化。dense source_face_ids保持B/C行索引接口；original_source_face_ids显式对应原始面。完整GT、bbox、映射、接口仅用于监督/评分；load_condition只返回C、N和mask。

训练任务冻结1000×8=8000身份，每对象4C，128个组合各62或63次。训练任务序、Gaussian、t、排列、验证、测试、绘图/模型初始化随机域分离。测试8×2C×2noise=32case，三模型复用C/N/Gaussian共96生成；同对象两C复用自由Gaussian，不声称不同N有逐坐标对应。验证仅8对象p0、一份noise、t=.5，最终3端点共24只读teacher；不用于选checkpoint。本Session未生成模型输出。

曝光边界：32训练＋8验证来自官方train，可能已见于官方无条件预训练；验证只表示本轮条件微调未见。8测试来自官方test，但具体checkpoint预训练内容未独立认证。历史双对象调参模型不构成新对象独立证据。

独立CPU审计PASS：48对象/160条件、原始hash与最小清理映射、官方尺度、原始接口、分组/划分、全部任务流和随机规则均校验。详见independent_repair_validation.json；其中实际检查范围与耗时为准。0模型调用、0训练更新、0OT、0GPU worker；1 CPU构造worker/1数值线程，独立审计按24可用CPU与约61GiB WSL RAM并行。资源与阶段耗时在registration/data_repair_resources_*.json、data_repair_runtime.json；并发时间不当作隔离benchmark。

最新契约registration/data_contract_repair_v2.json，READY绑定其hash；B原接口签名/密集索引/尺度保持兼容，B应重新绑定最新READY。A未修改B/C代码或启动它们。旧DATA_BLOCKED原样保留为严格版历史，由带repair_v2标识的DATA_READY明确取代；旧报告/加载器在data/history/strict_v1/。不reset/clean、commit/push。
