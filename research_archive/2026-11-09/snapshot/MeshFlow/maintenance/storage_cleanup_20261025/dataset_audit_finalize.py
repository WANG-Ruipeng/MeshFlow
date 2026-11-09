from pathlib import Path
import json, hashlib
O=Path('/mnt/e/MeshFlow/maintenance/storage_cleanup_20261025')
read=lambda n:json.loads((O/n).read_text())
inv=read('dataset_root_sizes.json');deps=read('dataset_active_dependencies.json');pkg=read('dataset_archived_package_whitelist.json')
weight_ext={'.pt','.pth','.safetensors','.ckpt'}
weights=sum(sum(v for k,v in r['extension_bytes'].items() if k in weight_ext) for r in inv['rows'])
protect=[
 {'root':'experiments/r2_bench_user_data_v2','why':'Current Native22 full-seat cases directly reference this original rebuilt dataset.'},
 {'root':'experiments/r2_bench_user_data_v2_expansion_v1','why':'Current Native/Coverage targets and the backend16 training-parent sources depend on this 25-parent/81-condition dataset.'},
 {'root':'experiments/r2_bench_user_relation_rebuild_v1','why':'943/9e full-seat cases remain in current evaluation and data lineage.'},
 {'root':'experiments/r2_bench_full_seat_curriculum_v1/data','why':'Retained Native6000 and SEAM3000 exact sources, evaluation inputs and fixed training schedule.'},
 {'root':'experiments/r2_bench_seat_coverage_v1/data and evaluation','why':'Retained Coverage1500 sources plus 30 original RAW meshes shared by later backend experiments.'},
 {'root':'experiments/r2_bench_backend_shape_preservation_v1/data','why':'Relative-deformation branch directly reuses the exact frozen 2000 training arrays and source validation inputs; not independent obsolete copies.'},
 {'root':'experiments/r2_bench_backend_relative_deformation_v1/data','why':'Frozen target-normalization caches are necessary to recompute this loss and analysis.'},
 {'root':'experiments/r2_bench_raw_local_repair_v1/data and targets','why':'Latest accepted/rejected patch, UNKNOWN/EDIT and source-corner evidence; not validated replacement training ground truth.'},
 {'root':'downloaded_data/archives/shapenet-cls.tar.gz and experiments/mf_h1_local0/assets/shapenet-03001627','why':'Official source archives/original geometries. The archive is only25.48MB; retains cross-category source supply.'},
 {'root':'Historical C/Y, fixed noise, query, RAW/POST, source IDs, scores and reports','why':'Original evidence remains protected by user preferences. A closed/negative experiment alone does not prove its data is redundant.'},
]
keep_large=[
 {'root':'experiments/r2_user_mix_from_official_v1/data','bytes':2677819450,'decision':'KEEP_UNIQUE_REGISTERED_TRAINING_ARRAYS','why':'6000 batch NPZ contain 48000 exact paired examples,96000 materialized Gaussian/epsilon arrays and OT maps. Stream samples readonly and refuses missing batches; seeds/index alone are not a verified byte-identical archival copy. Not a redundant download/cache.'},
 {'root':'E:/MeshFlow-Control-artifacts/experiments/r2_user_condition_finetune_pilot_v1/data','bytes':842777406,'decision':'KEEP_UNIQUE_REGISTERED_TRAINING_ARRAYS','why':'Two historical paired CONT arms16000 examples and32000 materialized Gaussian/epsilon arrays. Deleting would retire exact replay and input audits, not just remove unused source copies.'},
 {'root':'native_t1/runs/chair_moment_channel_v2/precheck/debug','decision':'KEEP_UNIQUE_DEBUG_EVIDENCE','why':'Large NPZ are original numeric/precheck evidence; route retirement did not establish a retained exact archive for these files.'},
]
result={'status':'READ_ONLY_COMPLETE_NO_DELETIONS','selected_roots':len(inv['rows']),'selected_root_bytes':sum(x['bytes'] for x in inv['rows']),'weight_bytes_in_assets_excluded_from_data_total':weights,'selected_nonweight_bytes':sum(x['bytes'] for x in inv['rows'])-weights,'scope_note':'53 selected data/assets/input/cache roots in workspace and artifacts; not exhaustive disk/dataset enumeration. assets can contain protected official weights; do not interpret total as deletable data. Separate exact expanded-package audit is outside those53roots.','current_manifests_checked':len(deps['manifests']),'current_nonweight_refs':len(deps['nonweight_references']),'missing_current_refs':sum(not x['exists'] for x in deps['nonweight_references']),'safe_primary_dataset_deletion_candidates':[],'safe_derived_package_whitelist':'dataset_archived_package_whitelist.json','safe_derived_package_count':pkg['count'],'safe_derived_package_bytes':pkg['bytes'],'protected_roots':protect,'large_data_review':keep_large,'prior_cleanup':'20261022 already removed exact duplicate ZIP-validation expansion and r2/build; no new benefit from re-listing them.','action_boundary':'Parent owns any deletion. All original datasets and evidence untouched. No model, training, generation, scoring, moving or deletion performed by this reviewer.'}
(O/'dataset_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
lines=['# 数据与派生缓存清理审计（2026-10-25）','',
'结论：当前没有确认可以无损删除的主训练/评测数据池。确认了一份完整保存在 ZIP 中的重复审阅包展开副本，可按逐文件白名单清理；本审计没有执行删除。','',
f"- 精确副本：`experiments/r2_single_step_denoiser_distillation_v1/review_package/` 中 **{pkg['count']} 个文件、{pkg['bytes']:,} bytes（{pkg['bytes']/2**20:.3f} MiB）**。",
'- 全部 1,023 个归档成员已经逐字节 SHA256 对照通过。保留 21,964,835-byte ZIP；其 SHA 与原 completion 一致。',
'- 保留展开目录中的 `PACKAGE_MANIFEST.csv`，因为旧完成核验入口仍读取它。删除其他白名单文件后，原完整包可从 ZIP 无损恢复，展开版网页浏览退役。不得整删原实验目录。',
'', '## 当前数据依赖', '',
f"读取 7 份当前冻结清单，解析 {len(deps['nonweight_references']):,} 条非权重引用（包括目录引用），全部存在。清单：`dataset_active_dependencies.json`。",'',
'Native22 完整座面数据直接引用旧 USER v2 / expansion 的 C/Y/case；Coverage24 又继承这些，并加入自身来源。943/9e 还依赖 RELATION 重建版本。不能因为这些数据目录日期较早就整包删除。','',
'SHAPE 后端的16训练父、2,000固定实例、15父来源验证被 RELDEF 逐字节复用。最新RAW局部修复还使用 Coverage30 RAW、官方/SHAPE/RELDEF proposals 和已有几何输出。上述数据均列保护。','',
'## 大目录并非重复缓存','',
'对 53 个选定 data/assets/input/cache 根做了实际尺寸统计。合计12,461,927,720 bytes，其中7,289,979,546 bytes是 assets 中受保护的官方Chair D0权重；实际非权重为5,171,948,174 bytes。此统计不包含所有源数据根，也不代表都能删除。详见 `dataset_root_sizes.csv/json`。','',
'最大的非权重目录为旧 USER_MIX/data（2,677,819,450B）和 CONT/data（842,777,406B）。源码确认它们保存精确训练Gaussian/epsilon、OT映射及输入，不是下载临时文件。删除会导致既有只读训练回放与输入核验失效；本轮未把它们列安全白名单。对应统计/计划/逐批attempt记录也不能冒充保留了完整数组。','',
'官方ShapeNet原归档仅25.48MB，且包含后续类别来源；原Chair源几何约47MB。保留原件的价值明显高于这点节省。MOMENT大debug NPZ也是独有数值证据，未核到完整备份，不列安全删除。','',
'20261022已删除ZIP校验临时展开树与r2/build，本轮不重复计算它们的释放量。所有RAW/POST、固定噪声、query、C/Y、source IDs、图审和原评分继续保留。','',
'## 执行边界','',
'仅CPU目录、源码、清单、归档SHA核查；没有模型调用、训练、生成、重新评分、移动或删除。父会话负责决定并执行精确清理白名单。']
(O/'dataset_audit.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps({k:result[k] for k in ['status','selected_roots','selected_nonweight_bytes','current_nonweight_refs','missing_current_refs','safe_derived_package_count','safe_derived_package_bytes']},indent=2))
