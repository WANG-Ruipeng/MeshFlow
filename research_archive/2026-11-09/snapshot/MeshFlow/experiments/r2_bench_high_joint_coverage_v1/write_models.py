from r2.pilots.bench_high_joint_coverage_v1.common import OUT,NARROW_ROOT,read,sha,write_new
status=read(OUT/'training/WIDE_HIGH/status.json')
assert status['status']=='COMPLETE' and status['stage_step']==1500
rows=[]
for label,folder in [('WIDE_HIGH',OUT/'training/WIDE_HIGH'),('NATURAL_HIGH (read-only)',NARROW_ROOT/'training/NATURAL_HIGH')]:
    for name in ('generator_step1500','checkpoint_step1500'):
        r=read(folder/(name+'.json'));assert sha(r['path'])==r['sha256']
        rows.append((label,name,r))
lines=['# 模型与恢复身份','',
 'W为本轮唯一新增训练终点；B为上一轮已完成窄HIGH参照。两者从同一曝光分配B/global21000的live生成器、AdamW与RNG出发，各新增1500更新到global22500/native10500。本轮没有重训B。','',
 '|模型|文件用途|路径|SHA256|','|---|---|---|---|']
for label,name,r in rows:
    lines.append('|'+label+'|'+('推理权重' if name.startswith('generator') else '完整恢复状态')+'|'+r['path']+'|`'+r['sha256']+'`|')
lines+=['','W生成器状态SHA256：`'+status['inference_export']['generator_state_sha256']+'`。',
 '本轮固定W1500为结果，不挑中途点。rolling latest恢复点及各次保存hash有账本；最终完整状态含匹配AdamW和随机状态，旧源文件只读。',
 '原Native/T1+role+Geo、OT后HYBRID0.25、batch8/micro1、原FM全batch自由坐标归一。无SEAM、占用头、额外loss或教师训练。',
 '本表只声明资产身份，不声明整体质量晋级。现有Bench研究参考、Chair维护CLI默认和官方资产不变。没有cleanup、commit或push。','']
(OUT/'MODELS.md').write_text('\n'.join(lines),encoding='utf-8')
write_new(OUT/'model_index.json',dict(status='VERIFIED_SAVED_ENDPOINTS',models=[dict(label=l,kind=n,**r) for l,n,r in rows],new_training_arms=1,automatic_default_change=False))
print('MODEL_IDENTITIES_VERIFIED')
