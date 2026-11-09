"""Write completed model identities and final receipt; never runs an experiment."""
import argparse,json
from datetime import datetime,timezone
from r2.pilots.bench_group_weight_match_v1.common import OUT,read,sha,write_new

def model_index():
    s=read(OUT/'training/W_MATCHED/status.json')
    assert s['status']=='COMPLETE' and s['local_step']==1500 and s['global_step']==22500
    r=read(OUT/'training_registration.json')
    d=dict(status='RETAINED_RESEARCH_CANDIDATE_NOT_DEFAULT',label='M',arm='W_MATCHED',
        source_global_step=21000,local_updates=1500,global_step=22500,native_total_step=10500,
        EMA=False,source_checkpoint=r['source_checkpoint'],source_sha256=r['source_checkpoint_sha256'],
        inference=s['inference_export'],recovery=s['checkpoint'],
        training_registration_sha256=sha(OUT/'training_registration.json'),
        data='Original W30/HIGH6; exact12000instance replay, no added/removed parents',
        difference='Narrow B per-update group coordinate-MSE coefficients on the same W stream',
        generator_structure='Unchanged Native/T1 + role + Geo; no SEAM or auxiliary head',
        backend='Unchanged single official Bench D0/GEO_C/source winding/HINGE_OPP015/ALPHA/HOLE',
        promotion='See REPORT.md; artifact retention alone does not promote a default')
    write_new(OUT/'model_index.json',d)
    text='# 模型身份\n\nM / W_MATCHED 固定1500步终点，global22500 / native10500，live权重（非EMA）。\n\n'
    text+='- 推理导出：`'+d['inference']['path']+'`\n- SHA256：`'+d['inference']['sha256']+'`\n'
    text+='- 完整恢复状态：`'+d['recovery']['path']+'`\n- SHA256：`'+d['recovery']['sha256']+'`\n\n'
    text+='从共同P0恢复生成器、匹配AdamW和RNG；不从B或W的22500终点续训。结构与原W相同，唯一变化是组MSE系数。保留研究权重不等于晋级默认；结论见REPORT.md。\n'
    path=OUT/'MODELS.md'
    if path.exists():assert path.read_text(encoding='utf8')==text
    else:path.write_text(text,encoding='utf8')
    return dict(status='COMPLETE',inference_sha256=d['inference']['sha256'])

def finish():
    for p in ('execution_closeout/complete.json','training_closeout/complete.json','audits/evaluation_closeout.json'):
        assert read(OUT/p)['status'] in ('PASS','COMPLETE')
    assert read(OUT/'package_receipt.json')['status']=='COMPLETE'
    names=['REPORT.md','RUNBOOK.md','MODELS.md','model_index.json','training_registration.json',
           'execution_closeout/complete.json','training_closeout/complete.json','audits/evaluation_closeout.json',
           'source_archive/final_source_hashes.json','source_archive/complete.json','code.diff','package_receipt.json']
    result=dict(status='COMPLETE',experiment='R2_BENCH_GROUP_WEIGHT_MATCH_V1',
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        actual=dict(training_updates=1500,training_forwards=12000,training_backwards=12000,
                    new_RAW=46,new_Euler_forwards=2300,new_D0_forwards=46,
                    new_score_cells=138,reused_score_cells=276,actual_visual_cells=276),
        identity={n:sha(OUT/n) for n in names},
        default_changed=False,commit=False,push=False,cleanup=False,automatically_extended=False)
    write_new(OUT/'COMPLETE.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('models','complete'));a=p.parse_args()
    print(json.dumps(model_index() if a.action=='models' else finish(),ensure_ascii=False))
