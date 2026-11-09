"""Read-only concise live status; no model imports or writes."""
from pathlib import Path
import json,datetime
OUT=Path('/mnt/e/MeshFlow/experiments/r2_user_data_scaling_32_67_v1')
def main():
    result={'local_time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds'),'arms':{}}
    for arm in ['USER_SMALL','USER_EXPANDED_67']:
        p=OUT/'training'/arm/'status.json'
        r=json.loads(p.read_text()) if p.exists() else {}
        result['arms'][arm]={k:r.get(k) for k in ['global_step','status','update_seconds','input_seconds']}
        result['arms'][arm]['failed']=r.get('calls',{}).get('failed',{})
        result['arms'][arm]['saved_step']=r.get('checkpoint',{}).get('global_step') if r.get('checkpoint') else None
    p=OUT/'controller/status.json'
    if p.exists():
        saved=json.loads(p.read_text())
        if saved.get('status')!='RESOURCE_BLOCKED' or not (OUT/'controller/owner.lock').exists():result['controller']=saved
        else:result['preserved_previous_stop']='RESOURCE_BLOCKED; active recovery owns controller lock; see live phase counts'
    ev=OUT/'evaluation';progress={}
    for label in ['USER_SMALL','USER_EXPANDED_67','HIST_USER_MIX_FROZEN']:
        p=ev/'sampling'/label/'progress.json'
        if p.exists():
            q=json.loads(p.read_text());progress[label]={k:q.get(k) for k in ['completed','registered','seconds']}
    p=ev/'D0/progress.json'
    if p.exists():progress['D0']=json.loads(p.read_text())
    progress['score_receipt_count']=sum(len(list((ev/'scores'/label).glob('*/*/score.json'))) for label in ['USER_SMALL','USER_EXPANDED_67','HIST_USER_MIX_FROZEN']) if (ev/'scores').exists() else 0
    progress['render_image_count']=len(list((ev/'gallery').glob('*/*.jpg'))) if (ev/'gallery').exists() else 0
    for name in ['raw_index','post_index','score_completion']:
        p=ev/(name+'.json')
        if p.exists():
            q=json.loads(p.read_text());progress[name]={'status':q.get('status'),'rows':len(q.get('rows',[])),'stages':q.get('stages')}
    result['evaluation_progress']=progress
    print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
