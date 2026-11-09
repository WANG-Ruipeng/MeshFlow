"""Lightweight progress companion; reads experiment records and updates one Markdown view."""
from pathlib import Path
import argparse,datetime,json,time,os
from zoneinfo import ZoneInfo

REPO=Path(__file__).resolve().parents[3]
QUEUE=REPO/'output/a_overnight_queue_20260925_01'
OFFICIAL=REPO/'output/a_system_official_closed16_20260925_01'
SINGLE=REPO/'output/a_margin_single_20260925_01'
OUT=Path('/mnt/e/MeshFlow/output/LIVE_TRAINING_20260925.md')

def read_optional(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else None

def update():
    now=datetime.datetime.now(datetime.timezone.utc);q=read_optional(QUEUE/'STATUS.json') or {}
    lines=['# 夜间训练实时状态','',f"更新时间：{now.astimezone(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')} 北京时间（每60秒自动刷新）",'',
        f"官方后台队列：**{q.get('status','UNKNOWN')} / {q.get('stage','FINAL')}**。"]
    if q.get('utc'):
        age=(now-datetime.datetime.fromisoformat(q['utc'])).total_seconds()
        lines.append(f'队列最后心跳距今 {age:.0f} 秒。'+(' **心跳已超过120秒，需要查看进程/日志；不是完成证明。**' if age>120 and q.get('status')=='RUNNING' else ''))
    lines+=['','|官方种子|最新记录步数|固定预算|最近100步均耗时|最近记录loss|','|---|---:|---:|---:|---:|']
    for seed in (10,11):
        p=read_optional(OFFICIAL/f'training/seed_{seed}/PROGRESS.json')
        if p:lines.append(f"|{seed}|{p['step']}|50000|{p['recent_100_update_mean_seconds']:.3f}s|{p['latest_loss']:.6g}|")
        else:lines.append(f'|{seed}|尚无100步进度记录|50000|—|—|')
    lines+=['','进度记录不是当前瞬时步数。不同batch的loss不能当作固定验证集；不据此早停或宣称收敛。完整检查点每5000步保存，两个种子完毕后自动进行固定最终EMA采样/评测。','']
    p=read_optional(SINGLE/'TRAINING_PROGRESS.json') or {};done=p.get('completed_jobs',[])
    lines.append(f"CPU教师配对诊断：完成 **{len(done)}/8** 个固定20000步任务。")
    if (SINGLE/'TRAINING_COMPLETE.json').exists():lines.append('CPU训练完整预算已完成；独立评测与科学解释见该目录。')
    if (SINGLE/'TRAINING_FAILURE.json').exists():lines.append('CPU任务有失败记录，请查看TRAINING_FAILURE.json，未自动重试。')
    if q.get('status')=='FAILED':lines+=['',f"队列错误：{q.get('error','详见FAILED.json')}"]
    lines+=['','人工实验总结：[OVERNIGHT_20260925.md](OVERNIGHT_20260925.md)','',
      '实时源记录：`E:\\MeshFlow\\.worktrees\\a-layout-value-20260924\\output\\a_overnight_queue_20260925_01\\STATUS.json`',
      '官方日志：同队列目录下 `logs/OFFICIAL_RUN_ALL_2X50000.log`。',
      '', '这份监视器只读取日志，不启动、停止或修改训练；16对象仍属于封闭开发集。']
    OUT.parent.mkdir(parents=True,exist_ok=True);tmp=OUT.with_name(OUT.name+'.tmp')
    tmp.write_text('\n'.join(lines)+'\n',encoding='utf-8');os.replace(tmp,OUT)
    return q.get('status') in ('COMPLETE_UNQUALIFIED','FAILED')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args()
    while True:
        done=update()
        if a.once or done:break
        time.sleep(60)