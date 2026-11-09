"""Read active WSL process metadata without recording complete command lines."""
import json, os, subprocess
from pathlib import Path
from datetime import datetime, timezone
here=Path(__file__).resolve().parent
rows=[]
for line in subprocess.check_output(['ps','-eo','pid=,comm=,args='],text=True).splitlines():
    parts=line.strip().split(None,2)
    if len(parts)<3: continue
    pid,name,args=parts
    if int(pid)==os.getpid() or 'check_wsl_processes.py' in args: continue
    if any(t in name.lower() for t in ('python','train','torch')):
        rows.append(dict(pid=int(pid),name=name,references_project=any(t in args for t in ('MeshFlow','r2.','r2/')),references_training=any(t in args for t in ('training','train.py','torchrun'))))
assert not any(r['references_project'] and r['references_training'] for r in rows), 'Potential active project training: do not delete'
result=dict(utc=datetime.now(timezone.utc).isoformat(),processes=rows,project_training_found=False,full_command_lines_not_recorded=True)
(here/'wsl_process_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
