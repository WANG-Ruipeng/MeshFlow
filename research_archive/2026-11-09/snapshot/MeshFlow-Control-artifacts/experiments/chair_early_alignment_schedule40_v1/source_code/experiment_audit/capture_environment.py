from pathlib import Path
from importlib import metadata
import sys,platform,json,subprocess,hashlib,os,time
out=Path(__file__).resolve().parent/'environment.json'
if out.exists(): raise RuntimeError('Environment record already exists')
versions={}
for name in ('torch','torchvision','numpy','scipy','trimesh','matplotlib','safetensors','timm','transformers','einops','meshflow-control'):
    try: versions[name]=metadata.version(name)
    except metadata.PackageNotFoundError: versions[name]='NOT_INSTALLED_AS_DISTRIBUTION'
r=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True)
data=dict(time_unix=time.time(),python=sys.version,executable=sys.executable,platform=platform.platform(),cpu_count=os.cpu_count(),packages=versions,nvidia_smi=r.stdout.strip(),nvidia_smi_exitcode=r.returncode,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model_imports=0,model_calls=0)
out.write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data))
