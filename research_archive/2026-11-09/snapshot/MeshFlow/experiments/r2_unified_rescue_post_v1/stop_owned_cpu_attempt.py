import pathlib,json,os,signal,time
root=pathlib.Path(__file__).resolve().parent
p=next((root/'cells').glob('*/*/started.json'));pid=json.loads(p.read_text())['pid']
proc=pathlib.Path('/proc');cmd=(proc/str(pid)/'cmdline').read_bytes()
assert b'r2.pilots.unified_rescue_post_v1.run' in cmd
children=[]
for d in proc.iterdir():
    if not d.name.isdigit():continue
    try:
        if int((d/'stat').read_text().split()[3])==pid:children.append(int(d.name))
    except (FileNotFoundError,ProcessLookupError):pass
record=dict(reason='Fix protected same-face future original-coordinate collision',parent_pid=pid,verified_cmd=cmd.decode().replace('\x00',' '),own_children=children,model_retries=0)
(root/'stop_owned_cpu_attempt.json').write_text(json.dumps(record,indent=2))
for child in children:
    try:os.kill(child,signal.SIGTERM)
    except ProcessLookupError:pass
os.kill(pid,signal.SIGTERM)
print(json.dumps(record))
