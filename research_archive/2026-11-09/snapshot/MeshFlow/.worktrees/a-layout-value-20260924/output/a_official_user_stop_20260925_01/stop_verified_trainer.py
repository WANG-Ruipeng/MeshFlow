import json, os, signal, pathlib, datetime
pid=5760
args=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
assert b'experiments.a_radial.system_formal.train' in args, args
assert args[args.index(b'--seed')+1]==b'10'
assert args[args.index(b'--run')+1]==b'/mnt/e/MeshFlow/.worktrees/a-layout-value-20260924/output/a_system_official_closed16_20260925_01'
os.kill(pid, signal.SIGINT)
record={'action':'SIGINT_exact_verified_trainer','pid':pid,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'requested_by_user':True,'checkpoint_20000_hash_verified_before_signal':True,'source_files_modified':False}
pathlib.Path('output/a_official_user_stop_20260925_01/SIGNAL_SENT.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record))