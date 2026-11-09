from pathlib import Path
p=Path('experiments/mf_h1_local0/jedge5_solver_pilot/worker.py');s=p.read_text(encoding='utf-8-sig')
s=s.replace("lock=open(info['lock_path'],'a+');waiting=time.perf_counter();print('WAIT_LOCK case',a.case,flush=True)\n    fcntl.flock(lock,fcntl.LOCK_EX)","lock_fd=os.open(info['lock_path'],os.O_RDWR);waiting=time.perf_counter();print('WAIT_LOCK case',a.case,flush=True)\n    fcntl.flock(lock_fd,fcntl.LOCK_EX)\n    info['lock_inode']=os.fstat(lock_fd).st_ino\n    info['lock_owner_pid']=os.getpid()")
s=s.replace('Keep descriptor live until process exit','Raw OS descriptor is deliberately not closed until process exit')
p.write_text(s,encoding='utf-8')
