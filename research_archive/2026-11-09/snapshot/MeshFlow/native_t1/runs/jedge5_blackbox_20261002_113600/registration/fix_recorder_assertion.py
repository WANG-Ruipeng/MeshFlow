from pathlib import Path
import json
p=Path('experiments/mf_h1_local0/jedge5_blackbox/recorder.py')
s=p.read_text(encoding='utf-8-sig').replace("    if np.any(velocities[:,:12]!=0):raise RuntimeError('Known velocity not exactly zero')\n",'')
s=s.replace("known_velocities_zero=True,euler_reconstruction_bitwise=True", "raw_known_velocity_zero=bool(np.all(velocities[:,:12]==0)),known_coordinates_clamped_bitwise=True,euler_reconstruction_bitwise=True")
p.write_text(s,encoding='utf-8')
f=Path('native_t1/runs/jedge5_blackbox_20261002_113600/trajectories/B/results.json')
r=json.loads(f.read_text());assert r['status']=='STOPPED' and r['counts']['model_forward_attempts']==50
f.rename(f.with_name('failed_attempt1.json'))
reg=Path('native_t1/runs/jedge5_blackbox_20261002_113600/registration')
protocol=json.loads((reg/'protocol.json').read_text())
protocol['amendment']='Recorder incorrectly required raw known-slot model velocity to be zero. Actual C remains clamped bitwise. Removed only observer assertion, unchanged sampler. One recorded failed attempt completed 50 model forwards/full rollout; repeat needed to store raw velocities lost by assertion. No new seed/context/model.'
protocol['failed_completed_rollouts']=1;protocol['complete_replays_including_failed_recording']=17;protocol['total_model_forwards_max']=1490;protocol['replay_forwards_including_failed_recording']=1250
(reg/'protocol_amendment1.json').write_text(json.dumps(protocol,indent=2)+'\n')
