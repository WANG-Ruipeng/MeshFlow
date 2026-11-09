import json
from pathlib import Path
r=Path('native_t1/runs/jedge5_blackbox_20261002_113600')
for f in ['geometry/endpoint_results.json','geometry/trajectory_results.json','edge/teacher_summary.json','edge/trajectory_summary.json','probes/B/results.json','probes/J/results.json']:
 a=json.loads((r/f).read_text());print(f, list(a))
 if 'timing' in a: print('timing',a['timing'])
 if 'seconds' in a:print('seconds',a['seconds'])
 if 'counts' in a:print('counts',a['counts'])
 if f.endswith('endpoint_results.json'):print('example row keys',list(a['rows'][0]))
 if f.endswith('trajectory_results.json'):print('firstrow',a.get('rows',[None])[0])
