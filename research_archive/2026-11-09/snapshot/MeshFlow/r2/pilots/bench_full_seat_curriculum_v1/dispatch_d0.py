"""Resource-checked manual dispatch of an already registered independent D0 label."""
import argparse
import json
from .controller import dispatch, fresh
from .run import status

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', choices=('d0-common', 'd0-native', 'd0-seam'), required=True)
    args = parser.parse_args()
    states = status()
    if not all(states[k]['status'] == 'COMPLETE' for k in ('raw-common', 'raw-native', 'raw-seam')):
        raise RuntimeError('All registered RAW must be complete')
    if not fresh(states, args.job):
        print(json.dumps(states[args.job]))
    else:
        launched = dispatch(args.job, 6000., 'Root inspected real first D0 outputs, free VRAM and CPU load; conservative 6000 MiB increment, independent registered label only')
        print(json.dumps(dict(job=args.job, launched=launched, automatic_retry=False)))
