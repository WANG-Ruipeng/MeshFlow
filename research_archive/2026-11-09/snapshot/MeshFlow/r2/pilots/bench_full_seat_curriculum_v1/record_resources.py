"""CPU-only resource receipt for manually supervised stage boundaries."""
import argparse
import json
import re
import time

from .common import OUT, resources, write_new

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9_]+', args.tag):
        raise ValueError('Use a plain stage identifier')
    snapshot = resources()
    path = OUT / 'resources' / (str(time.time_ns()) + '_' + args.tag + '.json')
    write_new(path, dict(tag=args.tag, resources=snapshot,
                         model_calls=0, measurement_scope='Actual WSL device/host; not an isolated process peak'))
    print(json.dumps(dict(path=str(path), resources=snapshot), indent=2))
