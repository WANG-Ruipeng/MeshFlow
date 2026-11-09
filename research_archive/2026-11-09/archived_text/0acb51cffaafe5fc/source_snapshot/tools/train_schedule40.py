#!/usr/bin/env python3
"""Controller-owned schedule40 worker; this entrypoint never self-retries."""
from meshflow_control import runtime
import argparse
import json
import sys
from meshflow_control.training.schedule40 import train_segment, SEGMENTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--segment', required=True, choices=SEGMENTS)
    parser.add_argument('--official', required=True)
    parser.add_argument('--data-manifest', required=True)
    parser.add_argument('--stream-root', required=True)
    parser.add_argument('--teacher-registration', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--attempt-id', required=True)
    parser.add_argument('--device', default='cuda')
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument('--resume')
    recovery.add_argument('--fork')
    args = parser.parse_args()
    result = train_segment(**vars(args))
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result['status'] == 'COMPLETE' else 75


if __name__ == '__main__':
    sys.exit(main())
