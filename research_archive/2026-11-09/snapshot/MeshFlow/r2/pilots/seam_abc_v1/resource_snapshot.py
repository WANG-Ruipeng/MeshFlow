"""Read-only environment snapshot; no torch/model initialization."""
from pathlib import Path
import json
import sys
import time
from r2.pilots.seam_quality_four_v1.common import resources

def main():
    root = Path(__file__).resolve().parents[3] / 'experiments/r2_seam_abc_v1'
    dest = root / 'resources' / (sys.argv[1] if len(sys.argv) > 1 else str(time.time_ns()))
    dest = dest.with_suffix('.json')
    dest.parent.mkdir(parents=True, exist_ok=True)
    value = resources()
    with dest.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2)
    print(json.dumps(dict(path=str(dest), gpu=value['gpu'], CPU_cores=value['CPU_cores'], RAM_available_bytes=value['RAM_available_bytes'])))

if __name__ == '__main__':
    main()
