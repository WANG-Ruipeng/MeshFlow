"""Read-only resource snapshots and completed-stage recovery inspection."""
import argparse,json,time
from . import common as C

def snapshot(tag):
    value=C.resources()
    C.write(C.OUT/'resources'/f'dispatch_{tag}_{time.time_ns()}.json',value)
    print(json.dumps(dict(tag=tag,gpu=value['gpu'],cpu_logical=value['cpu_logical'],load=value['loadavg'],
        memory_available=next((x for x in (value['memory'] or '').splitlines() if x.startswith('MemAvailable:')),None)),indent=2))
    return value

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('tag');a=p.parse_args();snapshot(a.tag)
