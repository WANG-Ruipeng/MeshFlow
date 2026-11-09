"""Resource inventory only; no model initialization."""
import json
import time
from .common import OUT,resources,write

if __name__=='__main__':
    value=resources()
    path=OUT/('resources_dispatch_'+str(time.time_ns())+'.json')
    write(path,value)
    print(json.dumps(dict(path=str(path),**value)),flush=True)
