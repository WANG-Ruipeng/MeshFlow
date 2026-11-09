"""Read-only receipt inspection. Never automatically replay an interrupted wave."""
import json
from .common import OUT, read
if __name__ == '__main__':
    completed=OUT/'COMPLETE.json'
    state=OUT/'workflow/state.json'
    print(json.dumps(read(completed) if completed.exists() else read(state) if state.exists() else dict(status='NOT_STARTED'),ensure_ascii=False,indent=2))
