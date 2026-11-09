#!/usr/bin/env bash
set -euo pipefail
cd /mnt/e/MeshFlow/.worktrees/a-layout-value-20260924
queue=output/a_overnight_queue_20260925_01
test -f "$queue/PLAN.json"
test ! -e "$queue/STARTED.json"
test ! -e "$queue/controller.pid"
/usr/bin/nohup /usr/bin/setsid /home/warpwang/miniforge3/envs/meshflow/bin/python -B -m experiments.a_radial.system_formal.overnight_queue --run --queue "$queue" > "$queue/controller.log" 2>&1 < /dev/null &
queue_pid=$!
printf '%s\n' "$queue_pid" > "$queue/controller.pid"
printf 'Detached overnight queue PID=%s\n' "$queue_pid"