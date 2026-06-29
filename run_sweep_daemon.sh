#!/bin/bash
# Idempotent watchdog: ensure the contamination-boundary sweep is running until
# all 108 units are done. Safe to call repeatedly (from the SessionStart hook,
# a scheduled job, or by hand) — it never double-launches and exits quietly when
# the sweep is already running or already complete.
#
# The sweep itself is resumable (results/sweep/sweep_state.json), and the
# workspace filesystem persists across container resets, so relaunching simply
# continues from the last completed unit.

cd /home/user/structural-defect-curation 2>/dev/null || exit 0

STATE=results/sweep/sweep_state.json
TOTAL=108

done=$(python3 -c "import json;print(len(json.load(open('$STATE'))['units']))" 2>/dev/null || echo 0)

if [ "$done" -ge "$TOTAL" ]; then
  echo "[daemon] sweep complete ($done/$TOTAL) — nothing to do"
  exit 0
fi

if pgrep -f "sweep_contamination_boundary.py" >/dev/null 2>&1; then
  echo "[daemon] sweep already running ($done/$TOTAL)"
  exit 0
fi

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
mkdir -p results/sweep
setsid nohup python3 sweep_contamination_boundary.py \
  >> results/sweep/full_run_console.log 2>&1 < /dev/null &
sleep 2
if pgrep -f "sweep_contamination_boundary.py" >/dev/null 2>&1; then
  echo "[daemon] relaunched sweep (resuming from $done/$TOTAL)"
else
  echo "[daemon] WARNING: relaunch attempt did not start a process"
fi
