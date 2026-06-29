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
  echo "[daemon] sweep complete ($done/$TOTAL)"
  # Durably deliver the results: commit the CSVs to the branch if not already
  # pushed. This needs no Claude session — it runs from the hook on any restart.
  if [ -f results/sweep/results.csv ] && [ -f results/sweep/summary.csv ]; then
    if ! git diff --quiet --exit-code -- results/sweep/results.csv results/sweep/summary.csv 2>/dev/null \
       || ! git ls-files --error-unmatch results/sweep/results.csv >/dev/null 2>&1; then
      git add results/sweep/results.csv results/sweep/summary.csv
      git commit -q -m "Add contamination-boundary sweep results (108/108 units)" 2>/dev/null \
        && echo "[daemon] committed result CSVs"
      for i in 1 2 3 4; do
        git push -u origin ral-contamination-boundary 2>/dev/null && { echo "[daemon] pushed results"; break; }
        sleep $((2 ** i))
      done
    fi
    # Drop a completion marker a Claude session can notice and notify on.
    touch results/sweep/.COMPLETE
  fi
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
