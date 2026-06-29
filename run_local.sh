#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Run the contamination-boundary sweep on your OWN machine.
#
# Usage (from inside a clone of this repo, on the ral-contamination-boundary branch):
#
#     bash run_local.sh
#
# To keep it running after you close the terminal, wrap it:
#
#     nohup bash run_local.sh > sweep_console.log 2>&1 &     # detached
#   or run it inside tmux/screen.
#
# It is fully resumable: re-running continues from results/sweep/sweep_state.json,
# so losing/closing the machine never costs more than one in-progress unit.
#
# RECOMMENDED: a Linux machine (or Linux cloud VM). Headless MuJoCo rendering is
# reliable on Linux via OSMesa. macOS works but may need a GUI session for the
# GL context (see the macOS note below).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

# LIBERO lives next to the repo unless you point LIBERO_PATH elsewhere.
export LIBERO_PATH="${LIBERO_PATH:-$REPO/LIBERO}"

# ── 1. Python venv + dependencies ─────────────────────────────────────────────
if [ ! -d "$REPO/.venv" ]; then
  echo "[local] creating virtualenv at .venv"
  python3 -m venv "$REPO/.venv"
fi
# shellcheck disable=SC1091
source "$REPO/.venv/bin/activate"

echo "[local] installing Python dependencies (first run only; this takes a while)..."
pip install --quiet --upgrade pip setuptools wheel
# mujoco pinned to 2.3.7: robosuite 1.4.0 uses the 2.3.x mj_fullM signature.
# termcolor is an undeclared robosuite runtime dependency.
pip install --quiet \
  torch numpy h5py scikit-learn "mujoco==2.3.7" termcolor \
  easydict matplotlib cloudpickle "gym==0.25.2" einops "hydra-core>=1.2" bddl
pip install --quiet "robosuite==1.4.0"

# ── 2. Fetch LIBERO (source tarball; no git clone needed) ─────────────────────
if [ ! -d "$LIBERO_PATH/libero" ]; then
  echo "[local] downloading LIBERO into $LIBERO_PATH ..."
  curl -sSL -o /tmp/libero_src.tar.gz \
    https://codeload.github.com/Lifelong-Robot-Learning/LIBERO/tar.gz/refs/heads/master
  rm -rf /tmp/LIBERO-master
  tar xzf /tmp/libero_src.tar.gz -C /tmp
  rm -rf "$LIBERO_PATH"
  mv /tmp/LIBERO-master "$LIBERO_PATH"
  rm -f /tmp/libero_src.tar.gz
else
  echo "[local] LIBERO already present at $LIBERO_PATH"
fi

# ── 3. LIBERO config ──────────────────────────────────────────────────────────
mkdir -p "$HOME/.libero"
python3 - "$LIBERO_PATH" <<'PYEOF'
import sys, os, yaml
lp = os.path.join(sys.argv[1], 'libero', 'libero')
cfg = {'benchmark_root': lp,
       'bddl_files':  os.path.join(lp, 'bddl_files'),
       'init_states': os.path.join(lp, 'init_files'),
       'datasets':    os.path.join(lp, '../datasets'),
       'assets':      os.path.join(lp, 'assets')}
with open(os.path.expanduser('~/.libero/config.yaml'), 'w') as f:
    yaml.dump(cfg, f)
print('[local] wrote ~/.libero/config.yaml ->', lp)
PYEOF

# ── 4. Headless render backend ────────────────────────────────────────────────
if [ "$(uname)" = "Darwin" ]; then
  # macOS: MuJoCo uses the native GL backend. If env init fails with a GL/EGL
  # error, you likely need to run from a GUI Terminal session (not pure SSH).
  export MUJOCO_GL="${MUJOCO_GL:-glfw}"
  echo "[local] macOS detected — MUJOCO_GL=$MUJOCO_GL"
  echo "        (If you hit a rendering error, a Linux box / cloud VM is the safer host.)"
else
  # Linux: OSMesa for headless offscreen rendering. Install once if missing:
  #   sudo apt-get update && sudo apt-get install -y libosmesa6-dev
  export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
  echo "[local] Linux detected — MUJOCO_GL=$MUJOCO_GL"
fi

# ── 5. Run the resumable sweep ────────────────────────────────────────────────
echo "[local] starting sweep — progress in sweep.log (N/108). Ctrl-C is safe (resumable)."
exec python sweep_contamination_boundary.py
