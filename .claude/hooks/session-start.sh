#!/bin/bash
# Session start hook: install all LIBERO curation pipeline dependencies.
# Safe to run multiple times (idempotent). Non-interactive.
set -euo pipefail

# Only run in remote environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

echo '{"async": true, "asyncTimeout": 300000}'

# ── System packages ───────────────────────────────────────────────────────────
# apt-get update first: the prebuilt index can be stale and 404 on libosmesa6.
echo "[session-start] Installing OSMesa for headless MuJoCo rendering..."
apt-get update -q 2>&1 | tail -2 || true
apt-get install -y -q libosmesa6-dev 2>&1 | tail -2

# ── Python packages ───────────────────────────────────────────────────────────
echo "[session-start] Installing Python dependencies..."
pip3 install --quiet --upgrade setuptools

# mujoco is pinned to 2.3.7: robosuite 1.4.0 calls mj_fullM() with the 2.3.x
# binding signature; mujoco 3.x changed it and raises a TypeError at env.reset.
# termcolor is an undeclared robosuite 1.4.0 runtime dependency.
pip3 install --quiet \
  torch \
  numpy \
  h5py \
  scikit-learn \
  "mujoco==2.3.7" \
  termcolor \
  easydict \
  matplotlib \
  cloudpickle \
  "gym==0.25.2" \
  einops \
  "hydra-core>=1.2" \
  bddl

# robosuite 1.4.0 is required — 1.5.x has a different module layout that breaks LIBERO
pip3 install --quiet "robosuite==1.4.0"

# ── Fetch LIBERO ──────────────────────────────────────────────────────────────
# The egress policy scopes git smart-HTTP to this repo only, so `git clone` of
# LIBERO is denied (403). Plain HTTPS GET to codeload is permitted, so fetch the
# source tarball instead and lay it out at the path the pipeline expects.
if [ ! -d "/home/user/LIBERO/libero" ]; then
  echo "[session-start] Fetching LIBERO source tarball..."
  curl -sSL -o /tmp/libero_src.tar.gz \
    https://codeload.github.com/Lifelong-Robot-Learning/LIBERO/tar.gz/refs/heads/master
  rm -rf /tmp/LIBERO-master
  tar xzf /tmp/libero_src.tar.gz -C /tmp
  rm -rf /home/user/LIBERO
  mv /tmp/LIBERO-master /home/user/LIBERO
  rm -f /tmp/libero_src.tar.gz
else
  echo "[session-start] LIBERO already present at /home/user/LIBERO"
fi

# ── Configure LIBERO paths ────────────────────────────────────────────────────
mkdir -p ~/.libero
python3 - <<'PYEOF'
import yaml, os
libero_path = '/home/user/LIBERO/libero/libero'
config = {
    'benchmark_root': libero_path,
    'bddl_files':   os.path.join(libero_path, 'bddl_files'),
    'init_states':  os.path.join(libero_path, 'init_files'),
    'datasets':     os.path.join(libero_path, '../datasets'),
    'assets':       os.path.join(libero_path, 'assets'),
}
cfg_path = os.path.expanduser('~/.libero/config.yaml')
with open(cfg_path, 'w') as f:
    yaml.dump(config, f)
print(f"[session-start] LIBERO config written to {cfg_path}")
PYEOF

# ── Persist MuJoCo/OpenGL env vars ────────────────────────────────────────────
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo 'export MUJOCO_GL=osmesa' >> "$CLAUDE_ENV_FILE"
  echo 'export PYOPENGL_PLATFORM=osmesa' >> "$CLAUDE_ENV_FILE"
  echo "[session-start] MuJoCo env vars written to CLAUDE_ENV_FILE"
fi

echo "[session-start] All dependencies installed successfully."

# ── Auto-resume the contamination-boundary sweep ──────────────────────────────
# The container is ephemeral; whenever it restarts this hook fires, so resume the
# long sweep here if it isn't finished. Idempotent (won't double-launch).
SWEEP_DAEMON="$CLAUDE_PROJECT_DIR/run_sweep_daemon.sh"
if [ -f "$SWEEP_DAEMON" ]; then
  echo "[session-start] Ensuring contamination-boundary sweep is running..."
  bash "$SWEEP_DAEMON" || true
fi
