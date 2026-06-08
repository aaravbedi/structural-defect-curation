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
echo "[session-start] Installing OSMesa for headless MuJoCo rendering..."
apt-get install -y -q libosmesa6-dev 2>&1 | tail -2

# ── Python packages ───────────────────────────────────────────────────────────
echo "[session-start] Installing Python dependencies..."
pip3 install --quiet --upgrade setuptools

pip3 install --quiet \
  torch \
  numpy \
  h5py \
  scikit-learn \
  mujoco \
  easydict \
  matplotlib \
  cloudpickle \
  "gym==0.25.2" \
  einops \
  "hydra-core>=1.2" \
  bddl

# robosuite 1.4.0 is required — 1.5.x has a different module layout that breaks LIBERO
pip3 install --quiet "robosuite==1.4.0"

# ── Clone LIBERO ──────────────────────────────────────────────────────────────
if [ ! -d "/home/user/LIBERO" ]; then
  echo "[session-start] Cloning LIBERO..."
  git clone --depth=1 https://github.com/Lifelong-Robot-Learning/LIBERO.git /home/user/LIBERO
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
