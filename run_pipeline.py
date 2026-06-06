"""
End-to-end pipeline: collect demos → train BC → evaluate.

Each stage runs in a subprocess to keep MuJoCo and PyTorch shared libraries
from conflicting within a single Python process (causes segfault on CPU builds).
"""

import argparse
import json
import os
import subprocess
import sys
import time
import yaml

PYTHON = sys.executable
LIBERO_PATH = '/home/user/LIBERO'
REPO_PATH   = '/home/user/structural-defect-curation'

ENV = {
    **os.environ,
    'MUJOCO_GL': 'osmesa',
    'PYTHONPATH': f'{LIBERO_PATH}:{REPO_PATH}',
}


def run(cmd, desc=''):
    print(f"\n{'='*60}")
    if desc:
        print(desc)
        print('='*60)
    proc = subprocess.run(cmd, env=ENV, cwd=REPO_PATH)
    if proc.returncode != 0:
        sys.exit(proc.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/libero_spatial.yaml')
    parser.add_argument('--skip-collect', action='store_true')
    parser.add_argument('--skip-train',   action='store_true')
    args = parser.parse_args()

    with open(os.path.join(REPO_PATH, args.config)) as f:
        cfg = yaml.safe_load(f)

    clean_demo  = 'data/raw/clean_demos.hdf5'
    cont_demo   = 'data/raw/contaminated_demos.hdf5'
    clean_ckpt  = 'results/bc_clean.pt'
    cont_ckpt   = 'results/bc_contaminated.pt'
    result_path = 'results/baseline_results.json'

    os.makedirs(os.path.join(REPO_PATH, 'data/raw'),  exist_ok=True)
    os.makedirs(os.path.join(REPO_PATH, 'results'),   exist_ok=True)

    # ── Step 1: collect ──────────────────────────────────────────────────────
    if not args.skip_collect:
        run([PYTHON, 'data/collect_demos.py', '--config', args.config],
            'STEP 1: Collecting demonstrations')
    else:
        print("Skipping demo collection (--skip-collect)")

    # ── Step 2a: train on clean ───────────────────────────────────────────────
    if not args.skip_train:
        run([PYTHON, '-c', f"""
import sys; sys.path.insert(0,'{REPO_PATH}')
import yaml, time
from methods.bc_policy import train
with open('{args.config}') as f: cfg = yaml.safe_load(f)
t0 = time.time()
train('{clean_demo}', '{clean_ckpt}', cfg['train'], success_only=True)
print(f'Training time: {{time.time()-t0:.1f}}s')
"""], 'STEP 2a: Training BC on CLEAN data (successful demos only)')

        # ── Step 2b: train on contaminated ────────────────────────────────────
        run([PYTHON, '-c', f"""
import sys; sys.path.insert(0,'{REPO_PATH}')
import yaml, time
from methods.bc_policy import train
with open('{args.config}') as f: cfg = yaml.safe_load(f)
t0 = time.time()
train('{cont_demo}', '{cont_ckpt}', cfg['train'], success_only=False)
print(f'Training time: {{time.time()-t0:.1f}}s')
"""], 'STEP 2b: Training BC on CONTAMINATED data (all demos)')
    else:
        print("Skipping training (--skip-train)")

    # ── Step 3: evaluate ─────────────────────────────────────────────────────
    for label, ckpt in [('CLEAN', clean_ckpt), ('CONTAMINATED', cont_ckpt)]:
        run([PYTHON, 'eval/evaluate.py',
             '--config', args.config,
             '--policy', ckpt,
             '--label',  label],
            f'STEP 3: Evaluating {label} policy')

    # ── Read per-policy results and report gap ────────────────────────────────
    # evaluate.py prints rates; collect them by re-running quickly without env
    results = {}
    for label, ckpt in [('clean', clean_ckpt), ('contaminated', cont_ckpt)]:
        out = subprocess.check_output([PYTHON, '-c', f"""
import sys; sys.path.insert(0,'{REPO_PATH}')
import json, torch
from methods.bc_policy import load_policy, load_dataset, obs_to_vec, OBS_KEYS
import h5py, numpy as np

model, obs_mean, obs_std = load_policy('{ckpt}')
print(json.dumps({{'label':'{label}', 'ckpt':'{ckpt}'}}))
"""], env=ENV, cwd=REPO_PATH)

    # Parse success rates from the evaluate.py stdout files that were printed
    # Simpler: just re-read from a temp JSON each evaluate.py writes
    print("\n" + "="*60)
    print("NOTE: see per-policy output above for success rates.")
    print("Run eval/evaluate.py --policy results/bc_clean.pt for clean rate.")
    print("Run eval/evaluate.py --policy results/bc_contaminated.pt for contaminated rate.")


if __name__ == '__main__':
    main()
