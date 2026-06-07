#!/usr/bin/env python3
"""
End-to-end curation evaluation pipeline.

Uses subprocess isolation (collect/eval in MuJoCo process, train in PyTorch process)
to avoid segfaults on headless systems.

Step 0: Collect clean_demos.hdf5 (50 clean) and contaminated_demos.hdf5
        (20 clean + 60 defective) if not already present.
Step 1: Train BC on clean_demos.hdf5, 10 rollouts. Gate >= 50% or STOP.
Step 2: Score 80 contaminated demos with 7 metrics, curate top-75%,
        train BC for seeds [42, 0, 7], 30 rollouts each.
        Also run contaminated baseline and oracle (3 seeds each).
Step 3: Save results/libero_curation_results.json and print table.
"""

import sys, os, subprocess, json
import numpy as np
import h5py
from sklearn.metrics import roc_auc_score

ROOT    = os.path.dirname(os.path.abspath(__file__))
WORKER  = os.path.join(ROOT, '_worker.py')
PYTHON  = sys.executable
DATA    = os.path.join(ROOT, 'data', 'raw')
RES     = os.path.join(ROOT, 'results')
CLEAN   = os.path.join(DATA, 'clean_demos.hdf5')
CONT    = os.path.join(DATA, 'contaminated_demos.hdf5')
CONFIG  = os.path.join(ROOT, 'configs', 'libero_spatial.yaml')

EPOCHS      = 300
N_ROLLOUTS  = 30
SEEDS       = [42, 0, 7]
TOP_K       = 60        # top-75% of 80
N_CLEAN_IN_CONT = 20    # clean demos embedded in contaminated pool
N_DEFECT        = 60

METRIC_ORDER = [
    'smoothness', 'entropy', 'gripper_timing',
    'isolation_forest', 'ensemble', 'kNN', 'trajectory_alignment',
]


# ── subprocess helpers ─────────────────────────────────────────────────────────

def run_worker(extra_args, label):
    """Run _worker.py with extra_args, stream stdout, return parsed RESULT dict."""
    cmd = [PYTHON, WORKER, '--config', CONFIG] + [str(a) for a in extra_args]
    print(f"\n>>> {label}", flush=True)
    env = dict(os.environ)
    env.setdefault('MUJOCO_GL', 'osmesa')
    env.setdefault('PYOPENGL_PLATFORM', 'osmesa')
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=ROOT, env=env,
    )
    result_json = None
    for line in proc.stdout:
        line = line.rstrip('\n')
        if line.startswith('RESULT:'):
            result_json = line[7:]
        elif line.strip():
            print(f"  {line}", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Worker failed (exit {proc.returncode}): {label}")
    return json.loads(result_json) if result_json else {}


# ── HDF5 helpers ───────────────────────────────────────────────────────────────

def hdf5_subset(src, dst, indices):
    with h5py.File(src, 'r') as s, h5py.File(dst, 'w') as d:
        for new_i, old_i in enumerate(indices):
            s.copy(f'demo_{old_i}', d, name=f'demo_{new_i}')


def hdf5_info(path):
    with h5py.File(path, 'r') as f:
        keys = sorted(f.keys())
        n_suc = sum(1 for k in keys if f[k].attrs['success'])
    return len(keys), n_suc


def hdf5_successful_indices(path):
    with h5py.File(path, 'r') as f:
        return [int(k.split('_')[1]) for k in sorted(f.keys())
                if f[k].attrs['success']]


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(RES,  exist_ok=True)

    # ── STEP 0: Collect data if needed ────────────────────────────────────────
    if not os.path.exists(CLEAN):
        print("\n=== COLLECTING 50 CLEAN DEMOS ===")
        run_worker(['--task', 'collect',
                    '--n-clean', 50, '--seed-clean-start', 0,
                    '--out', CLEAN],
                   "collect 50 clean demos")
    else:
        print(f"\nFound {CLEAN}")

    if not os.path.exists(CONT):
        print("\n=== COLLECTING 80 CONTAMINATED DEMOS (20 clean + 60 defective) ===")
        run_worker(['--task', 'collect',
                    '--n-clean', N_CLEAN_IN_CONT, '--seed-clean-start', 2000,
                    '--n-defect', N_DEFECT,        '--seed-defect-start', 1000,
                    '--out', CONT],
                   "collect 80 contaminated demos")
    else:
        print(f"Found {CONT}")

    n_clean, n_clean_suc = hdf5_info(CLEAN)
    n_cont,  n_cont_suc  = hdf5_info(CONT)
    print(f"\nDataset summary:")
    print(f"  clean_demos.hdf5:        {n_clean} demos  ({n_clean_suc} successful)")
    print(f"  contaminated_demos.hdf5: {n_cont} demos  ({n_cont_suc} successful)")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1: Verify pipeline — clean BC, 10 rollouts
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*68)
    print("STEP 1: Verify pipeline — train clean BC, 10 rollouts (gate >=50%)")
    print("="*68)

    s1_ckpt = os.path.join(RES, 'step1_clean.pt')
    run_worker(['--task', 'train', '--data', CLEAN, '--ckpt', s1_ckpt,
                '--n-epochs', EPOCHS],
               f"Step 1: train clean BC ({n_clean} demos, {EPOCHS} epochs)")

    r1 = run_worker(['--task', 'eval', '--ckpt', s1_ckpt, '--n', 10],
                    "Step 1: eval clean BC (10 rollouts)")

    s1_rate = r1.get('rate', 0.0)
    print(f"\nStep 1 result: {s1_rate:.0%}  ({r1.get('n_success',0)}/10)")

    if s1_rate < 0.50:
        print("\n*** GATE FAILED: clean BC < 50% — container reset likely. STOP. ***")
        return

    print(f"Gate PASSED ({s1_rate:.0%} >= 50%). Proceeding to Step 2.")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2: Score + curate + train + eval
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*68)
    print("STEP 2: Curation eval — 7 metrics × 3 seeds × 30 rollouts")
    print("="*68)

    # Labels: first N_CLEAN_IN_CONT positions are "clean" (quality=1)
    labels = [1] * N_CLEAN_IN_CONT + [0] * N_DEFECT

    # Score 80 demos with 6 non-ensemble metrics
    r_sc = run_worker(['--task', 'score',
                       '--data', CONT, '--ref', CLEAN,
                       '--labels', json.dumps(labels)],
                      f"Score {n_cont} contaminated demos with 6 metrics")

    scores = r_sc['scores']   # dict: metric_name → list[float] (len 80)
    aurocs = r_sc['aurocs']   # dict: metric_name → float

    # Compute ensemble as 0.5*smoothness + 0.5*gripper_timing
    scores['ensemble'] = [0.5 * s + 0.5 * g
                          for s, g in zip(scores['smoothness'], scores['gripper_timing'])]
    aurocs['ensemble'] = float(roc_auc_score(labels, scores['ensemble']))
    print(f"  ensemble: AUROC={aurocs['ensemble']:.3f}")

    # ── 2a: Contaminated baseline (all 80 demos, 3 seeds) ────────────────────
    print(f"\n--- Contaminated baseline: all {n_cont} demos, {len(SEEDS)} seeds ---")
    cont_rates = []
    for seed in SEEDS:
        ckpt = os.path.join(RES, f'cont_baseline_s{seed}.pt')
        run_worker(['--task', 'train', '--data', CONT, '--ckpt', ckpt,
                    '--n-epochs', EPOCHS, '--seed', seed],
                   f"  cont baseline train seed={seed}")
        r = run_worker(['--task', 'eval', '--ckpt', ckpt, '--n', N_ROLLOUTS],
                       f"  cont baseline eval  seed={seed}")
        cont_rates.append(r['rate'])
        print(f"    seed={seed}: {r['rate']:.0%}", flush=True)
    cont_mean = float(np.mean(cont_rates))
    cont_std  = float(np.std(cont_rates))
    print(f"  => contaminated baseline: {cont_mean:.1%} ± {cont_std:.1%}")

    # ── 2b: Oracle (successful demos from contaminated pool, 3 seeds) ─────────
    oracle_idx  = hdf5_successful_indices(CONT)
    oracle_hdf5 = os.path.join(RES, 'oracle_demos.hdf5')
    hdf5_subset(CONT, oracle_hdf5, oracle_idx)
    print(f"\n--- Oracle: {len(oracle_idx)} successful demos from contaminated, {len(SEEDS)} seeds ---")
    oracle_rates = []
    for seed in SEEDS:
        ckpt = os.path.join(RES, f'oracle_s{seed}.pt')
        run_worker(['--task', 'train', '--data', oracle_hdf5, '--ckpt', ckpt,
                    '--n-epochs', EPOCHS, '--seed', seed],
                   f"  oracle train seed={seed}")
        r = run_worker(['--task', 'eval', '--ckpt', ckpt, '--n', N_ROLLOUTS],
                       f"  oracle eval  seed={seed}")
        oracle_rates.append(r['rate'])
        print(f"    seed={seed}: {r['rate']:.0%}", flush=True)
    oracle_mean = float(np.mean(oracle_rates))
    oracle_std  = float(np.std(oracle_rates))
    print(f"  => oracle: {oracle_mean:.1%} ± {oracle_std:.1%}")

    # ── 2c: Per-metric curation (top-75%, 3 seeds) ────────────────────────────
    metric_results = {}
    for mname in METRIC_ORDER:
        sc_arr   = np.array(scores[mname])
        top_idx  = np.argsort(sc_arr)[-TOP_K:].tolist()
        n_cl_sel = sum(1 for i in top_idx if i < N_CLEAN_IN_CONT)
        auroc    = aurocs[mname]
        print(f"\n--- {mname}  AUROC={auroc:.3f}  top-{TOP_K}: {n_cl_sel} clean, {TOP_K-n_cl_sel} defective ---")

        sel_hdf5 = os.path.join(RES, f'{mname}_sel.hdf5')
        hdf5_subset(CONT, sel_hdf5, top_idx)

        seed_rates = []
        for seed in SEEDS:
            ckpt = os.path.join(RES, f'{mname}_s{seed}.pt')
            run_worker(['--task', 'train', '--data', sel_hdf5, '--ckpt', ckpt,
                        '--n-epochs', EPOCHS, '--seed', seed],
                       f"  {mname} train seed={seed}")
            r = run_worker(['--task', 'eval', '--ckpt', ckpt, '--n', N_ROLLOUTS],
                           f"  {mname} eval  seed={seed}")
            seed_rates.append(r['rate'])
            print(f"    seed={seed}: {r['rate']:.0%}", flush=True)

        m_mean = float(np.mean(seed_rates))
        m_std  = float(np.std(seed_rates))
        vs_pp  = m_mean - cont_mean
        metric_results[mname] = dict(
            auroc=auroc, mean=m_mean, std=m_std,
            vs_baseline=vs_pp,
            n_clean_selected=n_cl_sel,
            seed_rates=seed_rates,
        )
        print(f"  => {mname}: {m_mean:.1%} ± {m_std:.1%}  vs_baseline={vs_pp:+.1%}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3: Save and report
    # ══════════════════════════════════════════════════════════════════════════
    results = {
        'step1_clean': {'rate': s1_rate, 'n_demos': n_clean, 'n_rollouts': 10},
        'contaminated_baseline': {
            'mean': cont_mean, 'std': cont_std,
            'n_demos': n_cont, 'seed_rates': cont_rates,
        },
        'oracle': {
            'mean': oracle_mean, 'std': oracle_std,
            'n_demos': len(oracle_idx), 'seed_rates': oracle_rates,
        },
        'metrics': metric_results,
    }
    out_path = os.path.join(RES, 'libero_curation_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # ── Print final table ─────────────────────────────────────────────────────
    print("\n" + "="*74)
    print("RESULTS TABLE")
    print("="*74)
    print(f"{'Metric':<22} {'AUROC':>6}  {'Downstream (%)':>17}  {'vs Baseline (pp)':>17}  Flag")
    print(f"{'-'*22} {'-'*6}  {'-'*17}  {'-'*17}  ----")

    for mname in METRIC_ORDER:
        r    = metric_results[mname]
        auroc = r['auroc']
        mu   = r['mean'] * 100
        sd   = r['std']  * 100
        vs   = r['vs_baseline'] * 100
        if r['vs_baseline'] > cont_std:
            flag = 'HELPS'
        elif r['vs_baseline'] < -cont_std:
            flag = 'HURTS'
        else:
            flag = 'NEUTRAL'
        print(f"{mname:<22} {auroc:6.3f}  {mu:6.1f} ± {sd:<6.1f}  {vs:>+17.1f}  {flag}")

    vs_oracle = (oracle_mean - cont_mean) * 100
    print(f"{'contaminated baseline':<22} {'—':>6}  "
          f"{cont_mean*100:6.1f} ± {cont_std*100:<6.1f}  {'0':>17}")
    print(f"{'oracle':<22} {'—':>6}  "
          f"{oracle_mean*100:6.1f} ± {oracle_std*100:<6.1f}  {vs_oracle:>+17.1f}")
    print("="*74)
    print(f"\nFlag key: HELPS = mean > baseline + 1σ  |  HURTS = mean < baseline - 1σ  |  NEUTRAL = within 1σ")


if __name__ == '__main__':
    main()
