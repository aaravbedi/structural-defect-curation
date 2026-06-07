"""
Curation evaluation: score contaminated demos with 7 metrics, train BC on
top-50% subset, evaluate 20 rollouts, report AUROC and downstream success.
Also runs an oracle (only the actually-successful contaminated demos).
"""

import sys, os
sys.path.insert(0, '/home/user/LIBERO')
sys.path.insert(0, '/home/user/structural-defect-curation')
os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')

import json
import tempfile
import argparse
import numpy as np
import h5py
import torch
import yaml

from sklearn.metrics import roc_auc_score

from methods.curation_metrics import (
    STANDALONE_METRICS,
    FITTABLE_METRIC_CLASSES,
)
from methods.bc_policy import train as train_bc, load_policy, OBS_KEYS
from eval.evaluate import run_rollout

SEEDS = [42, 0, 7]
N_ROLLOUTS = 20
TOP_K_FRAC = 0.75  # keep top 75%


# ── data helpers ─────────────────────────────────────────────────────────────

def load_demos(hdf5_path):
    """Return list of (obs_dict, actions, success_bool) for every demo."""
    demos = []
    with h5py.File(hdf5_path, 'r') as f:
        for key in sorted(f.keys()):
            d = f[key]
            obs_seq = {k: d['obs'][k][:] for k in OBS_KEYS if k in d['obs']}
            actions = d['actions'][:]
            success = bool(d.attrs.get('success', False))
            demos.append((obs_seq, actions, success))
    return demos


def write_subset_hdf5(demos, path):
    """Write a subset of demos (obs_dict, actions, success) to HDF5."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, 'w') as f:
        for i, (obs_seq, actions, success) in enumerate(demos):
            grp = f.create_group(f'demo_{i}')
            obs_grp = grp.create_group('obs')
            for k, v in obs_seq.items():
                obs_grp.create_dataset(k, data=v)
            grp.create_dataset('actions', data=actions)
            # rewards not needed for training — write zeros
            grp.create_dataset('rewards', data=np.zeros(len(actions)))
            grp.attrs['success'] = bool(success)


# ── BC training and evaluation ───────────────────────────────────────────────
# MuJoCo's OpenGL context and PyTorch's allocator segfault when active together.
# Fix: train all checkpoints first (no env open), then open env only for rollouts.

def train_and_eval(demos, cfg, label='', n_seeds=3):
    """
    Train BC on `demos` with multiple seeds, evaluate, return mean ± std.
    Training and rollout evaluation are strictly separated to avoid the
    MuJoCo/PyTorch OpenGL segfault on headless systems.
    """
    train_cfg = dict(cfg['train'])
    train_cfg['n_epochs'] = 300

    ckpt_dir = tempfile.mkdtemp(prefix='bc_ckpts_')
    hdf5_path = os.path.join(ckpt_dir, 'subset.hdf5')
    write_subset_hdf5(demos, hdf5_path)

    # ── Phase 1: train all seeds (no LIBERO env open) ────────────────────────
    ckpt_paths = []
    for seed in SEEDS[:n_seeds]:
        ckpt_path = os.path.join(ckpt_dir, f'bc_seed{seed}.pt')
        train_cfg_s = dict(train_cfg, seed=seed)
        train_bc(hdf5_path, ckpt_path, train_cfg_s, device='cpu')
        ckpt_paths.append(ckpt_path)

    # ── Phase 2: open env, run rollouts, close env ───────────────────────────
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    bm = get_benchmark(cfg['env']['benchmark'])(task_order_index=0)
    task_bddl = bm.get_task_bddl_file_path(cfg['env']['task_idx'])
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl,
        camera_heights=cfg['env']['camera_heights'],
        camera_widths=cfg['env']['camera_widths'],
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )

    rates = []
    for seed, ckpt_path in zip(SEEDS[:n_seeds], ckpt_paths):
        model, obs_mean, obs_std, n_history = load_policy(ckpt_path, device='cpu')
        successes = []
        for _ in range(N_ROLLOUTS):
            s = run_rollout(env, model, obs_mean, obs_std,
                            horizon=cfg['eval']['horizon'], device='cpu', n_history=n_history)
            successes.append(s)
        rate = float(np.mean(successes))
        rates.append(rate)
        print(f"    seed={seed}  success={rate:.1%} ({sum(successes)}/{N_ROLLOUTS})")

    env.close()

    # clean up temp files
    import shutil
    shutil.rmtree(ckpt_dir, ignore_errors=True)

    mean_r = float(np.mean(rates))
    std_r  = float(np.std(rates))
    print(f"  [{label}] {mean_r:.1%} ± {std_r:.1%}")
    return mean_r, std_r


# ── scoring ───────────────────────────────────────────────────────────────────

def score_demos(contaminated_demos, clean_demos):
    """
    Score all contaminated demos with all 7 metrics.
    Returns dict: metric_name → np.array of scores (len = n_cont_demos).
    """
    n = len(contaminated_demos)
    scores = {}

    # --- standalone metrics (no fitting needed) ---
    for name, fn in STANDALONE_METRICS.items():
        print(f"  Scoring with {name}...")
        vals = []
        for obs_seq, actions, _ in contaminated_demos:
            vals.append(fn(obs_seq, actions))
        scores[name] = np.array(vals)

    # --- fittable metrics (fit on clean, score contaminated) ---
    for name, cls in FITTABLE_METRIC_CLASSES.items():
        print(f"  Fitting and scoring with {name}...")
        scorer = cls()
        scorer.fit([(o, a) for o, a, _ in clean_demos])
        vals = []
        for obs_seq, actions, _ in contaminated_demos:
            vals.append(scorer.score(obs_seq, actions))
        scores[name] = np.array(vals)

    return scores


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/libero_spatial.yaml')
    parser.add_argument('--clean-demos', default='data/raw/clean_demos.hdf5')
    parser.add_argument('--cont-demos',  default='data/raw/contaminated_demos.hdf5')
    parser.add_argument('--out', default='results/curation_baseline_results.json')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("Loading demos...")
    clean_demos = load_demos(args.clean_demos)
    cont_demos  = load_demos(args.cont_demos)
    n_cont = len(cont_demos)
    true_labels = np.array([int(s) for _, _, s in cont_demos])
    n_success = int(true_labels.sum())
    print(f"  Clean: {len(clean_demos)} demos")
    print(f"  Contaminated: {n_cont} demos, {n_success} successful")

    top_k = max(1, int(n_cont * TOP_K_FRAC))
    print(f"  Keeping top {top_k} demos per metric")

    # ── Score all demos ──
    print("\n=== Scoring demos ===")
    scores = score_demos(cont_demos, clean_demos)

    # ── Compute AUROC for each metric ──
    aurocs = {}
    for name, sc in scores.items():
        try:
            auroc = roc_auc_score(true_labels, sc)
        except Exception:
            auroc = float('nan')
        aurocs[name] = auroc
        print(f"  {name}: AUROC={auroc:.3f}")

    # ── Train and eval for each metric ──
    results = {}
    print("\n=== Training and evaluating curated subsets ===")

    for name, sc in scores.items():
        print(f"\n--- Metric: {name} ---")
        top_idx = np.argsort(sc)[::-1][:top_k]
        subset = [cont_demos[i] for i in top_idx]
        mean_r, std_r = train_and_eval(subset, cfg, label=name)
        results[name] = {
            'auroc': aurocs[name],
            'success_mean': mean_r,
            'success_std': std_r,
            'n_kept': top_k,
        }

    # ── Oracle: train on only the successful contaminated demos ──
    print("\n--- Oracle: successful demos only ---")
    oracle_demos = [(o, a, s) for o, a, s in cont_demos if s]
    print(f"  Oracle set size: {len(oracle_demos)}")
    oracle_mean, oracle_std = train_and_eval(oracle_demos, cfg, label='oracle')
    results['oracle'] = {
        'auroc': 1.0,
        'success_mean': oracle_mean,
        'success_std': oracle_std,
        'n_kept': len(oracle_demos),
    }

    # ── Contaminated baseline (all 20 demos) ──
    print("\n--- Contaminated baseline (all demos, 300 epochs) ---")
    cont_mean, cont_std = train_and_eval(cont_demos, cfg, label='contaminated_baseline')
    results['contaminated_baseline'] = {
        'auroc': float('nan'),
        'success_mean': cont_mean,
        'success_std': cont_std,
        'n_kept': n_cont,
    }

    # ── Print summary table ──
    print("\n" + "=" * 75)
    print(f"{'Metric':<22} {'AUROC':>7} {'Success%':>10} {'vs Baseline':>12} {'Status':>8}")
    print("-" * 75)
    baseline = cont_mean
    for name, r in results.items():
        if name in ('oracle', 'contaminated_baseline'):
            continue
        delta = r['success_mean'] - baseline
        flag = 'HELPS' if delta > 0 else ('HURTS' if delta < 0 else 'NEUTRAL')
        auroc_str = f"{r['auroc']:.3f}" if not np.isnan(r['auroc']) else '  N/A'
        print(f"{name:<22} {auroc_str:>7} {r['success_mean']*100:>9.1f}%  {delta*100:>+10.1f}pp  {flag:>8}")

    print("-" * 75)
    r = results['contaminated_baseline']
    print(f"{'contaminated_base':<22} {'  N/A':>7} {r['success_mean']*100:>9.1f}% (baseline)")
    r = results['oracle']
    print(f"{'oracle (ceiling)':<22} {'  N/A':>7} {r['success_mean']*100:>9.1f}% (oracle, {r['n_kept']} demos)")
    print("=" * 75)

    # ── Save results ──
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_data = {
        'contaminated_baseline_success': cont_mean,
        'oracle_success': oracle_mean,
        'metrics': results,
        'config': {
            'top_k_frac': TOP_K_FRAC,
            'n_rollouts': N_ROLLOUTS,
            'seeds': SEEDS,
            'n_epochs': 300,
        }
    }
    with open(args.out, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == '__main__':
    main()
