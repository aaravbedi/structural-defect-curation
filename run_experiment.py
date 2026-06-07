#!/usr/bin/env python3
"""
Full 3-step curation pipeline:
  Step 1 — Clean BC ground truth   (50 demos, gate >=30%)
  Step 2 — Contamination baselines (80 demos, gate oracle >= clean-15pp)
  Step 3 — Curation metrics        (6 metrics, AUROC + downstream BC success)
"""
import sys, os
sys.path.insert(0, '/home/user/LIBERO')
sys.path.insert(0, '/home/user/structural-defect-curation')
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import gc, tempfile, shutil
import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from data.collect_demos import collect_episode, save_demos
from methods.bc_policy import train as train_bc, load_policy
from eval.evaluate import run_rollout
from methods.curation_metrics import (
    smoothness, entropy, gripper_timing,
    IsolationForestScorer, KNNScorer, TrajectoryAlignmentScorer,
)


def build_env(cfg):
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv
    bm = get_benchmark(cfg['env']['benchmark'])(task_order_index=0)
    bddl = bm.get_task_bddl_file_path(cfg['env']['task_idx'])
    return OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=cfg['env']['camera_heights'],
        camera_widths=cfg['env']['camera_widths'],
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )


def eval_policy(env, ckpt, n, device='cpu'):
    model, obs_mean, obs_std, n_history = load_policy(ckpt, device=device)
    results = []
    for i in range(n):
        s = run_rollout(env, model, obs_mean, obs_std, horizon=500,
                        device=device, n_history=n_history)
        results.append(s)
        print(f"    [{i+1}/{n}] success={s}", flush=True)
    return results


def main():
    with open('configs/libero_spatial.yaml') as f:
        cfg = yaml.safe_load(f)
    train_cfg = dict(cfg['train'])
    release_frac = cfg['demo']['release_fraction']
    D = tempfile.mkdtemp(prefix='pipeline_')
    print(f"Working dir: {D}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Clean BC ground truth
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "="*62)
    print("STEP 1: Clean BC ground truth  (50 demos, 500 epochs, 50 rollouts)")
    print("="*62)

    print("\n[1a] Collecting 50 clean demos (seeds 0-49)...")
    env = build_env(cfg)
    clean_eps = []
    for i in range(50):
        ep = collect_episode(env, inject_defect=False, horizon=500, seed=i)
        clean_eps.append(ep)
        print(f"  [{i+1}/50] steps={len(ep[1])}, success={ep[3]}")
    env.close(); del env; gc.collect()

    n_clean_suc = sum(e[3] for e in clean_eps)
    print(f"\nScripted clean success: {n_clean_suc}/50")

    clean_hdf5 = f'{D}/clean_demos.hdf5'
    save_demos(clean_hdf5, clean_eps)

    print("\n[1b] Training clean BC (500 epochs)...")
    clean_ckpt = f'{D}/clean_bc.pt'
    train_bc(clean_hdf5, clean_ckpt, dict(train_cfg, n_epochs=500, seed=42), device='cpu')

    print("\n[1c] Evaluating: 50 rollouts...")
    env = build_env(cfg)
    s1_res = eval_policy(env, clean_ckpt, n=50, device='cpu')
    env.close(); del env; gc.collect()

    s1_rate = np.mean(s1_res)
    print(f"\n{'='*62}")
    print(f"STEP 1 RESULT: {s1_rate:.0%}  ({sum(s1_res)}/50)")
    print(f"{'='*62}")

    if s1_rate < 0.30:
        print("GATE FAILED: <30%. Stopping pipeline.")
        shutil.rmtree(D)
        return

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Contamination baselines
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "="*62)
    print("STEP 2: Contamination baselines  (80 demos: 20 clean + 60 defective)")
    print("="*62)

    print("\n[2a] Collecting 80 contaminated demos...")
    env = build_env(cfg)
    cont_eps = []

    for i in range(20):
        ep = collect_episode(env, inject_defect=False, horizon=500, seed=2000+i)
        cont_eps.append(ep)
        print(f"  [{i+1}/80] steps={len(ep[1])}, success={ep[3]}  [clean]")

    for i in range(60):
        ep = collect_episode(env, inject_defect=True, release_frac=release_frac,
                             horizon=500, seed=1000+i)
        cont_eps.append(ep)
        print(f"  [{20+i+1}/80] steps={len(ep[1])}, success={ep[3]}  [defect]")

    env.close(); del env; gc.collect()

    cont_hdf5 = f'{D}/cont_demos.hdf5'
    save_demos(cont_hdf5, cont_eps)

    oracle_eps = [e for e in cont_eps if e[3]]
    if len(oracle_eps) == 0:
        print("GATE FAILED: No successful demos in contaminated set. Stopping.")
        shutil.rmtree(D)
        return
    oracle_hdf5 = f'{D}/oracle_demos.hdf5'
    save_demos(oracle_hdf5, oracle_eps)
    print(f"\nOracle set: {len(oracle_eps)} successful demos")

    print("\n[2b] Training contaminated BC (all 80 demos, 500 epochs)...")
    cont_ckpt = f'{D}/cont_bc.pt'
    train_bc(cont_hdf5, cont_ckpt, dict(train_cfg, n_epochs=500, seed=42), device='cpu')

    print(f"\n[2c] Training oracle BC ({len(oracle_eps)} demos, 500 epochs)...")
    oracle_ckpt = f'{D}/oracle_bc.pt'
    train_bc(oracle_hdf5, oracle_ckpt, dict(train_cfg, n_epochs=500, seed=42), device='cpu')

    print("\n[2d] Evaluating contaminated BC: 50 rollouts...")
    env = build_env(cfg)
    s2_cont = eval_policy(env, cont_ckpt, n=50, device='cpu')
    env.close(); del env; gc.collect()

    print("\n[2e] Evaluating oracle BC: 50 rollouts...")
    env = build_env(cfg)
    s2_oracle = eval_policy(env, oracle_ckpt, n=50, device='cpu')
    env.close(); del env; gc.collect()

    cont_rate   = np.mean(s2_cont)
    oracle_rate = np.mean(s2_oracle)
    print(f"\n{'='*62}")
    print(f"STEP 2 RESULTS:")
    print(f"  Contaminated BC ({len(cont_eps):2d} demos): {cont_rate:.0%}  ({sum(s2_cont)}/50)")
    print(f"  Oracle BC       ({len(oracle_eps):2d} demos): {oracle_rate:.0%}  ({sum(s2_oracle)}/50)")
    print(f"{'='*62}")

    if oracle_rate < s1_rate - 0.15:
        print(f"GATE FAILED: Oracle {oracle_rate:.0%} < Clean {s1_rate:.0%} - 15pp. Stopping.")
        shutil.rmtree(D)
        return

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: Curation metrics
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "="*62)
    print("STEP 3: Curation metrics  (6 metrics, top-75%=60 demos, 50 rollouts each)")
    print("="*62)

    # Labels: positions 0-19 = clean (1), 20-79 = defective (0)
    labels    = np.array([1] * 20 + [0] * 60)
    obs_list  = [e[0] for e in cont_eps]
    acts_list = [e[1] for e in cont_eps]

    # Fittable metrics fit on Step-1 clean demos (50 clean reference distribution)
    clean_ref = [(e[0], e[1]) for e in clean_eps]

    print("\n[3a] Scoring all 80 contaminated demos with 6 metrics...")
    all_scores = {}

    for name, fn in [('smoothness',      smoothness),
                     ('entropy',         entropy),
                     ('gripper_timing',  gripper_timing)]:
        sc = np.array([fn(o, a) for o, a in zip(obs_list, acts_list)])
        all_scores[name] = sc
        print(f"  {name:<22}: AUROC={roc_auc_score(labels, sc):.3f}")

    for name, Cls in [('isolation_forest',      IsolationForestScorer),
                      ('kNN',                   KNNScorer),
                      ('trajectory_alignment',  TrajectoryAlignmentScorer)]:
        scorer = Cls()
        scorer.fit(clean_ref)
        sc = np.array([scorer.score(o, a) for o, a in zip(obs_list, acts_list)])
        all_scores[name] = sc
        print(f"  {name:<22}: AUROC={roc_auc_score(labels, sc):.3f}")

    # Per-metric: keep top-75% (60 of 80), train BC, 50 rollouts
    print("\n[3b] Per-metric curation BC...")
    top_k = int(0.75 * 80)   # 60
    metric_results = {}

    for name, sc in all_scores.items():
        print(f"\n  [{name}]")
        top_idx    = np.argsort(sc)[-top_k:]
        n_clean_in = int(np.sum(top_idx < 20))
        n_def_in   = top_k - n_clean_in
        auroc      = roc_auc_score(labels, sc)
        print(f"    Selected {top_k}: {n_clean_in} clean, {n_def_in} defective  (AUROC={auroc:.3f})")

        sel_hdf5 = f'{D}/{name}_sel.hdf5'
        save_demos(sel_hdf5, [cont_eps[i] for i in top_idx])

        sel_ckpt = f'{D}/{name}_bc.pt'
        train_bc(sel_hdf5, sel_ckpt, dict(train_cfg, n_epochs=500, seed=42), device='cpu')

        print(f"    Running 50 rollouts...")
        env = build_env(cfg)
        m_res = eval_policy(env, sel_ckpt, n=50, device='cpu')
        env.close(); del env; gc.collect()

        m_rate = np.mean(m_res)
        metric_results[name] = dict(auroc=auroc, success=m_rate,
                                    vs_cont=m_rate - cont_rate,
                                    n_clean=n_clean_in)
        print(f"    => success={m_rate:.0%}  vs_cont={m_rate-cont_rate:+.0%}")

    # ══════════════════════════════════════════════════════════════════
    # FINAL TABLE
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "="*62)
    print("FINAL RESULTS")
    print("="*62)
    print(f"  Clean BC        (50 demos): {s1_rate:.0%}  ({sum(s1_res)}/50)")
    print(f"  Contaminated BC (80 demos): {cont_rate:.0%}  ({sum(s2_cont)}/50)")
    print(f"  Oracle BC       ({len(oracle_eps):2d} demos): {oracle_rate:.0%}  ({sum(s2_oracle)}/50)")
    print()
    print(f"  {'Metric':<22} {'AUROC':>6}  {'Success':>8}  {'vs Cont':>8}  {'Clean/60':>9}")
    print(f"  {'-'*22} {'-'*6}  {'-'*8}  {'-'*8}  {'-'*9}")
    for name, r in metric_results.items():
        print(f"  {name:<22} {r['auroc']:>6.3f}  {r['success']:>8.0%}  {r['vs_cont']:>+8.0%}  {r['n_clean']:>9}/60")

    shutil.rmtree(D)
    print("\nPipeline complete.")


if __name__ == '__main__':
    main()
