"""
Phase-isolated worker for the curation pipeline.
Run as a subprocess so MuJoCo and PyTorch never share a process.

Tasks:
  collect  --n-clean N --n-defect M --seed-clean-start S1 --seed-defect-start S2 --out PATH
  train    --data PATH --ckpt PATH [--n-epochs N]
  score    --data PATH --ref PATH --labels JSON --out-scores PATH
  eval     --ckpt PATH --n N
"""
import sys, os, argparse, json
sys.path.insert(0, '/home/user/LIBERO')
sys.path.insert(0, '/home/user/structural-defect-curation')
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import numpy as np
import yaml


# ── collect ──────────────────────────────────────────────────────────────────

def task_collect(args):
    from data.collect_demos import collect_episode, save_demos
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv
    bm = get_benchmark(cfg['env']['benchmark'])(task_order_index=0)
    bddl = bm.get_task_bddl_file_path(cfg['env']['task_idx'])
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=cfg['env']['camera_heights'],
        camera_widths=cfg['env']['camera_widths'],
        has_offscreen_renderer=False, use_camera_obs=False,
    )
    release_frac = float(cfg['demo']['release_fraction'])
    n_total = args.n_clean + args.n_defect
    eps = []

    for i in range(args.n_clean):
        ep = collect_episode(env, inject_defect=False, horizon=500,
                             seed=args.seed_clean_start + i)
        eps.append(ep)
        print(f"  [{i+1}/{n_total}] steps={len(ep[1])}, success={ep[3]}  [clean]", flush=True)

    for i in range(args.n_defect):
        ep = collect_episode(env, inject_defect=True, release_frac=release_frac,
                             horizon=500, seed=args.seed_defect_start + i)
        eps.append(ep)
        print(f"  [{args.n_clean+i+1}/{n_total}] steps={len(ep[1])}, success={ep[3]}  [defect]",
              flush=True)

    env.close()
    save_demos(args.out, eps)
    n_suc = sum(e[3] for e in eps)
    print(f"RESULT:{json.dumps({'n_success': n_suc, 'n_total': n_total})}")


# ── train ────────────────────────────────────────────────────────────────────

def task_train(args):
    from methods.bc_policy import train as train_bc
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train_cfg = dict(cfg['train'])
    train_cfg['n_epochs'] = args.n_epochs
    train_bc(args.data, args.ckpt, train_cfg, device='cpu')


# ── score ────────────────────────────────────────────────────────────────────

def task_score(args):
    import h5py
    from methods.bc_policy import OBS_KEYS
    from methods.curation_metrics import (
        smoothness, entropy, gripper_timing,
        IsolationForestScorer, KNNScorer, TrajectoryAlignmentScorer,
    )
    from sklearn.metrics import roc_auc_score

    def load_hdf5(path):
        demos = []
        with h5py.File(path, 'r') as f:
            for key in sorted(f.keys()):
                d = f[key]
                obs = {k: d['obs'][k][:] for k in OBS_KEYS if k in d['obs']}
                acts = d['actions'][:]
                demos.append((obs, acts))
        return demos

    all_demos = load_hdf5(args.data)
    ref_demos = load_hdf5(args.ref)
    labels    = json.loads(args.labels)

    scores = {}
    for name, fn in [('smoothness', smoothness),
                     ('entropy', entropy),
                     ('gripper_timing', gripper_timing)]:
        scores[name] = [float(fn(o, a)) for o, a in all_demos]

    for name, Cls in [('isolation_forest', IsolationForestScorer),
                      ('kNN', KNNScorer),
                      ('trajectory_alignment', TrajectoryAlignmentScorer)]:
        scorer = Cls()
        scorer.fit(ref_demos)
        scores[name] = [float(scorer.score(o, a)) for o, a in all_demos]

    aurocs = {}
    for name, sc in scores.items():
        auroc = float(roc_auc_score(labels, sc))
        aurocs[name] = auroc
        print(f"  {name:<22}: AUROC={auroc:.3f}", flush=True)

    print(f"RESULT:{json.dumps({'scores': scores, 'aurocs': aurocs})}")


# ── eval ─────────────────────────────────────────────────────────────────────

def task_eval(args):
    # Load PyTorch model FIRST, then create MuJoCo env (safe order)
    from methods.bc_policy import load_policy
    from eval.evaluate import run_rollout
    model, obs_mean, obs_std, n_history = load_policy(args.ckpt, device='cpu')

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv
    bm = get_benchmark(cfg['env']['benchmark'])(task_order_index=0)
    bddl = bm.get_task_bddl_file_path(cfg['env']['task_idx'])
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=cfg['env']['camera_heights'],
        camera_widths=cfg['env']['camera_widths'],
        has_offscreen_renderer=False, use_camera_obs=False,
    )

    results = []
    for i in range(args.n):
        s = run_rollout(env, model, obs_mean, obs_std, horizon=500,
                        device='cpu', n_history=n_history)
        results.append(s)
        print(f"  [{i+1}/{args.n}] success={s}", flush=True)

    env.close()
    n_suc = sum(results)
    print(f"RESULT:{json.dumps({'rate': n_suc / args.n, 'n_success': n_suc, 'n': args.n})}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--task',   required=True, choices=['collect', 'train', 'score', 'eval'])
    p.add_argument('--config', default='configs/libero_spatial.yaml')
    # collect
    p.add_argument('--n-clean',           type=int, default=0)
    p.add_argument('--n-defect',          type=int, default=0)
    p.add_argument('--seed-clean-start',  type=int, default=0)
    p.add_argument('--seed-defect-start', type=int, default=1000)
    # train
    p.add_argument('--n-epochs', type=int, default=500)
    # score
    p.add_argument('--ref',    default=None)
    p.add_argument('--labels', default=None)
    # eval
    p.add_argument('--n', type=int, default=50)
    # shared
    p.add_argument('--data', default=None)
    p.add_argument('--out',  default=None)
    p.add_argument('--ckpt', default=None)
    args = p.parse_args()

    if   args.task == 'collect': task_collect(args)
    elif args.task == 'train':   task_train(args)
    elif args.task == 'score':   task_score(args)
    elif args.task == 'eval':    task_eval(args)
