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
                        device='cpu', n_history=n_history, seed=i)
        results.append(s)
        print(f"  [{i+1}/{args.n}] success={s}", flush=True)

    env.close()
    n_suc = sum(results)
    print(f"RESULT:{json.dumps({'rate': n_suc / args.n, 'n_success': n_suc, 'n': args.n})}")


# ── diag ─────────────────────────────────────────────────────────────────────

def task_diag(args):
    """One verbose rollout — print phase, arm position, and actions each step."""
    from methods.bc_policy import load_policy, obs_to_vec, OBS_KEYS, phase_to_onehot
    from eval.evaluate import run_rollout
    model, obs_mean, obs_std, n_history = load_policy(args.ckpt, device='cpu')
    print(f"Model loaded: obs_dim={len(obs_mean)}, n_history={n_history}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from data.collect_demos import (scripted_policy, PHASE_RISE, PHASE_PREGRASP,
                                    PHASE_DESCEND, PHASE_GRASP, PHASE_LIFT, PHASE_TRANSPORT)
    from collections import deque

    bm = get_benchmark(cfg['env']['benchmark'])(task_order_index=0)
    bddl = bm.get_task_bddl_file_path(cfg['env']['task_idx'])
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=cfg['env']['camera_heights'],
        camera_widths=cfg['env']['camera_widths'],
        has_offscreen_renderer=False, use_camera_obs=False,
    )

    np.random.seed(0)
    obs = env.reset()
    settle = np.zeros(7); settle[-1] = 1.0
    for _ in range(30):
        obs, _, _, _ = env.step(settle)

    init_bowl_pos  = obs['akita_black_bowl_1_pos'].copy()
    init_plate_pos = obs['plate_1_pos'].copy()
    print(f"After settle: eef={obs['robot0_eef_pos'].round(3)}  bowl={init_bowl_pos.round(3)}  plate={init_plate_pos.round(3)}")

    PNAME = {0:'RISE', 1:'PREGRASP', 2:'DESCEND', 3:'GRASP',
             4:'LIFT', 5:'TRANSPORT', 6:'LOWER', 7:'RELEASE', 8:'DONE'}
    obs_buf  = deque(maxlen=n_history)
    phase, phase_step = PHASE_RISE, 0

    # Scripted warmup through TRANSPORT (mirrors run_rollout)
    for _ in range(500):
        if phase not in (PHASE_RISE, PHASE_PREGRASP, PHASE_DESCEND,
                         PHASE_GRASP, PHASE_LIFT, PHASE_TRANSPORT):
            break
        act, phase, phase_step = scripted_policy(obs, phase, phase_step, init_bowl_pos, init_plate_pos)
        obs, _, done, _ = env.step(act)
        if done:
            print("\n*** SUCCESS (scripted warmup) ***"); env.close(); return

    print(f"Warmup done: phase={PNAME[phase]}, eef=({obs['robot0_eef_pos'][0]:+.3f},{obs['robot0_eef_pos'][1]:+.3f},{obs['robot0_eef_pos'][2]:.3f})", flush=True)
    prev_phase = phase - 1  # force first transition print

    for t in range(500):
        obs_base = obs_to_vec({k: obs[k] for k in OBS_KEYS})
        obs_vec  = np.concatenate([obs_base, phase_to_onehot(phase)])
        if len(obs_buf) == 0:
            for _ in range(n_history): obs_buf.append(obs_vec)
        else:
            obs_buf.append(obs_vec)
        obs_norm = (np.concatenate(list(obs_buf)) - obs_mean) / obs_std
        action = model.predict(obs_norm, device='cpu')

        eef     = obs['robot0_eef_pos']
        bowl    = obs['akita_black_bowl_1_pos']
        plate   = obs['plate_1_pos']
        xy_dist = float(np.linalg.norm(eef[:2] - bowl[:2]))
        plate_err = float(np.linalg.norm(eef[:2] - plate[:2]))

        if phase != prev_phase:
            print(f"\n--- phase → {PNAME[phase]} (t={t}) ---", flush=True)
            prev_phase = phase
        if t < 80 or t % 20 == 0:
            print(f"t={t:3d} {PNAME[phase]:<10} "
                  f"eef=({eef[0]:+.3f},{eef[1]:+.3f},{eef[2]:.3f}) "
                  f"plate=({plate[0]:+.3f},{plate[1]:+.3f}) plate_err={plate_err:.3f} "
                  f"bowl_z={bowl[2]:.3f} g={obs['robot0_gripper_qpos'][0]:.2f} "
                  f"act=[{action[0]:+.3f},{action[1]:+.3f},{action[2]:+.3f},grip={action[-1]:+.3f}]",
                  flush=True)

        obs, reward, done, _ = env.step(action)
        if done:
            print(f"\n*** SUCCESS at t={t} ***"); break
        _, phase, phase_step = scripted_policy(
            obs, phase, phase_step, init_bowl_pos, init_plate_pos)

    env.close()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--task',   required=True, choices=['collect', 'train', 'score', 'eval', 'diag'])
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
    elif args.task == 'diag':    task_diag(args)
