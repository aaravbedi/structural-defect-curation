"""
Evaluate a trained BC policy on the LIBERO environment.
Reports task success rate over N rollouts.
"""

import sys, os
sys.path.insert(0, '/home/user/LIBERO')
sys.path.insert(0, '/home/user/structural-defect-curation')
os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')

import argparse
import numpy as np
import torch
import yaml

# LIBERO / MuJoCo imports are deferred to function bodies to prevent their
# OpenGL context from conflicting with PyTorch's allocator on headless systems.
from methods.bc_policy import load_policy, obs_to_vec, OBS_KEYS, phase_to_onehot


def run_rollout(env, model, obs_mean, obs_std, horizon=500, device='cpu', n_history=1):
    from collections import deque
    from data.collect_demos import scripted_policy, PHASE_RISE

    obs_buf = deque(maxlen=n_history)

    obs = env.reset()
    settle = np.zeros(7); settle[-1] = 1.0
    for _ in range(30):
        obs, _, _, _ = env.step(settle)

    init_bowl_pos  = obs['akita_black_bowl_1_pos'].copy()
    init_plate_pos = obs['plate_1_pos'].copy()

    # BC runs the full trajectory from RISE. Phase tracking via scripted_policy
    # transitions (we use BC's action but track phase separately for conditioning).
    phase, phase_step = PHASE_RISE, 0
    success = False
    for _ in range(horizon):
        obs_base = obs_to_vec({k: obs[k] for k in OBS_KEYS})
        obs_vec = np.concatenate([obs_base, phase_to_onehot(phase)])
        if len(obs_buf) == 0:
            for _ in range(n_history):
                obs_buf.append(obs_vec)
        else:
            obs_buf.append(obs_vec)
        hist_obs = np.concatenate(list(obs_buf))
        obs_norm = (hist_obs - obs_mean) / obs_std
        action = model.predict(obs_norm, device=device)
        obs, reward, done, _ = env.step(action)
        if done:
            success = True
            break
        # Advance phase tracker using scripted transitions (ignore returned action)
        _, phase, phase_step = scripted_policy(obs, phase, phase_step, init_bowl_pos, init_plate_pos)
    return success


def evaluate(policy_path, cfg, device='cpu', label=''):
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    model, obs_mean, obs_std, n_history = load_policy(policy_path, device=device)

    benchmark_name = cfg['env']['benchmark']
    task_idx = cfg['env']['task_idx']
    n_rollouts = cfg['eval']['n_rollouts']
    horizon    = cfg['eval']['horizon']

    bm = get_benchmark(benchmark_name)(task_order_index=0)
    task_bddl = bm.get_task_bddl_file_path(task_idx)
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl,
        camera_heights=cfg['env']['camera_heights'],
        camera_widths=cfg['env']['camera_widths'],
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )

    successes = []
    for i in range(n_rollouts):
        s = run_rollout(env, model, obs_mean, obs_std, horizon=horizon, device=device, n_history=n_history)
        successes.append(s)
        print(f"  [{i+1}/{n_rollouts}] success={s}")

    env.close()
    rate = np.mean(successes)
    print(f"\n{label} Success rate: {rate:.1%} ({sum(successes)}/{n_rollouts})")
    return rate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/libero_spatial.yaml')
    parser.add_argument('--policy', required=True)
    parser.add_argument('--label', default='')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluate(args.policy, cfg, device=device, label=args.label)
