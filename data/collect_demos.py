"""
Collect demonstrations for libero_spatial task 0 using a Markovian pick-and-place policy.
Each action depends only on the current obs + init_bowl_pos (a per-episode constant),
so the demonstrations are fully Markovian and suitable for behavior cloning.

Saves clean and contaminated (early gripper release) demos to HDF5.
"""

import sys, os
sys.path.insert(0, '/home/user/LIBERO')
os.environ.setdefault('MUJOCO_GL', 'osmesa')

import argparse
import h5py
import numpy as np
import yaml

from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv


# OBS_KEYS read from the environment at each step.
# 'init_bowl_pos' is NOT an env key — it is stored separately as a per-episode constant.
# eef_quat excluded: constant orientation → near-zero std → normalization explosions in BC.
ENV_OBS_KEYS = [
    'robot0_eef_pos',
    'robot0_gripper_qpos',
    'akita_black_bowl_1_pos',
    'akita_black_bowl_1_to_robot0_eef_pos',
    'plate_1_pos',
    'plate_1_to_robot0_eef_pos',
]

# All keys stored in HDF5 (includes the synthetic init_bowl_pos constant).
ALL_OBS_KEYS = ENV_OBS_KEYS + ['init_bowl_pos']

SAFE_Z    = 1.20   # safe transit height (world frame)
GRASP_DZ  = 0.022  # descend to bowl_z + GRASP_DZ; bowl contact stops EEF ~2cm above center
PLACE_DZ  = 0.06   # place bowl at plate_z + PLACE_DZ


def markovian_policy(obs, init_bowl_pos, init_plate_pos,
                     inject_defect=False, release_frac=0.3):
    """
    Fully Markovian pick-and-place policy.

    Phase is inferred from current obs + fixed per-episode constants.
    No internal phase counter needed, so demonstrations are i.i.d. in obs→action.

    action: [dx, dy, dz, dax, day, daz, gripper]  (OSC_POSE)
      gripper: +1 = open, -1 = close
    """
    eef         = obs['robot0_eef_pos']
    bowl_live   = obs['akita_black_bowl_1_pos']
    bowl_to_eef = obs['akita_black_bowl_1_to_robot0_eef_pos']  # eef_pos - bowl_pos
    gripper_q   = obs['robot0_gripper_qpos']

    gain    = 3.0
    rot     = np.zeros(3)
    GO      = +1.0   # gripper open
    GC      = -1.0   # gripper close
    XY_TOL  = 0.015
    Z_TOL   = 0.010

    # Gripper is "closed" when fingers have moved from the open rest position.
    # robot0_gripper_qpos for Panda: open ≈ 0.0005, closed ≈ 0.031 (qpos grows when closing)
    GRIP_CLOSE_Q = 0.020   # above this → gripper is closing / closed

    # Bowl is considered held when gripper is closed AND bowl is near EEF.
    grip_closing = gripper_q[0] > GRIP_CLOSE_Q
    bowl_dist    = np.linalg.norm(bowl_to_eef)
    HOLD_DIST    = 0.06    # when bowl is grasped, bowl-to-eef is this small
    bowl_held    = grip_closing and bowl_dist < HOLD_DIST

    def act_to(target, gp):
        delta = np.asarray(target, dtype=float) - eef
        return np.concatenate([np.clip(gain * delta, -1, 1), rot, [gp]])

    def xy_err(p, q):
        return np.linalg.norm(np.asarray(p[:2]) - np.asarray(q[:2]))

    grasp_z = init_bowl_pos[2] + GRASP_DZ
    place_z = init_plate_pos[2] + PLACE_DZ

    if not bowl_held:
        # ── Phase 1: approach and grasp ──────────────────────────────────────
        if xy_err(eef, init_bowl_pos) > XY_TOL:
            # Not yet over bowl: move toward [bowl_xy, SAFE_Z].
            # This subsumes RISE + PREGRASP — no oscillation between them.
            return act_to([init_bowl_pos[0], init_bowl_pos[1], SAFE_Z], GO)

        elif eef[2] > grasp_z + Z_TOL:
            # Directly above bowl: DESCEND straight down
            return act_to([init_bowl_pos[0], init_bowl_pos[1], grasp_z], GO)

        else:
            # At grasp height: close gripper
            return act_to([init_bowl_pos[0], init_bowl_pos[1], grasp_z], GC)

    else:
        # ── Phase 2: lift and place ───────────────────────────────────────────
        # Defect: release_z is a fixed height computed from init_bowl_pos.
        if inject_defect:
            release_z = init_bowl_pos[2] + release_frac * (SAFE_Z - init_bowl_pos[2])
            lift_grip = GO if eef[2] > release_z else GC
        else:
            lift_grip = GC

        if xy_err(eef, init_plate_pos) > XY_TOL:
            # Not yet over plate: move toward [plate_xy, SAFE_Z].
            # This subsumes LIFT + TRANSPORT — no oscillation between them.
            return act_to([init_plate_pos[0], init_plate_pos[1], SAFE_Z], lift_grip)

        elif eef[2] > place_z + Z_TOL:
            # Directly above plate: LOWER straight down
            return act_to([init_plate_pos[0], init_plate_pos[1], place_z], GC)

        else:
            # At place height: open gripper to release bowl
            return act_to([init_plate_pos[0], init_plate_pos[1], place_z], GO)


def collect_episode(env, inject_defect=False, release_frac=0.3, horizon=500, seed=None):
    """Run one episode and return (obs_dict, actions, rewards, success)."""
    if seed is not None:
        np.random.seed(seed)
    obs = env.reset()

    # 30-step settle: gripper open, arm still. Resolves physics init artifacts.
    settle_action = np.zeros(7)
    settle_action[-1] = 1.0
    for _ in range(30):
        obs, _, _, _ = env.step(settle_action)

    init_bowl_pos  = obs['akita_black_bowl_1_pos'].copy()
    init_plate_pos = obs['plate_1_pos'].copy()

    obs_list = {k: [] for k in ALL_OBS_KEYS}
    actions = []
    rewards = []
    success = False

    for t in range(horizon):
        action = markovian_policy(
            obs, init_bowl_pos, init_plate_pos,
            inject_defect=inject_defect,
            release_frac=release_frac,
        )

        for k in ENV_OBS_KEYS:
            obs_list[k].append(obs[k].copy())
        obs_list['init_bowl_pos'].append(init_bowl_pos.copy())  # constant per episode
        actions.append(action.copy())

        obs, reward, done, _ = env.step(action)
        rewards.append(reward)

        if done:
            success = True
            break

    obs_arrays = {k: np.array(v) for k, v in obs_list.items()}
    return obs_arrays, np.array(actions), np.array(rewards), success


def save_demos(filepath, episodes):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with h5py.File(filepath, 'w') as f:
        for i, (obs_dict, actions, rewards, success) in enumerate(episodes):
            grp = f.create_group(f'demo_{i}')
            obs_grp = grp.create_group('obs')
            for k, v in obs_dict.items():
                obs_grp.create_dataset(k, data=v)
            grp.create_dataset('actions', data=actions)
            grp.create_dataset('rewards', data=rewards)
            grp.attrs['success'] = bool(success)
    print(f"Saved {len(episodes)} demos to {filepath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/libero_spatial.yaml')
    parser.add_argument('--n-clean', type=int, default=None)
    parser.add_argument('--n-contaminated', type=int, default=None)
    parser.add_argument('--out-dir', default='data/raw')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    n_clean      = args.n_clean or cfg['demo']['n_clean']
    n_cont       = args.n_contaminated or cfg['demo']['n_contaminated']
    release_frac = cfg['demo']['release_fraction']
    horizon      = cfg['env']['horizon']
    task_idx     = cfg['env']['task_idx']
    benchmark_name = cfg['env']['benchmark']

    print(f"Setting up LIBERO {benchmark_name} task {task_idx}...")
    bm = get_benchmark(benchmark_name)(task_order_index=0)
    task_bddl = bm.get_task_bddl_file_path(task_idx)

    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl,
        camera_heights=cfg['env']['camera_heights'],
        camera_widths=cfg['env']['camera_widths'],
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )

    print(f"\nCollecting {n_clean} clean demos...")
    clean_eps = []
    for i in range(n_clean):
        obs_d, acts, rews, suc = collect_episode(
            env, inject_defect=False, horizon=horizon, seed=i)
        clean_eps.append((obs_d, acts, rews, suc))
        print(f"  [{i+1}/{n_clean}] steps={len(acts)}, success={suc}")

    print(f"\nCollecting {n_cont} contaminated demos "
          f"(early release at {release_frac:.0%} of lift)...")
    cont_eps = []
    for i in range(n_cont):
        obs_d, acts, rews, suc = collect_episode(
            env, inject_defect=True, release_frac=release_frac,
            horizon=horizon, seed=1000 + i)
        cont_eps.append((obs_d, acts, rews, suc))
        print(f"  [{i+1}/{n_cont}] steps={len(acts)}, success={suc}")

    env.close()

    clean_success = sum(e[3] for e in clean_eps) / len(clean_eps)
    cont_success  = sum(e[3] for e in cont_eps) / len(cont_eps)
    print(f"\nScripted policy success rates:")
    print(f"  Clean:        {clean_success:.1%} ({sum(e[3] for e in clean_eps)}/{n_clean})")
    print(f"  Contaminated: {cont_success:.1%} ({sum(e[3] for e in cont_eps)}/{n_cont})")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.out_dir)
    save_demos(os.path.join(out_dir, 'clean_demos.hdf5'), clean_eps)
    save_demos(os.path.join(out_dir, 'contaminated_demos.hdf5'), cont_eps)


if __name__ == '__main__':
    main()
