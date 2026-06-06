"""
Collect demonstrations for libero_spatial task 0 using a scripted pick-and-place policy.
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


OBS_KEYS = [
    'robot0_eef_pos',
    'robot0_eef_quat',
    'robot0_gripper_qpos',
    'akita_black_bowl_1_pos',
    'akita_black_bowl_1_to_robot0_eef_pos',
    'plate_1_pos',
    'plate_1_to_robot0_eef_pos',
]

# Phases of the pick-and-place scripted policy.
# Uses an "up → over → down" trajectory to avoid knocking objects.
PHASE_RISE       = 0  # first move straight up to safe clear height
PHASE_PREGRASP   = 1  # move horizontally over bowl (staying high)
PHASE_DESCEND    = 2  # lower straight down to grasp height
PHASE_GRASP      = 3  # hold still and close gripper
PHASE_LIFT       = 4  # lift straight back up
PHASE_TRANSPORT  = 5  # move horizontally over plate
PHASE_LOWER      = 6  # lower to plate
PHASE_RELEASE    = 7  # open gripper
PHASE_DONE       = 8

SAFE_Z       = 1.20   # safe transit height above all objects (world frame)
GRASP_DZ     = 0.01   # descend to bowl_z + this before grasping
PLACE_DZ     = 0.06   # place bowl at plate_z + this


def scripted_policy(obs, phase, phase_step, init_bowl_pos, init_plate_pos,
                    inject_defect=False, release_frac=0.3):
    """
    Returns (action, next_phase, next_phase_step).
    action: [dx, dy, dz, dax, day, daz, gripper]  (OSC_POSE)
      gripper: +1 = open, -1 = close

    Uses init_bowl_pos / init_plate_pos to avoid chasing a displaced object.
    """
    eef  = obs['robot0_eef_pos']
    bowl = init_bowl_pos
    plate = init_plate_pos

    gain = 3.0            # P-gain; 1.0 action ≈ 10mm/step → gain=3 saturates at ~33mm err
    rot  = np.zeros(3)
    GO   = +1.0           # gripper open
    GC   = -1.0           # gripper closed
    XY_TOL  = 0.015       # 1.5 cm horizontal tolerance
    Z_TOL   = 0.010       # 1.0 cm vertical tolerance

    def action_to(target, gp, xy_only=False):
        delta = target - eef
        if xy_only:
            delta[2] = 0.0
        return np.concatenate([np.clip(gain * delta, -1, 1), rot, [gp]])

    def xy_err(target):
        return np.linalg.norm(eef[:2] - target[:2])

    def z_err(target_z):
        return abs(eef[2] - target_z)

    if phase == PHASE_RISE:
        # Move straight up to safe transit height
        target = np.array([eef[0], eef[1], SAFE_Z])
        if z_err(SAFE_Z) < Z_TOL:
            return action_to(target, GO), PHASE_PREGRASP, 0
        return action_to(target, GO), PHASE_RISE, phase_step + 1

    elif phase == PHASE_PREGRASP:
        # Move horizontally to above the bowl
        target = np.array([bowl[0], bowl[1], SAFE_Z])
        if xy_err(target) < XY_TOL:
            return action_to(target, GO), PHASE_DESCEND, 0
        return action_to(target, GO), PHASE_PREGRASP, phase_step + 1

    elif phase == PHASE_DESCEND:
        # Lower straight down to grasp height
        grasp_z = bowl[2] + GRASP_DZ
        target = np.array([bowl[0], bowl[1], grasp_z])
        if z_err(grasp_z) < Z_TOL or phase_step >= 50:
            return action_to(target, GO), PHASE_GRASP, 0
        return action_to(target, GO), PHASE_DESCEND, phase_step + 1

    elif phase == PHASE_GRASP:
        # Hold position, close gripper
        grasp_z = bowl[2] + GRASP_DZ
        target = np.array([bowl[0], bowl[1], grasp_z])
        if phase_step >= 20:
            return action_to(target, GC), PHASE_LIFT, 0
        return action_to(target, GC), PHASE_GRASP, phase_step + 1

    elif phase == PHASE_LIFT:
        # Lift straight up to safe height
        target = np.array([bowl[0], bowl[1], SAFE_Z])
        if inject_defect:
            lift_steps = 25
            release_step = int(release_frac * lift_steps)
            gp = GO if phase_step >= release_step else GC
        else:
            gp = GC
        if z_err(SAFE_Z) < Z_TOL or phase_step >= 40:
            return action_to(target, GC), PHASE_TRANSPORT, 0
        return action_to(target, gp), PHASE_LIFT, phase_step + 1

    elif phase == PHASE_TRANSPORT:
        # Move horizontally over plate at safe height
        target = np.array([plate[0], plate[1], SAFE_Z])
        if xy_err(target) < XY_TOL:
            return action_to(target, GC), PHASE_LOWER, 0
        return action_to(target, GC), PHASE_TRANSPORT, phase_step + 1

    elif phase == PHASE_LOWER:
        # Lower onto plate
        target = np.array([plate[0], plate[1], plate[2] + PLACE_DZ])
        if z_err(plate[2] + PLACE_DZ) < Z_TOL or phase_step >= 50:
            return action_to(target, GC), PHASE_RELEASE, 0
        return action_to(target, GC), PHASE_LOWER, phase_step + 1

    elif phase == PHASE_RELEASE:
        target = np.array([plate[0], plate[1], plate[2] + PLACE_DZ])
        if phase_step >= 15:
            return action_to(target, GO), PHASE_DONE, 0
        return action_to(target, GO), PHASE_RELEASE, phase_step + 1

    else:  # DONE
        action = np.zeros(7)
        action[-1] = GO
        return action, PHASE_DONE, phase_step + 1


def collect_episode(env, inject_defect=False, release_frac=0.3, horizon=500, seed=None):
    """Run one episode and return (obs_dict, actions, rewards, success)."""
    if seed is not None:
        np.random.seed(seed)
    obs = env.reset()

    # Let physics settle for 30 steps before snapshotting object positions.
    # Without this, the bowl/objects may drift from arm-body intersection artifacts.
    settle_action = np.zeros(7); settle_action[-1] = 1.0  # open gripper, stay still
    for _ in range(30):
        obs, _, _, _ = env.step(settle_action)

    # Snapshot positions after settling
    init_bowl_pos  = obs['akita_black_bowl_1_pos'].copy()
    init_plate_pos = obs['plate_1_pos'].copy()

    obs_list = {k: [] for k in OBS_KEYS}
    actions = []
    rewards = []

    phase = PHASE_RISE
    phase_step = 0
    success = False

    for t in range(horizon):
        action, next_phase, next_step = scripted_policy(
            obs, phase, phase_step,
            init_bowl_pos=init_bowl_pos,
            init_plate_pos=init_plate_pos,
            inject_defect=inject_defect,
            release_frac=release_frac,
        )
        for k in OBS_KEYS:
            obs_list[k].append(obs[k].copy())
        actions.append(action.copy())

        obs, reward, done, _ = env.step(action)
        rewards.append(reward)

        phase = next_phase
        phase_step = next_step

        if done:
            success = True
            break

    obs_arrays = {k: np.array(v) for k, v in obs_list.items()}
    return obs_arrays, np.array(actions), np.array(rewards), success


def save_demos(filepath, episodes):
    """Save episodes to HDF5 file."""
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

    n_clean = args.n_clean or cfg['demo']['n_clean']
    n_cont  = args.n_contaminated or cfg['demo']['n_contaminated']
    release_frac = cfg['demo']['release_fraction']
    horizon = cfg['env']['horizon']
    task_idx = cfg['env']['task_idx']
    benchmark_name = cfg['env']['benchmark']

    print(f"Setting up LIBERO {benchmark_name} task {task_idx}...")
    bm = get_benchmark(benchmark_name)(task_order_index=0)
    task_bddl = bm.get_task_bddl_file_path(task_idx)

    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl,
        camera_heights=cfg['env']['camera_heights'],
        camera_widths=cfg['env']['camera_widths'],
        has_offscreen_renderer=False,  # no rendering needed for state-based BC
        use_camera_obs=False,
    )

    print(f"\nCollecting {n_clean} clean demos...")
    clean_eps = []
    for i in range(n_clean):
        obs_d, acts, rews, suc = collect_episode(env, inject_defect=False, horizon=horizon, seed=i)
        clean_eps.append((obs_d, acts, rews, suc))
        print(f"  [{i+1}/{n_clean}] steps={len(acts)}, success={suc}")

    print(f"\nCollecting {n_cont} contaminated demos (early release at {release_frac:.0%} of lift)...")
    cont_eps = []
    for i in range(n_cont):
        obs_d, acts, rews, suc = collect_episode(
            env, inject_defect=True, release_frac=release_frac, horizon=horizon, seed=1000+i
        )
        cont_eps.append((obs_d, acts, rews, suc))
        print(f"  [{i+1}/{n_cont}] steps={len(acts)}, success={suc}")

    env.close()

    clean_success = sum(e[3] for e in clean_eps) / len(clean_eps)
    cont_success  = sum(e[3] for e in cont_eps)  / len(cont_eps)
    print(f"\nScripted policy success rates:")
    print(f"  Clean:        {clean_success:.1%} ({sum(e[3] for e in clean_eps)}/{n_clean})")
    print(f"  Contaminated: {cont_success:.1%} ({sum(e[3] for e in cont_eps)}/{n_cont})")

    save_demos(os.path.join(args.out_dir, 'clean_demos.hdf5'), clean_eps)
    save_demos(os.path.join(args.out_dir, 'contaminated_demos.hdf5'), cont_eps)


if __name__ == '__main__':
    main()
