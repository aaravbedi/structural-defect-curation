"""One-off: find the real LIBERO task-success signal for this robosuite/LIBERO setup."""
import sys, os
os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')
sys.path.insert(0, os.environ.get('LIBERO_PATH', '/home/user/LIBERO'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, yaml
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
from data.collect_demos import scripted_policy, PHASE_RISE, PHASE_DONE, OBS_KEYS

cfg = yaml.safe_load(open('configs/libero_spatial.yaml'))
bm = get_benchmark(cfg['env']['benchmark'])(task_order_index=0)
env = OffScreenRenderEnv(
    bddl_file_name=bm.get_task_bddl_file_path(cfg['env']['task_idx']),
    camera_heights=128, camera_widths=128,
    has_offscreen_renderer=False, use_camera_obs=False)

def check_success(e):
    """Try every plausible robosuite/LIBERO success accessor."""
    for obj in (e, getattr(e, 'env', None)):
        if obj is None:
            continue
        for attr in ('_check_success', 'check_success'):
            fn = getattr(obj, attr, None)
            if callable(fn):
                try:
                    return bool(fn()), f"{type(obj).__name__}.{attr}()"
                except Exception as ex:
                    return f"ERR:{ex}", f"{type(obj).__name__}.{attr}()"
    return None, "no check_success found"

np.random.seed(0)
obs = env.reset()
settle = np.zeros(7); settle[-1] = 1.0
for _ in range(30):
    obs, _, _, _ = env.step(settle)
init_bowl = obs['akita_black_bowl_1_pos'].copy()
init_plate = obs['plate_1_pos'].copy()

phase, phase_step = PHASE_RISE, 0
done_step = None
success_step = None
last = {}
for t in range(500):
    action, phase, phase_step = scripted_policy(obs, phase, phase_step, init_bowl, init_plate)
    obs, reward, done, info = env.step(action)
    cs, cs_src = check_success(env)
    if cs is True and success_step is None:
        success_step = t
    if done and done_step is None:
        done_step = t
    last = dict(t=t, reward=reward, done=done, info=info, cs=cs, cs_src=cs_src, phase=phase)
    if done:
        break

print("=== CLEAN episode (scripted, should genuinely succeed) ===")
print(f"final step t          : {last['t']}")
print(f"final reward          : {last['reward']!r}")
print(f"final done            : {last['done']!r}")
print(f"final info            : {last['info']!r}")
print(f"check_success accessor: {last['cs_src']}")
print(f"check_success final   : {last['cs']!r}")
print(f"first step done=True  : {done_step}")
print(f"first step success=True: {success_step}")
print(f"final reward >= 1.0   : {bool(np.asarray(last['reward']) >= 1.0)}")
env.close()
