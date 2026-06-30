"""
Sanity check for the success-label fix in data/collect_demos.py.

Collects 5 clean and 5 defective (early-gripper-release) demos with the fixed
collect_episode() and prints the two success rates. With a correct success
signal we expect clean to be high (~90%+) and defective to be low (most fail,
because the dropped bowl is never placed).
"""
import sys, os
os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')
sys.path.insert(0, os.environ.get('LIBERO_PATH', '/home/user/LIBERO'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
from data.collect_demos import collect_episode

cfg = yaml.safe_load(open('configs/libero_spatial.yaml'))
release_frac = float(cfg['demo']['release_fraction'])
N = 5

bm = get_benchmark(cfg['env']['benchmark'])(task_order_index=0)
env = OffScreenRenderEnv(
    bddl_file_name=bm.get_task_bddl_file_path(cfg['env']['task_idx']),
    camera_heights=128, camera_widths=128,
    has_offscreen_renderer=False, use_camera_obs=False)

print(f"\nCollecting {N} CLEAN demos...")
clean = []
for i in range(N):
    _, acts, _, suc = collect_episode(env, inject_defect=False, horizon=500, seed=i)
    clean.append(suc)
    print(f"  clean  [{i+1}/{N}]  steps={len(acts):3d}  success={suc}")

print(f"\nCollecting {N} DEFECTIVE demos (early release at {release_frac:.0%} of lift)...")
defect = []
for i in range(N):
    _, acts, _, suc = collect_episode(env, inject_defect=True, release_frac=release_frac,
                                      horizon=500, seed=1000 + i)
    defect.append(suc)
    print(f"  defect [{i+1}/{N}]  steps={len(acts):3d}  success={suc}")

env.close()

clean_rate  = sum(clean)  / len(clean)
defect_rate = sum(defect) / len(defect)
print("\n" + "=" * 48)
print(f"CLEAN     success rate: {clean_rate:.0%} ({sum(clean)}/{N})")
print(f"DEFECTIVE success rate: {defect_rate:.0%} ({sum(defect)}/{N})")
print("=" * 48)
if clean_rate >= 0.8 and defect_rate <= 0.4:
    print("PASS: clean high, defective low — the defect now genuinely fails the task.")
else:
    print("CHECK: rates not as expected — review before running the sweep.")
