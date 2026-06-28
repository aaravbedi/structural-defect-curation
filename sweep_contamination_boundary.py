#!/usr/bin/env python3
"""
Contamination-boundary sweep: at what contamination rate does curation stop
working, and why?

Maps the boundary in (contamination_rate, keep_rate) space for a fixed BC
policy and the LIBERO early-gripper-release structural defect.  Reuses the
existing repo infrastructure verbatim:

  - LIBERO env + scripted defect collection ... data/collect_demos.py
  - phase-conditioned BC policy             ... methods/bc_policy.py
  - curation metrics                        ... methods/curation_metrics.py
  - subprocess worker (MuJoCo/PyTorch split) ... _worker.py
  - headless 3-phase rollout eval           ... eval/evaluate.py

Grid (12 cells):
  contamination_rate in {0.2, 0.4, 0.6, 0.8}
  keep_rate          in {0.25, 0.5, 0.75}

Three conditions per cell:
  oracle      - curate with ground-truth contamination labels (upper bound)
  best_metric - curate with the single metric that had the best cached
                downstream success (auto-picked from results/; see pick_best_metric)
  baseline    - no curation, train on the full contaminated set (lower bound)

3 seeds per (cell, condition)  ->  12 * 3 * 3 = 108 units total.

Each (cell, condition, seed) unit trains in one subprocess and evaluates in a
second subprocess, so a MuJoCo segfault kills only that unit.  Completed units
are recorded in the resumable JSON state after EVERY unit; relaunching skips
anything already recorded, so losing the machine mid-run costs at most one unit.

Usage:
  python sweep_contamination_boundary.py            # full resumable sweep
  python sweep_contamination_boundary.py --smoke    # ONE unit, isolated state, then stop

The smoke run uses a separate state/log/csv namespace (results/sweep/smoke_*)
so it never pollutes the real 30-rollout sweep state.
"""

import sys, os, subprocess, json, time, argparse, tempfile, csv, datetime
import numpy as np
import h5py

ROOT   = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(ROOT, '_worker.py')
PYTHON = sys.executable
CONFIG = os.path.join(ROOT, 'configs', 'libero_spatial.yaml')

# Sweep artifacts live under results/sweep/ so the existing results/ stay intact.
SWEEP_DIR = os.path.join(ROOT, 'results', 'sweep')
POOL_DIR  = os.path.join(SWEEP_DIR, 'pools')
SET_DIR   = os.path.join(SWEEP_DIR, 'datasets')
CKPT_DIR  = os.path.join(SWEEP_DIR, 'ckpts')

# Prior cached benchmark (for picking best_metric).
CACHE_RESULTS = os.path.join(ROOT, 'results', 'libero_curation_results.json')

# ── Grid + protocol constants ──────────────────────────────────────────────────

CONTAMINATION_RATES = [0.2, 0.4, 0.6, 0.8]
KEEP_RATES          = [0.25, 0.5, 0.75]
CONDITIONS          = ['oracle', 'best_metric', 'baseline']
SEEDS               = [42, 0, 7]          # identical to prior stable runs

N_TOTAL    = 80          # contaminated-set size per cell (matches prior protocol)
EPOCHS     = 300         # full training, same as the most recent stable runs
N_ROLLOUTS = 30          # headless rollouts per trained policy

# Reference clean set (for fitting the fittable curation metrics).
N_REF_CLEAN = 50

# Seed schemes mirror data/collect_demos.py exactly:
REF_SEED_START    = 0       # reference clean demos
CLEAN_SEED_START  = 2000    # in-contamination clean demos
DEFECT_SEED_START = 1000    # defective demos

POOL_CLEAN  = os.path.join(POOL_DIR, 'pool_clean.hdf5')
POOL_DEFECT = os.path.join(POOL_DIR, 'pool_defect.hdf5')
REF_CLEAN   = os.path.join(POOL_DIR, 'ref_clean.hdf5')

N_UNITS_TOTAL = len(CONTAMINATION_RATES) * len(KEEP_RATES) * len(CONDITIONS) * len(SEEDS)


# ── small utilities ─────────────────────────────────────────────────────────────

def ts():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def n_defect_for(cr):
    return int(round(cr * N_TOTAL))


def n_clean_for(cr):
    return N_TOTAL - n_defect_for(cr)


def keep_n_for(keep_rate):
    return int(round(keep_rate * N_TOTAL))


def hdf5_count(path):
    if not os.path.exists(path):
        return 0
    with h5py.File(path, 'r') as f:
        return len(f.keys())


def worker_env():
    env = dict(os.environ)
    env.setdefault('MUJOCO_GL', 'osmesa')
    env.setdefault('PYOPENGL_PLATFORM', 'osmesa')
    return env


def run_worker(extra_args, label, log):
    """Run _worker.py in an isolated subprocess.

    Returns (returncode, result_dict_or_None).  Never raises on a non-zero exit
    or a segfault: callers decide what a failure means for the unit.
    """
    cmd = [PYTHON, WORKER, '--config', CONFIG] + [str(a) for a in extra_args]
    log(f"    >>> {label}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=ROOT, env=worker_env(),
    )
    result_json, tail = None, []
    for line in proc.stdout:
        line = line.rstrip('\n')
        if line.startswith('RESULT:'):
            result_json = line[7:]
        elif line.strip():
            tail.append(line)
            if len(tail) > 40:
                tail.pop(0)
    proc.wait()
    result = json.loads(result_json) if result_json else None
    if proc.returncode != 0:
        log(f"    [worker FAILED exit={proc.returncode}] {label}")
        for ln in tail[-8:]:
            log(f"      | {ln}")
    return proc.returncode, result


# ── data collection (growable pools) ────────────────────────────────────────────

def _merge_into_pool(src, dst, start_index):
    """Append every demo_* group in src into dst, renumbered from start_index."""
    mode = 'a' if os.path.exists(dst) else 'w'
    with h5py.File(src, 'r') as s, h5py.File(dst, mode) as d:
        for j, key in enumerate(sorted(s.keys(), key=lambda k: int(k.split('_')[1]))):
            s.copy(key, d, name=f'demo_{start_index + j}')


def ensure_pool(path, kind, n_needed, seed_start, log):
    """Collect demos until `path` holds at least n_needed, appending as needed."""
    have = hdf5_count(path)
    if have >= n_needed:
        log(f"  [pool] {os.path.basename(path)}: {have} demos (>= {n_needed} needed) [skip]")
        return
    n_new = n_needed - have
    log(f"  [pool] {os.path.basename(path)}: have {have}, collecting {n_new} more "
        f"({kind}, seeds {seed_start + have}..{seed_start + n_needed - 1})")
    tmp = os.path.join(POOL_DIR, f'_tmp_{kind}_{have}_{n_needed}.hdf5')
    if kind == 'clean':
        args = ['--task', 'collect', '--n-clean', n_new,
                '--seed-clean-start', seed_start + have, '--out', tmp]
    else:
        args = ['--task', 'collect', '--n-defect', n_new,
                '--seed-defect-start', seed_start + have, '--out', tmp]
    rc, _ = run_worker(args, f"collect {n_new} {kind} demos", log)
    if rc != 0 or not os.path.exists(tmp):
        raise RuntimeError(f"Pool collection failed for {kind} ({path})")
    _merge_into_pool(tmp, path, have)
    os.remove(tmp)
    log(f"  [pool] {os.path.basename(path)}: now {hdf5_count(path)} demos")


def ensure_pools_for_cells(cells, need_ref, log):
    """Grow the clean/defect pools (and optionally the ref set) for the requested cells."""
    os.makedirs(POOL_DIR, exist_ok=True)
    crs = sorted({cr for cr, _ in cells})
    max_clean  = max(n_clean_for(cr)  for cr in crs)
    max_defect = max(n_defect_for(cr) for cr in crs)
    ensure_pool(POOL_CLEAN,  'clean',  max_clean,  CLEAN_SEED_START,  log)
    ensure_pool(POOL_DEFECT, 'defect', max_defect, DEFECT_SEED_START, log)
    if need_ref:
        ensure_pool(REF_CLEAN, 'clean', N_REF_CLEAN, REF_SEED_START, log)


# ── per-cr contaminated set + curated subsets ───────────────────────────────────

def contaminated_set_path(cr):
    return os.path.join(SET_DIR, f'contaminated_cr{cr:.2f}.hdf5')


def build_contaminated_set(cr, log):
    """Assemble the N_TOTAL-demo contaminated set for a contamination rate.

    Layout is [n_clean clean demos, n_defect defective demos] so the
    ground-truth labels are [1]*n_clean + [0]*n_defect (1 = clean/good).
    """
    os.makedirs(SET_DIR, exist_ok=True)
    dst = contaminated_set_path(cr)
    if os.path.exists(dst) and hdf5_count(dst) == N_TOTAL:
        return dst
    n_clean, n_defect = n_clean_for(cr), n_defect_for(cr)
    with h5py.File(POOL_CLEAN, 'r') as fc, \
         h5py.File(POOL_DEFECT, 'r') as fd, \
         h5py.File(dst, 'w') as out:
        for i in range(n_clean):
            fc.copy(f'demo_{i}', out, name=f'demo_{i}')
        for j in range(n_defect):
            fd.copy(f'demo_{j}', out, name=f'demo_{n_clean + j}')
    log(f"  [data] cr={cr}: contaminated set = {n_clean} clean + {n_defect} defect")
    return dst


def labels_for(cr):
    return [1] * n_clean_for(cr) + [0] * n_defect_for(cr)


def scores_path(cr):
    return os.path.join(SET_DIR, f'scores_cr{cr:.2f}.json')


def compute_scores(cr, log):
    """Score the contaminated set with the repo's metrics (cached per cr)."""
    path = scores_path(cr)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    cont = build_contaminated_set(cr, log)
    rc, r = run_worker(
        ['--task', 'score', '--data', cont, '--ref', REF_CLEAN,
         '--labels', json.dumps(labels_for(cr))],
        f"score contaminated set (cr={cr})", log)
    if rc != 0 or r is None:
        raise RuntimeError(f"Scoring failed for cr={cr}")
    scores = r['scores']
    # ensemble is derived exactly as in run_curation_eval.py
    scores['ensemble'] = [0.5 * s + 0.5 * g
                          for s, g in zip(scores['smoothness'], scores['gripper_timing'])]
    with open(path, 'w') as f:
        json.dump({'scores': scores, 'aurocs': r['aurocs']}, f)
    return {'scores': scores, 'aurocs': r['aurocs']}


def curated_subset(cr, keep_rate, condition, best_metric, log):
    """Return the hdf5 path the policy should train on for this (cr, keep, condition).

    baseline    -> the full contaminated set (keep_rate is ignored).
    oracle      -> keep top-k by ground-truth label (clean first), ties by index.
    best_metric -> keep top-k by the chosen metric's score.
    Subsets are cached per (cr, keep, condition) and reused across seeds.
    """
    cont = build_contaminated_set(cr, log)
    if condition == 'baseline':
        return cont

    keep_n = keep_n_for(keep_rate)
    dst = os.path.join(SET_DIR,
                       f'sel_cr{cr:.2f}_keep{keep_rate:.2f}_{condition}.hdf5')
    if os.path.exists(dst) and hdf5_count(dst) == keep_n:
        return dst

    if condition == 'oracle':
        labels = labels_for(cr)                      # 1 = clean/good, 0 = defect
        # Highest label first, ties broken by original index (stable, deterministic).
        order = sorted(range(N_TOTAL), key=lambda i: (-labels[i], i))
        keep_idx = sorted(order[:keep_n])
        n_clean_kept = sum(labels[i] for i in keep_idx)
        log(f"  [select] cr={cr} keep={keep_rate} oracle: "
            f"{n_clean_kept} clean / {keep_n - n_clean_kept} defect kept")
    elif condition == 'best_metric':
        sc = np.asarray(compute_scores(cr, log)['scores'][best_metric])
        keep_idx = sorted(np.argsort(sc)[-keep_n:].tolist())
        labels = labels_for(cr)
        n_clean_kept = sum(labels[i] for i in keep_idx)
        log(f"  [select] cr={cr} keep={keep_rate} best_metric({best_metric}): "
            f"{n_clean_kept} clean / {keep_n - n_clean_kept} defect kept")
    else:
        raise ValueError(condition)

    with h5py.File(cont, 'r') as s, h5py.File(dst, 'w') as d:
        for new_i, old_i in enumerate(keep_idx):
            s.copy(f'demo_{old_i}', d, name=f'demo_{new_i}')
    return dst


# ── best-metric selection from cache ────────────────────────────────────────────

def pick_best_metric(log):
    """Pick the single curation metric with the best cached downstream success.

    The cached benchmark (results/libero_curation_results.json) is a single
    prior run; it does not contain a dedicated cr=0.4 sweep, so we use its
    overall best-downstream metric as the proxy and say so.  Falls back to
    trajectory_alignment when no cache is available.
    """
    if not os.path.exists(CACHE_RESULTS):
        log(f"[best_metric] no cache at {CACHE_RESULTS} -> default 'trajectory_alignment'")
        return 'trajectory_alignment'
    try:
        with open(CACHE_RESULTS) as f:
            cache = json.load(f)
        metrics = cache['metrics']
    except (json.JSONDecodeError, KeyError, OSError) as e:
        log(f"[best_metric] unreadable cache ({e}) -> default 'trajectory_alignment'")
        return 'trajectory_alignment'

    ranking = sorted(metrics.items(), key=lambda kv: kv[1]['mean'], reverse=True)
    best = ranking[0][0]
    log("[best_metric] cached downstream success (highest first):")
    for name, r in ranking:
        log(f"            {name:<22} {r['mean']*100:5.1f}% ± {r.get('std',0)*100:4.1f}%"
            f"   (AUROC {r.get('auroc', float('nan')):.3f})")
    log(f"[best_metric] -> '{best}' (best cached downstream success "
        f"{metrics[best]['mean']*100:.1f}%).")
    log( "[best_metric] caveat: the cache is the prior single-rate benchmark "
         "(~cr=0.75, top-75% selection), not a dedicated cr=0.4 sweep; using the")
    log( "            overall best-downstream metric as the cr=0.4 proxy.")
    return best


# ── resumable state ─────────────────────────────────────────────────────────────

def unit_key(cr, keep_rate, condition, seed):
    return f"cr{cr:.2f}_keep{keep_rate:.2f}_{condition}_seed{seed}"


def load_state(state_path):
    if os.path.exists(state_path):
        with open(state_path) as f:
            return json.load(f)
    return {'best_metric': None, 'units': {}}


def save_state(state, state_path):
    tmp = state_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, state_path)        # atomic: a crash never truncates the state


# ── one unit = one (cell, condition, seed) ──────────────────────────────────────

def run_unit(cr, keep_rate, condition, seed, best_metric, n_rollouts, log):
    """Train (subprocess) then eval (subprocess) one unit.

    Returns the success rate, or None if either subprocess failed/segfaulted —
    in which case the caller does NOT record the unit, so it is retried later.
    """
    data = curated_subset(cr, keep_rate, condition, best_metric, log)
    key  = unit_key(cr, keep_rate, condition, seed)
    ckpt = os.path.join(CKPT_DIR, f'{key}.pt')
    os.makedirs(CKPT_DIR, exist_ok=True)

    rc, _ = run_worker(
        ['--task', 'train', '--data', data, '--ckpt', ckpt,
         '--n-epochs', EPOCHS, '--seed', seed],
        f"train {key} ({EPOCHS} epochs)", log)
    if rc != 0 or not os.path.exists(ckpt):
        return None

    rc, r = run_worker(
        ['--task', 'eval', '--ckpt', ckpt, '--n', n_rollouts],
        f"eval {key} ({n_rollouts} rollouts)", log)
    # ckpt no longer needed once evaluated
    if os.path.exists(ckpt):
        os.remove(ckpt)
    if rc != 0 or r is None:
        return None
    return r['rate']


# ── CSV outputs ─────────────────────────────────────────────────────────────────

def write_csvs(state, results_csv, summary_csv, log):
    rows = []
    for u in state['units'].values():
        rows.append((u['contamination_rate'], u['keep_rate'], u['condition'],
                     u['seed'], u['success_rate']))
    rows.sort(key=lambda r: (r[0], r[1], CONDITIONS.index(r[2]) if r[2] in CONDITIONS else 9, r[3]))

    with open(results_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['contamination_rate', 'keep_rate', 'condition', 'seed', 'success_rate'])
        w.writerows(rows)

    # per (cell, condition) mean/std over seeds
    agg = {}
    for cr, keep, cond, seed, sr in rows:
        agg.setdefault((cr, keep, cond), []).append(sr)
    with open(summary_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['contamination_rate', 'keep_rate', 'condition',
                    'mean_success', 'std_success', 'n_seeds'])
        for (cr, keep, cond), vals in sorted(
                agg.items(),
                key=lambda kv: (kv[0][0], kv[0][1],
                                CONDITIONS.index(kv[0][2]) if kv[0][2] in CONDITIONS else 9)):
            w.writerow([cr, keep, cond, float(np.mean(vals)), float(np.std(vals)), len(vals)])
    log(f"[csv] wrote {results_csv} ({len(rows)} rows) and {summary_csv}")


# ── drivers ─────────────────────────────────────────────────────────────────────

def make_logger(log_path):
    def log(msg):
        line = f"[{ts()}] {msg}"
        print(line, flush=True)
        with open(log_path, 'a') as f:
            f.write(line + '\n')
    return log


def run_sweep(units, state_path, log_path, results_csv, summary_csv,
              n_rollouts, n_total_units, is_smoke):
    os.makedirs(SWEEP_DIR, exist_ok=True)
    log = make_logger(log_path)
    log("=" * 78)
    log(f"Contamination-boundary sweep  ({'SMOKE' if is_smoke else 'FULL'})")
    log(f"grid cr={CONTAMINATION_RATES} keep={KEEP_RATES} conditions={CONDITIONS} "
        f"seeds={SEEDS}")
    log(f"training: {EPOCHS} epochs (full) | eval: {n_rollouts} headless rollouts/policy")
    log("=" * 78)

    best_metric = pick_best_metric(log)

    state = load_state(state_path)
    state['best_metric'] = best_metric
    save_state(state, state_path)

    need_ref = any(cond == 'best_metric' for (_, _, cond, _) in units)
    cells = {(cr, keep) for (cr, keep, _, _) in units}
    ensure_pools_for_cells(cells, need_ref, log)

    done = len(state['units'])
    log(f"resuming: {done}/{n_total_units} units already complete")

    for (cr, keep, cond, seed) in units:
        key = unit_key(cr, keep, cond, seed)
        if key in state['units']:
            log(f"[skip] {key} (already in state)")
            continue
        t0 = time.time()
        rate = run_unit(cr, keep, cond, seed, best_metric, n_rollouts, log)
        if rate is None:
            log(f"[FAIL] {key}: subprocess error/segfault — not recorded, will retry "
                f"on relaunch")
            continue
        state['units'][key] = {
            'contamination_rate': cr, 'keep_rate': keep, 'condition': cond,
            'seed': seed, 'success_rate': rate,
            'n_rollouts': n_rollouts, 'best_metric': best_metric,
            'elapsed_s': round(time.time() - t0, 1), 'timestamp': ts(),
        }
        save_state(state, state_path)          # checkpoint after EVERY unit
        done = len(state['units'])
        log(f"[DONE] {key}: success_rate={rate:.3f}  ({done}/{n_total_units})")

    done = len(state['units'])
    if done >= n_total_units:
        log(f"ALL {n_total_units} UNITS COMPLETE")
        write_csvs(state, results_csv, summary_csv, log)
    else:
        remaining = n_total_units - done
        log(f"progress: {done}/{n_total_units} done, {remaining} remaining — relaunch to continue")


def all_units():
    units = []
    for cr in CONTAMINATION_RATES:
        for keep in KEEP_RATES:
            for cond in CONDITIONS:
                for seed in SEEDS:
                    units.append((cr, keep, cond, seed))
    return units


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--smoke', action='store_true',
                    help="run ONE unit (cr=0.4, keep=0.5, baseline, seed=42, 5 rollouts) "
                         "into an isolated state namespace, then stop")
    args = ap.parse_args()

    if args.smoke:
        units = [(0.4, 0.5, 'baseline', SEEDS[0])]
        run_sweep(
            units,
            state_path=os.path.join(SWEEP_DIR, 'smoke_state.json'),
            log_path=os.path.join(SWEEP_DIR, 'smoke.log'),
            results_csv=os.path.join(SWEEP_DIR, 'smoke_results.csv'),
            summary_csv=os.path.join(SWEEP_DIR, 'smoke_summary.csv'),
            n_rollouts=5, n_total_units=1, is_smoke=True)
    else:
        run_sweep(
            all_units(),
            state_path=os.path.join(SWEEP_DIR, 'sweep_state.json'),
            log_path=os.path.join(ROOT, 'sweep.log'),
            results_csv=os.path.join(SWEEP_DIR, 'results.csv'),
            summary_csv=os.path.join(SWEEP_DIR, 'summary.csv'),
            n_rollouts=N_ROLLOUTS, n_total_units=N_UNITS_TOTAL, is_smoke=False)


if __name__ == '__main__':
    main()
