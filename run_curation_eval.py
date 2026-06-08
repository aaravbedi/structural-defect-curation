#!/usr/bin/env python3
"""
Resumable end-to-end curation evaluation pipeline.

Each invocation checks results/pipeline_state.json for completed stages and
runs only what's left.  Run it repeatedly (or use run_all_stages.sh) until
it prints "ALL STAGES COMPLETE".

Stages (in order):
  collect_clean        → data/raw/clean_demos.hdf5
  collect_contaminated → data/raw/contaminated_demos.hdf5
  step1                → results/step1.json  (gate: >=50% success)
  score                → results/scores.json
  cont_baseline        → results/cont_baseline.json
  oracle               → results/oracle.json
  metric_<name>        → results/metric_<name>.json   (7 metrics)
"""

import sys, os, subprocess, json, time
import numpy as np
import h5py
from sklearn.metrics import roc_auc_score

ROOT    = os.path.dirname(os.path.abspath(__file__))
WORKER  = os.path.join(ROOT, '_worker.py')
PYTHON  = sys.executable
DATA    = os.path.join(ROOT, 'data', 'raw')
RES     = os.path.join(ROOT, 'results')
CLEAN   = os.path.join(DATA, 'clean_demos.hdf5')
CONT    = os.path.join(DATA, 'contaminated_demos.hdf5')
CONFIG  = os.path.join(ROOT, 'configs', 'libero_spatial.yaml')
STATE   = os.path.join(RES, 'pipeline_state.json')

EPOCHS      = 300
N_ROLLOUTS  = 30
SEEDS       = [42, 0, 7]
TOP_K       = 60        # top-75% of 80 demos
N_CLEAN_IN_CONT = 20
N_DEFECT        = 60

METRIC_ORDER = [
    'smoothness', 'entropy', 'gripper_timing',
    'isolation_forest', 'ensemble', 'kNN', 'trajectory_alignment',
]
ALL_STAGES = (
    ['collect_clean', 'collect_contaminated', 'step1', 'score',
     'cont_baseline', 'oracle']
    + [f'metric_{m}' for m in METRIC_ORDER]
)


# ── subprocess helpers ─────────────────────────────────────────────────────────

def run_worker(extra_args, label):
    cmd = [PYTHON, WORKER, '--config', CONFIG] + [str(a) for a in extra_args]
    print(f"\n>>> {label}", flush=True)
    env = dict(os.environ)
    env.setdefault('MUJOCO_GL', 'osmesa')
    env.setdefault('PYOPENGL_PLATFORM', 'osmesa')
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=ROOT, env=env,
    )
    result_json = None
    for line in proc.stdout:
        line = line.rstrip('\n')
        if line.startswith('RESULT:'):
            result_json = line[7:]
        elif line.strip():
            print(f"  {line}", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Worker failed (exit {proc.returncode}): {label}")
    return json.loads(result_json) if result_json else {}


# ── HDF5 helpers ───────────────────────────────────────────────────────────────

def hdf5_subset(src, dst, indices):
    with h5py.File(src, 'r') as s, h5py.File(dst, 'w') as d:
        for new_i, old_i in enumerate(indices):
            s.copy(f'demo_{old_i}', d, name=f'demo_{new_i}')


def hdf5_info(path):
    with h5py.File(path, 'r') as f:
        keys = sorted(f.keys())
        n_suc = sum(1 for k in keys if f[k].attrs['success'])
    return len(keys), n_suc


def hdf5_successful_indices(path):
    with h5py.File(path, 'r') as f:
        return [int(k.split('_')[1]) for k in sorted(f.keys())
                if f[k].attrs['success']]


# ── state helpers ──────────────────────────────────────────────────────────────

def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {'done': [], 'gate_failed': False}


def save_state(state):
    with open(STATE, 'w') as f:
        json.dump(state, f, indent=2)


def mark_done(state, stage):
    if stage not in state['done']:
        state['done'].append(stage)
    save_state(state)
    print(f"  [state] marked '{stage}' done", flush=True)


# ── stage runners ──────────────────────────────────────────────────────────────

def run_collect_clean():
    run_worker(['--task', 'collect',
                '--n-clean', 50, '--seed-clean-start', 0,
                '--out', CLEAN],
               "collect 50 clean demos")
    n, ns = hdf5_info(CLEAN)
    print(f"  clean_demos.hdf5: {n} demos, {ns} successful")


def run_collect_contaminated():
    run_worker(['--task', 'collect',
                '--n-clean', N_CLEAN_IN_CONT, '--seed-clean-start', 2000,
                '--n-defect', N_DEFECT, '--seed-defect-start', 1000,
                '--out', CONT],
               "collect 80 contaminated demos")
    n, ns = hdf5_info(CONT)
    print(f"  contaminated_demos.hdf5: {n} demos, {ns} successful")


def run_step1():
    ckpt = os.path.join(RES, 'step1_clean.pt')
    n_clean, _ = hdf5_info(CLEAN)
    run_worker(['--task', 'train', '--data', CLEAN, '--ckpt', ckpt,
                '--n-epochs', EPOCHS],
               f"Step 1: train clean BC ({n_clean} demos, {EPOCHS} epochs)")
    r = run_worker(['--task', 'eval', '--ckpt', ckpt, '--n', 10],
                   "Step 1: eval clean BC (10 rollouts)")
    rate = r.get('rate', 0.0)
    print(f"\n  Step 1: {rate:.0%} ({r.get('n_success',0)}/10)")
    result = {'rate': rate, 'n_success': r.get('n_success', 0), 'gate_passed': rate >= 0.50}
    with open(os.path.join(RES, 'step1.json'), 'w') as f:
        json.dump(result, f)
    if rate < 0.50:
        raise RuntimeError(f"GATE FAILED: clean BC = {rate:.0%} < 50%. STOP.")
    return result


def run_score():
    labels = [1] * N_CLEAN_IN_CONT + [0] * N_DEFECT
    r = run_worker(['--task', 'score',
                    '--data', CONT, '--ref', CLEAN,
                    '--labels', json.dumps(labels)],
                   "Score 80 contaminated demos (6 metrics)")
    scores = r['scores']
    aurocs = r['aurocs']
    # Add ensemble
    scores['ensemble'] = [0.5*s + 0.5*g
                          for s, g in zip(scores['smoothness'], scores['gripper_timing'])]
    aurocs['ensemble'] = float(roc_auc_score(labels, scores['ensemble']))
    print(f"  ensemble: AUROC={aurocs['ensemble']:.3f}")
    result = {'scores': scores, 'aurocs': aurocs, 'labels': labels}
    with open(os.path.join(RES, 'scores.json'), 'w') as f:
        json.dump(result, f)
    return result


def run_seeds(tag, data_hdf5, out_json):
    """Train sequentially (fast), then eval all seeds in parallel (3 MuJoCo procs)."""
    import tempfile

    # Train each seed sequentially (~20s each — fast)
    ckpts = {}
    for seed in SEEDS:
        ckpt = os.path.join(RES, f'{tag}_s{seed}.pt')
        run_worker(['--task', 'train', '--data', data_hdf5, '--ckpt', ckpt,
                    '--n-epochs', EPOCHS, '--seed', seed],
                   f"  {tag} train seed={seed}")
        ckpts[seed] = ckpt

    # Launch all eval workers simultaneously (each gets its own MuJoCo context)
    print(f"\n  {tag}: launching {len(SEEDS)} parallel eval workers "
          f"({N_ROLLOUTS} rollouts each)...", flush=True)
    env = dict(os.environ)
    env.setdefault('MUJOCO_GL', 'osmesa')
    env.setdefault('PYOPENGL_PLATFORM', 'osmesa')

    procs, tmpfiles = {}, {}
    for seed in SEEDS:
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix=f'_{tag}_{seed}.txt',
                                         delete=False)
        cmd = [PYTHON, WORKER, '--config', CONFIG,
               '--task', 'eval', '--ckpt', ckpts[seed], '--n', str(N_ROLLOUTS)]
        procs[seed]   = subprocess.Popen(cmd, stdout=tmp, stderr=subprocess.STDOUT,
                                         text=True, cwd=ROOT, env=env)
        tmpfiles[seed] = tmp.name
        tmp.close()

    # Collect results (processes run concurrently while we wait)
    rates = []
    for seed in SEEDS:
        procs[seed].wait()
        rc = procs[seed].returncode
        with open(tmpfiles[seed]) as f:
            content = f.read()
        os.unlink(tmpfiles[seed])
        result_json = None
        for line in content.splitlines():
            if line.startswith('RESULT:'):
                result_json = line[7:]
        if rc != 0 or result_json is None:
            raise RuntimeError(f"Eval worker failed for {tag} seed={seed} (exit {rc})")
        r = json.loads(result_json)
        rates.append(r['rate'])
        print(f"    seed={seed}: {r['rate']:.0%}", flush=True)

    mean = float(np.mean(rates))
    std  = float(np.std(rates))
    result = {'mean': mean, 'std': std, 'seed_rates': rates}
    with open(out_json, 'w') as f:
        json.dump(result, f)
    print(f"  => {tag}: {mean:.1%} ± {std:.1%}")
    return result


def run_cont_baseline():
    return run_seeds('cont_baseline', CONT,
                     os.path.join(RES, 'cont_baseline.json'))


def run_oracle():
    oracle_hdf5 = os.path.join(RES, 'oracle_demos.hdf5')
    if not os.path.exists(oracle_hdf5):
        idx = hdf5_successful_indices(CONT)
        hdf5_subset(CONT, oracle_hdf5, idx)
        print(f"  Oracle: {len(idx)} successful demos selected")
    return run_seeds('oracle', oracle_hdf5,
                     os.path.join(RES, 'oracle.json'))


def run_metric(mname):
    with open(os.path.join(RES, 'scores.json')) as f:
        sc_data = json.load(f)
    with open(os.path.join(RES, 'cont_baseline.json')) as f:
        baseline = json.load(f)
    scores = sc_data['scores']
    aurocs = sc_data['aurocs']
    sc_arr  = np.array(scores[mname])
    top_idx = np.argsort(sc_arr)[-TOP_K:].tolist()
    n_cl    = sum(1 for i in top_idx if i < N_CLEAN_IN_CONT)
    print(f"  {mname}: AUROC={aurocs[mname]:.3f}, top-{TOP_K}: {n_cl} clean / {TOP_K-n_cl} defective")
    sel_hdf5 = os.path.join(RES, f'{mname}_sel.hdf5')
    hdf5_subset(CONT, sel_hdf5, top_idx)
    r = run_seeds(mname, sel_hdf5, os.path.join(RES, f'metric_{mname}.json'))
    vs_baseline = r['mean'] - baseline['mean']
    # Update the partial file with extra fields
    r.update({'auroc': aurocs[mname], 'vs_baseline': vs_baseline, 'n_clean_selected': n_cl})
    with open(os.path.join(RES, f'metric_{mname}.json'), 'w') as f:
        json.dump(r, f)
    return r


# ── final table ────────────────────────────────────────────────────────────────

def print_table():
    with open(os.path.join(RES, 'cont_baseline.json')) as f:
        baseline = json.load(f)
    with open(os.path.join(RES, 'oracle.json')) as f:
        oracle = json.load(f)
    with open(os.path.join(RES, 'scores.json')) as f:
        sc_data = json.load(f)
    aurocs = sc_data['aurocs']

    metric_results = {}
    for mname in METRIC_ORDER:
        path = os.path.join(RES, f'metric_{mname}.json')
        with open(path) as f:
            metric_results[mname] = json.load(f)

    cont_mean = baseline['mean']
    cont_std  = baseline['std']

    # Save combined JSON
    combined = {
        'contaminated_baseline': baseline,
        'oracle': oracle,
        'metrics': metric_results,
    }
    with open(os.path.join(RES, 'libero_curation_results.json'), 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"\nResults saved → {os.path.join(RES, 'libero_curation_results.json')}")

    print("\n" + "="*74)
    print("RESULTS TABLE")
    print("="*74)
    print(f"{'Metric':<22} {'AUROC':>6}  {'Downstream (%)':>17}  {'vs Baseline (pp)':>17}  Flag")
    print(f"{'-'*22} {'-'*6}  {'-'*17}  {'-'*17}  ----")

    for mname in METRIC_ORDER:
        r    = metric_results[mname]
        auroc = r.get('auroc', aurocs.get(mname, float('nan')))
        mu   = r['mean'] * 100
        sd   = r['std']  * 100
        vs   = r.get('vs_baseline', r['mean'] - cont_mean) * 100
        if (r['mean'] - cont_mean) > cont_std:
            flag = 'HELPS'
        elif (r['mean'] - cont_mean) < -cont_std:
            flag = 'HURTS'
        else:
            flag = 'NEUTRAL'
        print(f"{mname:<22} {auroc:6.3f}  {mu:6.1f} ± {sd:<6.1f}  {vs:>+17.1f}  {flag}")

    vs_oracle = (oracle['mean'] - cont_mean) * 100
    print(f"{'contaminated baseline':<22} {'—':>6}  "
          f"{cont_mean*100:6.1f} ± {cont_std*100:<6.1f}  {'0':>17}")
    print(f"{'oracle':<22} {'—':>6}  "
          f"{oracle['mean']*100:6.1f} ± {oracle['std']*100:<6.1f}  {vs_oracle:>+17.1f}")
    print("="*74)
    print(f"\nFlag: HELPS > baseline+1σ | HURTS < baseline-1σ | NEUTRAL = within 1σ")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(RES,  exist_ok=True)

    state = load_state()
    done  = set(state['done'])
    print(f"Pipeline state: {len(done)}/{len(ALL_STAGES)} stages complete: {sorted(done)}")

    if state.get('gate_failed'):
        print("GATE FAILED in a prior run. Re-collect data to try again.")
        print("Delete results/pipeline_state.json and data/raw/ to restart fresh.")
        return

    def run_stage(stage_name, fn):
        if stage_name in done:
            print(f"  [skip] {stage_name} (already done)")
            return
        print(f"\n{'='*60}")
        print(f"STAGE: {stage_name}")
        print(f"{'='*60}")
        fn()
        mark_done(state, stage_name)

    try:
        # ── data collection ─────────────────────────────────────────────────
        if not os.path.exists(CLEAN):
            run_stage('collect_clean', run_collect_clean)
        else:
            mark_done(state, 'collect_clean')

        if not os.path.exists(CONT):
            run_stage('collect_contaminated', run_collect_contaminated)
        else:
            mark_done(state, 'collect_contaminated')

        # ── step 1 gate ─────────────────────────────────────────────────────
        if 'step1' not in done:
            print(f"\n{'='*60}\nSTAGE: step1\n{'='*60}")
            try:
                run_step1()
                mark_done(state, 'step1')
            except RuntimeError as e:
                if 'GATE FAILED' in str(e):
                    print(f"\n*** {e} ***")
                    state['gate_failed'] = True
                    save_state(state)
                    return
                raise

        # ── scoring ─────────────────────────────────────────────────────────
        run_stage('score', run_score)

        # ── baselines ───────────────────────────────────────────────────────
        run_stage('cont_baseline', run_cont_baseline)
        run_stage('oracle', run_oracle)

        # ── per-metric curation ─────────────────────────────────────────────
        for mname in METRIC_ORDER:
            run_stage(f'metric_{mname}', lambda m=mname: run_metric(m))

    except Exception as e:
        print(f"\n[ERROR] Stage failed: {e}")
        import traceback; traceback.print_exc()
        print("Re-run the script to resume from this stage.")
        return

    # ── all done ─────────────────────────────────────────────────────────────
    all_complete = all(s in state['done'] for s in ALL_STAGES)
    if all_complete:
        print("\n" + "="*60)
        print("ALL STAGES COMPLETE")
        print("="*60)
        print_table()
    else:
        remaining = [s for s in ALL_STAGES if s not in state['done']]
        print(f"\nProgress: {len(done)}/{len(ALL_STAGES)} done. Remaining: {remaining}")
        print("Re-run this script to continue.")


if __name__ == '__main__':
    main()
