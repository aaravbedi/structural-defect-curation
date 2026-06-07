#!/usr/bin/env python3
"""
3-step curation pipeline orchestrator.

Each phase (collect / train / eval) runs in an isolated subprocess via _worker.py
to avoid MuJoCo/PyTorch memory conflicts on headless systems.

Step 1: Clean BC ground truth   (50 demos, gate >=30%)
Step 2: Contamination baselines (80 demos, gate oracle >= clean-15pp)
Step 3: Curation metrics        (6 metrics, AUROC + downstream BC success)
"""
import sys, os, subprocess, json, tempfile, shutil
import numpy as np
import h5py

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_worker.py')
PYTHON = sys.executable


def phase(args_list, label):
    """Run _worker.py with args, stream output, return parsed RESULT dict."""
    cmd = [PYTHON, WORKER] + [str(a) for a in args_list]
    print(f"\n>>> {label}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    result_json = None
    for line in proc.stdout:
        line = line.rstrip('\n')
        if line.startswith('RESULT:'):
            result_json = line[7:]
        else:
            print(line, flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Worker failed (exit {proc.returncode}): {label}")
    return json.loads(result_json) if result_json else {}


def hdf5_subset(src_path, dst_path, indices):
    """Copy demos at given indices from src_path into a new dst_path."""
    with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
        for new_i, old_i in enumerate(indices):
            src.copy(f'demo_{old_i}', dst, name=f'demo_{new_i}')


def hdf5_successful_indices(path):
    """Return indices of demos where attrs['success'] is True."""
    with h5py.File(path, 'r') as f:
        return [int(k.split('_')[1]) for k in sorted(f.keys())
                if f[k].attrs['success']]


def main():
    D = tempfile.mkdtemp(prefix='pipeline_')
    print(f"Working dir: {D}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Clean BC ground truth
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "="*62)
    print("STEP 1: Clean BC ground truth  (50 demos, 500 epochs, 50 rollouts)")
    print("="*62)

    clean_hdf5 = f'{D}/clean.hdf5'
    clean_ckpt = f'{D}/clean.pt'

    r = phase(['--task', 'collect', '--n-clean', 50,
               '--seed-clean-start', 0, '--out', clean_hdf5],
              "1a: collect 50 clean demos")
    print(f"\nScripted clean success: {r['n_success']}/{r['n_total']}")

    phase(['--task', 'train', '--data', clean_hdf5,
           '--ckpt', clean_ckpt, '--n-epochs', 500],
          "1b: train clean BC (500 epochs)")

    r1 = phase(['--task', 'eval', '--ckpt', clean_ckpt, '--n', 50],
               "1c: evaluate clean BC (50 rollouts)")
    s1_rate = r1['rate']

    print(f"\n{'='*62}")
    print(f"STEP 1 RESULT: {s1_rate:.0%}  ({r1['n_success']}/50)")
    print(f"{'='*62}")

    if s1_rate < 0.30:
        print("GATE FAILED: <30%. Stopping pipeline.")
        shutil.rmtree(D); return

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Contamination baselines
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "="*62)
    print("STEP 2: Contamination baselines  (80 demos: 20 clean + 60 defective)")
    print("="*62)

    cont_hdf5   = f'{D}/cont.hdf5'
    cont_ckpt   = f'{D}/cont.pt'
    oracle_hdf5 = f'{D}/oracle.hdf5'
    oracle_ckpt = f'{D}/oracle.pt'

    r = phase(['--task', 'collect',
               '--n-clean', 20, '--seed-clean-start', 2000,
               '--n-defect', 60, '--seed-defect-start', 1000,
               '--out', cont_hdf5],
              "2a: collect 80 contaminated demos")
    print(f"\nContaminated scripted success: {r['n_success']}/{r['n_total']}")

    oracle_idx = hdf5_successful_indices(cont_hdf5)
    hdf5_subset(cont_hdf5, oracle_hdf5, oracle_idx)
    print(f"Oracle set: {len(oracle_idx)} successful demos")

    phase(['--task', 'train', '--data', cont_hdf5,
           '--ckpt', cont_ckpt, '--n-epochs', 500],
          "2b: train contaminated BC (all 80, 500 epochs)")

    phase(['--task', 'train', '--data', oracle_hdf5,
           '--ckpt', oracle_ckpt, '--n-epochs', 500],
          f"2c: train oracle BC ({len(oracle_idx)} demos, 500 epochs)")

    r_cont = phase(['--task', 'eval', '--ckpt', cont_ckpt, '--n', 50],
                   "2d: evaluate contaminated BC (50 rollouts)")
    r_orc  = phase(['--task', 'eval', '--ckpt', oracle_ckpt, '--n', 50],
                   "2e: evaluate oracle BC (50 rollouts)")

    cont_rate   = r_cont['rate']
    oracle_rate = r_orc['rate']

    print(f"\n{'='*62}")
    print(f"STEP 2 RESULTS:")
    print(f"  Contaminated BC (80 demos): {cont_rate:.0%}  ({r_cont['n_success']}/50)")
    print(f"  Oracle BC ({len(oracle_idx):2d} demos):      {oracle_rate:.0%}  ({r_orc['n_success']}/50)")
    print(f"{'='*62}")

    if oracle_rate < s1_rate - 0.15:
        print(f"GATE FAILED: Oracle {oracle_rate:.0%} < Clean {s1_rate:.0%} - 15pp. Stopping.")
        shutil.rmtree(D); return

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: Curation metrics
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "="*62)
    print("STEP 3: Curation metrics  (6 metrics, top-75%=60 demos, 50 rollouts each)")
    print("="*62)

    # Labels: positions 0-19 clean (1), 20-79 defective (0)
    labels = [1]*20 + [0]*60

    r_sc = phase(['--task', 'score',
                  '--data', cont_hdf5, '--ref', clean_hdf5,
                  '--labels', json.dumps(labels)],
                 "3a: score all 80 contaminated demos with 6 metrics")

    all_scores = r_sc['scores']
    aurocs     = r_sc['aurocs']

    print("\n[3b] Per-metric: select top-75%, train BC, 50 rollouts...")
    top_k = int(0.75 * 80)   # 60
    metric_results = {}

    for name, sc in all_scores.items():
        sc_arr  = np.array(sc)
        top_idx = np.argsort(sc_arr)[-top_k:].tolist()
        n_clean = sum(1 for i in top_idx if i < 20)
        n_def   = top_k - n_clean
        auroc   = aurocs[name]
        print(f"\n  [{name}]  AUROC={auroc:.3f}  top-{top_k}: {n_clean} clean, {n_def} defective")

        sel_hdf5 = f'{D}/{name}_sel.hdf5'
        sel_ckpt = f'{D}/{name}.pt'
        hdf5_subset(cont_hdf5, sel_hdf5, top_idx)

        phase(['--task', 'train', '--data', sel_hdf5,
               '--ckpt', sel_ckpt, '--n-epochs', 500],
              f"  train {name} BC")

        r_m = phase(['--task', 'eval', '--ckpt', sel_ckpt, '--n', 50],
                    f"  eval {name} BC (50 rollouts)")
        m_rate = r_m['rate']
        metric_results[name] = dict(auroc=auroc, success=m_rate,
                                    vs_cont=m_rate - cont_rate, n_clean=n_clean)
        print(f"  => {name}: success={m_rate:.0%}  vs_cont={m_rate-cont_rate:+.0%}")

    # ══════════════════════════════════════════════════════════════════
    # FINAL TABLE
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "="*62)
    print("FINAL RESULTS")
    print("="*62)
    print(f"  Clean BC        (50 demos):  {s1_rate:.0%}  ({r1['n_success']}/50)")
    print(f"  Contaminated BC (80 demos):  {cont_rate:.0%}  ({r_cont['n_success']}/50)")
    print(f"  Oracle BC ({len(oracle_idx):2d} demos):       {oracle_rate:.0%}  ({r_orc['n_success']}/50)")
    print()
    print(f"  {'Metric':<22} {'AUROC':>6}  {'Success':>8}  {'vs Cont':>8}  {'Clean/60':>9}")
    print(f"  {'-'*22} {'-'*6}  {'-'*8}  {'-'*8}  {'-'*9}")
    for name, r in metric_results.items():
        print(f"  {name:<22} {r['auroc']:>6.3f}  {r['success']:>8.0%}  "
              f"{r['vs_cont']:>+8.0%}  {r['n_clean']:>9}/60")

    shutil.rmtree(D)
    print("\nPipeline complete.")


if __name__ == '__main__':
    main()
