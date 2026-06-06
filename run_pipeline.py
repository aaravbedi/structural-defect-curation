"""
End-to-end pipeline: collect demos → train BC → evaluate.
Reports clean-data vs. contaminated-data baseline numbers.
"""

import sys, os
sys.path.insert(0, '/home/user/LIBERO')
sys.path.insert(0, '/home/user/structural-defect-curation')
os.environ.setdefault('MUJOCO_GL', 'osmesa')

import argparse
import json
import time
import yaml
import torch

from data.collect_demos import main as collect_demos_main
from methods.bc_policy import train as train_bc, load_policy
from eval.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/libero_spatial.yaml')
    parser.add_argument('--skip-collect', action='store_true', help='reuse existing demo files')
    parser.add_argument('--skip-train',   action='store_true', help='reuse existing policy ckpts')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print()

    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    clean_demo_path = 'data/raw/clean_demos.hdf5'
    cont_demo_path  = 'data/raw/contaminated_demos.hdf5'
    clean_ckpt_path = 'results/bc_clean.pt'
    cont_ckpt_path  = 'results/bc_contaminated.pt'

    # ── Step 1: Collect demonstrations ──────────────────────────────────────
    if not args.skip_collect:
        print("=" * 60)
        print("STEP 1: Collecting demonstrations")
        print("=" * 60)
        import sys as _sys
        _sys.argv = ['collect_demos.py', '--config', args.config]
        collect_demos_main()
    else:
        print("Skipping demo collection (--skip-collect)")
        assert os.path.exists(clean_demo_path), f"Missing {clean_demo_path}"
        assert os.path.exists(cont_demo_path),  f"Missing {cont_demo_path}"

    # ── Step 2: Train BC on clean data ───────────────────────────────────────
    if not args.skip_train:
        print("\n" + "=" * 60)
        print("STEP 2a: Training BC on CLEAN data")
        print("=" * 60)
        t0 = time.time()
        train_bc(clean_demo_path, clean_ckpt_path, cfg['train'], device=device)
        print(f"Training time: {time.time()-t0:.1f}s")

        print("\n" + "=" * 60)
        print("STEP 2b: Training BC on CONTAMINATED data")
        print("=" * 60)
        t0 = time.time()
        train_bc(cont_demo_path, cont_ckpt_path, cfg['train'], device=device)
        print(f"Training time: {time.time()-t0:.1f}s")
    else:
        print("Skipping training (--skip-train)")

    # ── Step 3: Evaluate ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Evaluating policies")
    print("=" * 60)

    print("\n--- Policy trained on CLEAN data ---")
    clean_rate = evaluate(clean_ckpt_path, cfg, device=device, label='CLEAN')

    print("\n--- Policy trained on CONTAMINATED data ---")
    cont_rate  = evaluate(cont_ckpt_path,  cfg, device=device, label='CONTAMINATED')

    # ── Summary ─────────────────────────────────────────────────────────────
    gap = clean_rate - cont_rate
    results = {
        'clean_success':         clean_rate,
        'contaminated_success':  cont_rate,
        'performance_gap':       gap,
        'config':                args.config,
    }

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Clean policy success:        {clean_rate:.1%}")
    print(f"  Contaminated policy success: {cont_rate:.1%}")
    print(f"  Performance gap:             {gap:.1%}")

    out_path = 'results/baseline_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
