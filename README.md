# Structural Defect Curation for Robot Demonstrations

This repository benchmarks seven demonstration-curation metrics on a controlled
structural-defect task: a pick-and-place policy trained with LIBERO where some
demos contain an early gripper release during the lift phase. The defect is
localized — only a few timesteps are wrong — making it invisible to global
statistics but catastrophic to downstream BC performance. We measure each
metric's AUROC (how well it detects defective demos) and downstream success rate
(how well a BC policy trained on the top-75% selected by that metric performs),
providing a clean benchmark for future defect-localization curation methods.

---

## Results

| Metric                | AUROC | Downstream (%) | vs Baseline (pp) |
|-----------------------|------:|---------------:|-----------------:|
| ensemble              | 0.761 |   91.1 ± 1.6   |           +87.8  |
| trajectory_alignment  | 0.638 |   90.0 ± 0.0   |           +86.7  |
| entropy               | 0.280 |   77.8 ± 12.6  |           +74.4  |
| kNN                   | 0.712 |   58.9 ± 41.7  |           +55.6  |
| smoothness            | 0.447 |   63.3 ± 30.7  |           +60.0  |
| gripper_timing        | 0.804 |   13.3 ± 16.6  |           +10.0  |
| isolation_forest      | 0.440 |    3.3 ± 0.0   |            +0.0  |
| contaminated baseline |   —   |    3.3 ± 0.0   |               0  |
| oracle (upper bound)  |   —   |   93.3 ± 0.0   |           +90.0  |

Each metric selects the top-75% (60/80) of contaminated demos by score;
downstream numbers are mean ± std over 3 random seeds × 30 rollouts each.

---

## Installation

**1. Clone LIBERO**

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /home/user/LIBERO
cd /home/user/LIBERO && pip install -e .
```

**2. Install dependencies**

```bash
pip install torch h5py scikit-learn robosuite==1.4.0 bddl easydict \
            matplotlib cloudpickle "gym==0.25.2" einops hydra-core
apt-get install -y libosmesa6-dev  # headless OpenGL rendering
```

**3. Configure LIBERO path**

```bash
mkdir -p ~/.libero
echo "libero_path: /home/user/LIBERO/libero" > ~/.libero/config.yaml
```

Set `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa` in your environment for
headless rendering.

---

## Collecting Demonstrations

Demos are collected by a scripted expert and saved to HDF5. Use `_worker.py`
directly (runs in an isolated subprocess to avoid MuJoCo/PyTorch conflicts):

```bash
# 50 clean demos
python _worker.py --task collect \
    --n-clean 50 --seed-clean-start 0 \
    --out data/raw/clean_demos.hdf5

# 80 contaminated demos (20 clean + 60 defective)
python _worker.py --task collect \
    --n-clean 20 --n-defect 60 \
    --seed-clean-start 100 --seed-defect-start 1000 \
    --out data/raw/contaminated_demos.hdf5
```

The structural defect is an early gripper release injected at 30% of the lift
phase (`release_fraction` in `configs/libero_spatial.yaml`).

---

## Running the Curation Evaluation

```bash
MUJOCO_GL=osmesa PYTHONPATH=.:$PYTHONPATH python run_curation_eval.py
```

The pipeline is resumable — completed stages are recorded in
`results/pipeline_state.json`. Re-run the same command to pick up where it
left off after an interruption.

**Stages (13 total):**

1. `collect_clean` / `collect_contaminated` — demo collection
2. `step1` — train and evaluate clean BC (gate: ≥50% success on 10 rollouts)
3. `score` — compute 6 metric scores on all 80 contaminated demos
4. `cont_baseline` — train BC on full contaminated set (3 seeds × 30 rollouts)
5. `oracle` — train BC on successful demos only (3 seeds × 30 rollouts)
6. `metric_*` — for each metric: select top-60, train 3 seeds, eval 30 rollouts

Training uses 300 epochs per seed. Eval runs 3 seeds in parallel.

---

## Reproducing the Results Table

With pre-collected demos already in `data/raw/` and completed stage JSONs in
`results/`, the results table can be regenerated without retraining:

```bash
python -c "
import sys, os; sys.path.insert(0, '.')
import run_curation_eval as p
p.print_table()
"
```

The full combined results are saved to `results/libero_curation_results.json`.

---

## Repository Layout

```
configs/               Task and training hyperparameters
data/
  collect_demos.py     Scripted expert + defect injection
  raw/                 HDF5 demo files (gitignored)
methods/
  bc_policy.py         Phase-conditioned MLP BC policy
  curation_metrics.py  Six curation metrics (smoothness, entropy,
                         gripper_timing, isolation_forest, kNN,
                         trajectory_alignment)
eval/
  evaluate.py          3-phase hybrid rollout (scripted warmup + BC + scripted transport)
_worker.py             Subprocess worker (MuJoCo/PyTorch process isolation)
run_curation_eval.py   13-stage resumable curation benchmark pipeline
results/               JSON result files and selected-demo HDF5s
```

---

## Citation

> Citation to be added after paper is posted.
