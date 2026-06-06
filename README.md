# Structural Defect Curation

Research project: curating robot demonstration data by localizing *where* in the
state space a demo deviates, rather than using global statistics.

## Motivation

Prior work on 7 curation metrics (smoothness, entropy, length, isolation forest,
ensemble, kNN, trajectory-alignment) finds that none recover more than ~1/3 of
the downstream policy performance gap caused by **structural defects** — wrong
actions at decisive moments (e.g. early gripper release during a pick-and-place).
Action-only metrics are blind; state-trajectory metrics partially detect but
don't recover.

## Goal

Build a curation method that scores demos by WHERE they deviate in state space
(defect localization) and show it recovers substantially more of the gap than
all 7 baselines.

## Repo Layout

```
configs/          Task and training hyperparameters
data/             Demo collection scripts; raw demos go in data/raw/ (gitignored)
methods/          BC policy and (later) curation methods
eval/             Evaluation rollout harness
results/          JSON result files; checkpoints are gitignored
run_pipeline.py   End-to-end: collect → train → evaluate
```

## Environment

- LIBERO (https://github.com/Lifelong-Robot-Learning/LIBERO), task: libero_spatial task 0
  (`pick_up_the_black_bowl_between_plate_and_ramekin_and_place_it_on_plate`)
- robosuite 1.4.1, MuJoCo 3.x, PyTorch 2.x, Python 3.11
- Headless rendering via OSMesa (`MUJOCO_GL=osmesa`)

## Quick Start

```bash
# Install deps (see below)
MUJOCO_GL=osmesa PYTHONPATH=/path/to/LIBERO python run_pipeline.py
```

## Step 1 Status

Pipeline confirmed working end-to-end:
- Scripted pick-and-place expert (up→over→down trajectory avoids knocking objects)
- 30-step physics settle at episode start ensures stable object positions
- BC policy: MLP(18→256→256→7) trained with MSE on state-action pairs
- Structural defect: early gripper release at 30% of lift phase
- See `results/baseline_results.json` for clean vs. contaminated success rates
