# Parallel RoboTwin evaluation

This directory contains the resumable fixed-state evaluator used to compare OpenWAM checkpoints with and without the world model. It keeps one model server resident per GPU, runs multiple simulation clients against each server, and isolates policy state by WebSocket session.

The evaluator samples expert-solvable simulator states into reusable manifests. For every method, task, and state it then runs independent policy seeds. Each rollout is written atomically, so restarting the same command only schedules missing or invalid fragments. Checkpoints, manifests, logs, and results are not tracked by Git.

## Prerequisites

Prepare an OpenWAM environment and a RoboTwin environment, then export paths:

```bash
export OPENWAM_REPO=/path/to/OpenWAM/OpenWAM
export ROBOTWIN_REPO=/path/to/RoboTwin
export ROBOTWIN_PYTHON=/path/to/robotwin/env/bin/python
export ROBOTWIN_TASK_CONFIG_DIR=/path/to/task_config
export ROBOTWIN_ASSETS_DIR=/path/to/assets
export NO_WM_CHECKPOINT_DIR=/path/to/no_wm/checkpoint_dir
export WM_CHECKPOINT_DIR=/path/to/wm/checkpoint_dir
```

`OPENWAM_REPO` is the Python project directory containing `scripts/deploy.py`, not the outer Git checkout when the checkout has a nested `OpenWAM/` directory. The example invokes OpenWAM through `conda run -n openwam`; replace the runtime entry with `{"command": ["/absolute/path/to/python"]}` if needed.

Copy `config.example.json` to a local ignored config if you need to change the protocol. The checked-in example evaluates 10 states and 32 rollouts per state for both `no_wm` and `wm` across eight GPUs.

## Validate and run

Run commands from this directory:

```bash
python run_experiment1.py validate --config config.example.json
python run_experiment1.py build-manifests --config config.example.json
python run_experiment1.py run --config config.example.json
```

For a bounded smoke test, use one task, state, and rollout with an action cap:

```bash
python run_experiment1.py run --config config.example.json \
  --method no_wm --task-limit 1 --state-limit 1 \
  --rollouts-per-state 1 --max-action-steps 3
```

The runner prints the run directory. Resume an interrupted evaluation by passing that exact directory and the same effective configuration:

```bash
python run_experiment1.py run --config config.example.json \
  --run-dir runs/experiment1/YYYYMMDD_HHMMSS
```

A run refuses to resume if its config or protocol dimensions differ. Rollout ranges (`--rollout-start` and `--rollout-end`) are available for persistent-server runs and are useful when distributing disjoint ranges across hosts.

## Outputs

Raw fragments are stored under `openwam/<method>/raw/<task>/state_NNN/rollout_NNN/`. Once a full run finishes, the runner merges fragments and writes `summary/summary.md` plus machine-readable CSV files. The Markdown report includes overall, per-task, and per-state WM vs No-WM success rates. A capped smoke run is explicitly marked as truncated and is not treated as a success-probability estimate.

## Concurrency model

`persistent_server=true` loads each checkpoint once per GPU and phase. `persistent_clients_per_gpu` controls concurrent RoboTwin clients sharing that server. Shared clients require `shared_server_sessions=true`, which starts the OpenWAM deployment server with per-WebSocket policy state. Keep `workers_per_gpu=1`; multiple scheduler workers resetting the same server would break episode isolation.

The scheduler test covers persistent-server reuse, fragment resume, multiple server slots, shared sessions, and final fragment merging:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```
