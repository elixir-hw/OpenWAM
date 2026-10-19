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

## Clean50 success-rate evaluation

`config.clean50.json` and `config.randomized50.json` describe the clean-trained 50,000-step checkpoint evaluated in `demo_clean` and `demo_randomized`. Each uses `tasks/all_50.txt`, 100 expert-accepted initial states per task, and one policy rollout per state. This produces 5,000 valid results per mode. The trainer's HDF5 demonstrations are not needed for evaluation; RoboTwin's simulator code, task configs, object and embodiment assets, and the 1,000 `unseen` background textures are needed for randomized evaluation.

Copy each checked-in template to a `config.local*.json` file and replace `REPLACE_ME` with local RoboTwin, asset, Python, and checkpoint paths. Activate the OpenWAM Python environment before running the commands below. The checkpoint directory must contain `config.yaml`, `normalization_stats.npy`, and `checkpoint_step_50000.safetensors`. The templates use all eight GPUs, one model server and six simulation clients per GPU; adjust `hardware.parallel_gpus` and `persistent_clients_per_gpu` to fit the host. Keep `workers_per_gpu=1` and `persistent_servers_per_gpu=1` when sharing model sessions.

```bash
cp config.clean50.json config.local.clean50.json
cp config.randomized50.json config.local.randomized50.json
# Edit the two local files with this host's paths.
python run_experiment1.py validate --config config.local.clean50.json
python run_experiment1.py validate --config config.local.randomized50.json

python run_experiment1.py run --config config.local.clean50.json \
  --method wm --run-dir runs/clean50_chunk --skip-text-cache
# Start randomized only after clean has completed and its summary is valid.
python run_experiment1.py run --config config.local.randomized50.json \
  --method wm --run-dir runs/randomized50_chunk --skip-text-cache
```

These formal runs use `interleaved_expert_check=true`: a client first runs the expert on each candidate scene, accepts it only when both `plan_success` and `check_success()` pass, and immediately evaluates the model on that state. Rejected candidates are not counted. Completed state results are written atomically and can be resumed with the same config and run directory. No separate manifest-building phase is needed. The runner fixes `ROBOTWIN_ENABLE_PLANNER_FALLBACK=0` to keep the expert check consistent.

The current chunk protocol has `use_action_chunk: true` in `benchmarks/robotwin/policy_config.yml` and `skip_get_obs_within_replan: true` in both Clean50 templates. The server returns an action horizon in one request; the client consumes it locally and fetches a new observation only when it replans. For the Clean50 checkpoint with `num_frames=33`, that is normally 32 executed actions per observation. During cached actions, rare randomized lighting scenes still advance their render state without capturing images. The original per-step request remains available by disabling both settings in a local copy. Results from different observation protocols should use separate run directories.

After every valid state is present, the summarizer writes `summary/summary.json`, `summary/per_task.csv`, and `summary/per_state.csv` to the run directory. Its metric is `Clean SR` or `Randomized SR`; it rejects incomplete, duplicate, errored, or truncated results. Action-capped smoke tests are not formal SR measurements.
