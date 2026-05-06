# Checkpoint / Restart tests on Exascale computing systems

For questions, please contact: Huihuo Zheng <huihuo.zheng@anl.gov>

Exascale computing systems often experience instabilities that can cause job terminations before completion.
To ensure large-scale simulations can continue efficiently, checkpoint/restart mechanisms are essential.

This repository provides:

- Simple programs to simulate common job execution issues: hanging, mid-run failures, and successful completion.
- Example submission scripts that automatically detect failures and restart jobs using healthy nodes.

The **key idea** is to over-allocate nodes, allowing jobs to be restarted on a healthy subset of nodes if a failure occurs.

![alt text](.docs/figures/schematic.png)

## Install the package

```bash
git clone https://github.com/argonne-lcf/checkpoint_restart
cd checkpoint_restart
pip install -e .
```

This will install the `check_hang.py`, `check_nan.py`, and `get_healthy_nodes.sh` scripts into your environment.

## Useful Scripts

This repository includes several scripts to help manage and monitor jobs.
After installation, `check_hang.py`, `check_nan.py`, and `get_healthy_nodes.sh` will be available in your PATH.

- `check_hang.py`: Monitors files for updates and kills a job if it stops changing for longer than a specified timeout.

  ```bash
  check_hang.py --timeout 600 --check 10 --command "mpiexec python train.py"
  ```

  Arguments:
  - `--timeout`: Seconds of inactivity after which the job will be killed (default: 300).
  - `--check`: Seconds between file-activity checks (default: 5).
  - `--kill-command`: Shell command to terminate the job (default: `pkill -u $USER mpiexec`).
  - `--outputs`: Colon-separated list of output files to watch (default: `chkpt/latest`).
  - `--grace`: Seconds to wait after sending the kill command before exiting (default: 10).
  - `--dry-run`: If set, do not actually run the kill command—only log the action.

- `check_nan.py`: Monitors text output files for `NaN` or `Inf` values and terminates the job if they are found.

  ```bash
  check_nan.py --outputs "logs/*.out" --check 15 --kill-command "scancel $SLURM_JOB_ID"
  ```

  Arguments:
  - `--outputs`: Glob pattern for files to watch.
  - `--recursive`: Enable recursive globbing.
  - `--check`: Polling interval in seconds (default: 15).
  - `--timeout`: Exit with code 0 if no NaN/Inf found after this many seconds (0 disables timeout).
  - `--include-inf`: Also treat 'inf' tokens as fatal.
  - `--pid`: If set, send a signal to this PID on detection.
  - `--signal`: Signal to send when using `--pid` (default: `TERM`).
  - `--grace`: Seconds to wait before escalating to `SIGKILL` if `--pid` is used (default: 15).
  - `--kill-command`: Arbitrary shell command to run on detection.
  - `--dry-run`: Detect and report but do not kill or run commands.
  - `--verbose`: Print verbose progress messages.

- `get_healthy_nodes.sh`: Selects a subset of healthy nodes from a larger allocation, writing them to a new nodefile.

  ```bash
  get_healthy_nodes.sh NODEFILE NUM_NODES_TO_SELECT NEW_NODEFILE
  ```

- `utils/flush.sh`: A utility to clean up processes on allocated nodes, excluding the head node.

  ```bash
  PBS_NODEFILE=NODEFILE ./utils/flush.sh
  ```

## Simulation of job execution: hang, fail, success

The `test_pyjob.py` script allows you to simulate various job behaviors:

```bash
--hang N              # Hang for N seconds
--fail N              # Fail after N seconds
--compute T           # Compute time per iteration
--niters NITERS       # Total number of iterations
--checkpoint PATH     # Checkpoint file path
--checkpoint_time T   # Time to write a single checkpoint
```

```bash
python test_pyjob.py --fail 120 --checkpoint ./chkpt --niters 1000
```

## Example submission scripts

- [qsub_multi_mpiexec.sc](./qsub_multi_mpiexec.sc):
  submission script doing continual trials of mpiexec until success or timeout

## System Monitoring
- [system_monitoring/README.md](./system_monitoring/README.md)
  Monitoring scripts and dashboard service for JSON-based node health visualization.

## YAML-driven microkernel health checks
- Config file: [system_monitoring/health_checks.yaml](./system_monitoring/health_checks.yaml)
- Runner: [system_monitoring/run_health_checks.py](./system_monitoring/run_health_checks.py)
- Build system (C/C++ microkernels): [utils/check_healthy_tests/CMakeLists.txt](./utils/check_healthy_tests/CMakeLists.txt)

Typical usage:
```bash
# list active checks from YAML
run_health_checks.py --list

# configure/build microkernels, then run enabled checks
run_health_checks.py --build

# run a subset by group
run_health_checks.py --build --groups injection_bisection,memory

# include checks marked disabled in YAML
run_health_checks.py --build --include-disabled --checks triad,flops
```

The YAML controls:
- which microkernels are enabled (`enabled: true|false`)
- grouping (`group`) for selective execution
- concrete launch command (`command`) and optional timeout/env
- build commands (`build.configure_command` and `build.build_command`)

By default, the YAML keeps MPI/PBS-sensitive checks disabled for local development.
Enable them on cluster allocations with:
```bash
run_health_checks.py --build --include-disabled --checks simple_injection_bisection,full_injection_bisection,triad,flops,topology
```

## Various simulation examples

- [fail/](./examples/fail): job failed after 100 seconds, restart
- [hang/](./examples/hang): job hang, kill and restart
- [success/](./examples/success): job run successfully
- [nan/](./examples/nan): NaN after a few iterations, restart

## Checkpoint interval optimization utility

- [optimal_checkpointing.py](./optimal_checkpointing.py):
  Determine the optimal time interval of computation between checkpoints
  for a job of determined node size and checkpointed memory per node

## AI Agent Monitoring (Claude-powered)

The `agents/` directory contains Claude Code (Anthropic SDK) powered monitoring
agents that provide intelligent, diagnostic-level analysis on top of the basic
polling scripts. When a problem is detected, an agent loop invokes Claude with
tool use to read recent output, inspect file state, and produce a human-readable
diagnostic report.

### Agents

- **`agents/job_monitor.py`**: Monitors output files for hangs and NaN/Inf values.
  When a problem is detected, invokes a Claude agent (with tool use) to read recent
  output, diagnose the failure mode, and produce a human-readable report.

  ```bash
  python agents/job_monitor.py \
      --outputs "output.log:train.log" \
      --timeout 300 --check 10 \
      --kill-command "pkill -u $USER mpiexec" \
      --report-file monitor_reports.jsonl
  ```

- **`agents/system_monitor.py`**: Continuously collects GPU (nvidia-smi / xpu-smi /
  rocm-smi), CPU load, and memory metrics. Detects anomalies (GPU util drop, memory
  pressure) and uses Claude to generate periodic health summaries.

  ```bash
  python agents/system_monitor.py \
      --check 30 --report-interval 300 \
      --gpu-type auto \
      --report-file sys_reports.jsonl
  ```

- **`agents/run_monitors.py`**: Orchestrator that launches both agents as background
  processes alongside a job, or wraps a command end-to-end.

  ```bash
  # Start monitors in background (returns immediately)
  python agents/run_monitors.py start \
      --outputs output.log --timeout 300 \
      --pid-file monitor.pids

  # Wrap a command (blocks until command finishes, then stops monitors)
  python agents/run_monitors.py run \
      --command "mpiexec python train.py" \
      --outputs output.log --timeout 300

  # Stop previously started monitors
  python agents/run_monitors.py stop --pid-file monitor.pids
  ```

### Integration with run_framework.sh

Set `AGENT_MONITORS=1` to use the Claude agents instead of the legacy
`check_hang.py` / `check_nan.py` polling scripts:

```bash
export AGENT_MONITORS=1             # enable Claude agents
export AGENT_GPU_TYPE=nvidia        # or intel, amd, auto
export AGENT_REPORT_FILE=agent_reports.jsonl
export ANTHROPIC_API_KEY=sk-...     # required for Claude analysis
source experiments/common/run_framework.sh
run_with_retry "mpiexec python train.py"
```

### Requirements

```bash
pip install anthropic psutil
```

`ANTHROPIC_API_KEY` must be set. Without it, the agents fall back to basic
monitoring (still detects hang/NaN, but no Claude diagnostic analysis).
