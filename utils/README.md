# Utilities

This folder contains helper scripts and microkernel assets used by checkpoint/restart workflows.

## Top-level scripts

- `get_healthy_nodes.sh`: Select reachable nodes from an allocation and write a filtered nodefile.
- `launcher.sh`: Launch helper script for multi-step runs.
- `flush.sh`: Cleanup helper for allocated nodes.
- `optimal_checkpointing.py`: Utility to estimate checkpoint interval tradeoffs.

## `check_healthy_tests/`

Microkernel source and support files used for node/communication health checks.

- Build definitions: `check_healthy_tests/CMakeLists.txt`
- Kernel/test sources:
  - `check_healthy_tests/injection_bisection_tests/`
  - `check_healthy_tests/memory_cpu_gpu_check/`
  - `check_healthy_tests/compute_node_interconnects/`

The YAML-driven health-check orchestrator now lives in `system_monitoring/`:

- `system_monitoring/run_health_checks.py`
- `system_monitoring/health_checks.yaml`

Those files call into binaries built from `utils/check_healthy_tests/CMakeLists.txt`.
