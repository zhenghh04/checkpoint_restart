#!/usr/bin/env python3
"""
run_monitors.py — Launch job and system monitors as background processes.

This orchestrator launches both agents alongside an HPC job and manages
their lifecycle. It is designed to be sourced from run_framework.sh or
called directly.

Usage:
  # Start both monitors in background, writing to a shared report file:
  python run_monitors.py start \
      --outputs "output.log:train.log" \
      --timeout 300 --check 10 \
      --kill-command "pkill -u $USER mpiexec" \
      --report-file monitor_reports.jsonl \
      --gpu-type auto \
      --pid-file monitor.pids

  # Stop monitors (reads PID file):
  python run_monitors.py stop --pid-file monitor.pids

  # Run a job and automatically manage monitors around it:
  python run_monitors.py run --command "mpiexec python train.py" \
      --outputs output.log --timeout 300 --report-file reports.jsonl
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


AGENTS_DIR = Path(__file__).parent


def start_monitor(script: str, extra_args: list[str], log_file: str) -> subprocess.Popen:
    cmd = [sys.executable, str(AGENTS_DIR / script)] + extra_args
    with open(log_file, "w") as fout:
        proc = subprocess.Popen(
            cmd, stdout=fout, stderr=subprocess.STDOUT, start_new_session=True
        )
    return proc


def write_pid_file(pid_file: str, pids: dict) -> None:
    with open(pid_file, "w") as f:
        for name, pid in pids.items():
            f.write(f"{name}={pid}\n")


def read_pid_file(pid_file: str) -> dict:
    pids = {}
    try:
        for line in open(pid_file):
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                pids[k] = int(v)
    except Exception:
        pass
    return pids


def kill_pids(pids: dict) -> None:
    for name, pid in pids.items():
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"Sent SIGTERM to {name} (PID {pid})", flush=True)
        except ProcessLookupError:
            print(f"{name} (PID {pid}) already gone", flush=True)
        except Exception as e:
            print(f"Failed to kill {name} (PID {pid}): {e}", flush=True)


def cmd_start(args: argparse.Namespace) -> None:
    job_log = args.log_dir + "/job_monitor.log"
    sys_log = args.log_dir + "/system_monitor.log"
    os.makedirs(args.log_dir, exist_ok=True)

    job_args = [
        "--outputs", args.outputs,
        "--timeout", str(args.timeout),
        "--check", str(args.check),
        "--report-file", args.report_file,
    ]
    if args.kill_command:
        job_args += ["--kill-command", args.kill_command]
    if args.dry_run:
        job_args += ["--dry-run"]
    if args.no_claude:
        job_args += ["--no-claude"]

    sys_args = [
        "--check", str(args.sys_check),
        "--report-interval", str(args.report_interval),
        "--report-file", args.report_file,
        "--gpu-type", args.gpu_type,
    ]
    if args.no_claude:
        sys_args += ["--no-claude"]

    print(f"Starting job monitor → {job_log}", flush=True)
    job_proc = start_monitor("job_monitor.py", job_args, job_log)
    print(f"Starting system monitor → {sys_log}", flush=True)
    sys_proc = start_monitor("system_monitor.py", sys_args, sys_log)

    pids = {"job_monitor": job_proc.pid, "system_monitor": sys_proc.pid}
    if args.pid_file:
        write_pid_file(args.pid_file, pids)
        print(f"PIDs written to {args.pid_file}: {pids}", flush=True)
    else:
        print(f"Monitor PIDs: {pids}", flush=True)
        print("(Pass --pid-file to save PIDs for later cleanup)", flush=True)


def cmd_stop(args: argparse.Namespace) -> None:
    if not args.pid_file or not Path(args.pid_file).exists():
        print(f"PID file not found: {args.pid_file}", flush=True)
        sys.exit(1)
    pids = read_pid_file(args.pid_file)
    kill_pids(pids)
    Path(args.pid_file).unlink(missing_ok=True)


def cmd_run(args: argparse.Namespace) -> None:
    """Launch both monitors, run a command, then stop monitors on exit."""
    job_log = args.log_dir + "/job_monitor.log"
    sys_log = args.log_dir + "/system_monitor.log"
    os.makedirs(args.log_dir, exist_ok=True)

    job_args = [
        "--outputs", args.outputs,
        "--timeout", str(args.timeout),
        "--check", str(args.check),
        "--report-file", args.report_file,
    ]
    if args.kill_command:
        job_args += ["--kill-command", args.kill_command]
    if args.no_claude:
        job_args += ["--no-claude"]

    sys_args = [
        "--check", str(args.sys_check),
        "--report-interval", str(args.report_interval),
        "--report-file", args.report_file,
        "--gpu-type", args.gpu_type,
    ]
    if args.no_claude:
        sys_args += ["--no-claude"]

    job_proc = start_monitor("job_monitor.py", job_args, job_log)
    sys_proc = start_monitor("system_monitor.py", sys_args, sys_log)
    print(f"Monitors started (job={job_proc.pid}, sys={sys_proc.pid}). Running: {args.command}", flush=True)

    app_result = subprocess.run(args.command, shell=True)

    job_proc.terminate()
    sys_proc.terminate()
    try:
        job_proc.wait(timeout=5)
        sys_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        job_proc.kill()
        sys_proc.kill()

    print(f"Job exited with code {app_result.returncode}. Monitors stopped.", flush=True)
    sys.exit(app_result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orchestrator for job and system monitors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # Shared args factory
    def add_shared(p: argparse.ArgumentParser) -> None:
        p.add_argument("--outputs", default="output.log", help="Colon-separated files to watch (job monitor)")
        p.add_argument("--timeout", type=int, default=300, help="Hang timeout in seconds")
        p.add_argument("--check", type=int, default=10, help="Job monitor poll interval")
        p.add_argument("--kill-command", dest="kill_command", default="", help="Kill command on hang/NaN")
        p.add_argument("--report-file", dest="report_file", default="monitor_reports.jsonl", help="Shared JSONL report file")
        p.add_argument("--gpu-type", dest="gpu_type", default="auto",
                       choices=["auto", "nvidia", "intel", "amd", "none"])
        p.add_argument("--sys-check", dest="sys_check", type=int, default=30, help="System monitor poll interval")
        p.add_argument("--report-interval", dest="report_interval", type=int, default=300,
                       help="Periodic Claude report interval for system monitor")
        p.add_argument("--log-dir", dest="log_dir", default="monitor_logs", help="Directory for monitor logs")
        p.add_argument("--dry-run", action="store_true", help="Detect but do not kill")
        p.add_argument("--no-claude", action="store_true", help="Disable Claude analysis")

    # start subcommand
    p_start = sub.add_parser("start", help="Start both monitors in background")
    add_shared(p_start)
    p_start.add_argument("--pid-file", dest="pid_file", default="monitor.pids", help="File to write PIDs to")

    # stop subcommand
    p_stop = sub.add_parser("stop", help="Stop monitors using saved PIDs")
    p_stop.add_argument("--pid-file", dest="pid_file", default="monitor.pids")

    # run subcommand
    p_run = sub.add_parser("run", help="Wrap a command with monitors")
    add_shared(p_run)
    p_run.add_argument("--command", required=True, help="Command to run (shell string)")

    args = parser.parse_args()

    if args.subcommand == "start":
        cmd_start(args)
    elif args.subcommand == "stop":
        cmd_stop(args)
    elif args.subcommand == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
