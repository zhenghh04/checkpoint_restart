#!/usr/bin/env python3
"""
Monitor one or more output files for recent updates and kill a job if they
stop changing for longer than a timeout.

Examples:
  python check_hang.py --outputs a.out:train.log --timeout 600 \
    --check 10 --kill-command "pkill -u $USER python my_script.py"
"""
import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from time import localtime, strftime
from typing import List, Optional

__version__ = "0.2.0"


def get_date(etime: float) -> str:
    return strftime("%Y-%m-%d %H:%M:%S", localtime(etime))


def most_recent_mtime(paths: List[Path]) -> Optional[float]:
    """Return the most recent mtime among existing paths; None if none exist."""
    mtimes: List[float] = []
    for p in paths:
        try:
            if p.exists():
                mtimes.append(p.stat().st_mtime)
        except Exception:
            # Ignore transient stat errors
            pass
    return max(mtimes) if mtimes else None


def main():
    parser = argparse.ArgumentParser(
        description="Monitor output files for activity and kill a command if it hangs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--timeout",
        default=300,
        type=int,
        help="Seconds of inactivity after which the job will be killed.",
    )
    parser.add_argument(
        "--check",
        default=5,
        type=int,
        help="Seconds between file-activity checks.",
    )
    parser.add_argument(
        "--kill-command",
        dest="kill_command",
        default="pkill -u $USER mpiexec",
        type=str,
        help="Shell command to terminate the job (e.g., 'pkill -u $USER mpiexec'). "
    )
    parser.add_argument(
        "--outputs",
        dest="outputs",
        default="chkpt/latest",
        type=str,
        help="Colon-separated list of output files to watch (e.g., 'a.out:train.log').",
    )
    parser.add_argument(
        "--grace",
        default=10,
        type=int,
        help="Seconds to wait after sending the kill command before exiting.",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="If set, do not actually run the kill command—only log the action.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args()

    # Deduplicate while preserving order
    _seen = set()
    files = []
    for part in args.outputs.split(":"):
        part = part.strip()
        if not part:
            continue
        if part not in _seen:
            files.append(Path(part))
            _seen.add(part)

    start_wall = time.time()
    last_change = most_recent_mtime(files)
    if last_change is None:
        # If nothing exists yet, treat as no updates since start.
        last_change = time.time()
    last_report = 0.0

    print(f"[{get_date(time.time())}] Job monitor started")
    print(f"Watching: {', '.join(str(p) for p in files) if files else '(none)'}")
    print(f"Timeout: {args.timeout}s | Check interval: {args.check}s")

    proc = None

    try:
        while True:
            time.sleep(args.check)

            if proc:
                # Poll the process to see if it has exited
                proc.poll()
                if proc.returncode is not None:
                    print(
                        f"[{get_date(time.time())}] Monitored command exited with "
                        f"return code {proc.returncode}. Exiting monitor."
                    )
                    break

            # Update last_change if any file advanced
            current_latest = most_recent_mtime(files)
            if current_latest is not None and current_latest > last_change:
                last_change = current_latest

            now = time.time()
            idle = now - last_change
            runtime = now - start_wall

            # Periodic status line (not every loop)
            if now - last_report >= max(5, args.check):
                print(
                    f"\n[{get_date(now)}] Checking job status\n"
                    + (
                        f"Most recent change among {args.outputs} at {get_date(last_change)}\n"
                        if any(p.exists() for p in files)
                        else f"None of the watched files exist yet; monitoring...\n"
                    )
                    + f"Job has been running for {runtime:.1f} seconds\n"
                    + f"No updates for {idle:.1f} seconds"
                )
                last_report = now

            if idle >= args.timeout:
                print(
                    f"[{get_date(now)}] Output has not been updated for {idle:.1f} seconds. "
                    f"Issuing kill command..."
                )
                if not args.dry_run:
                    try:
                        if proc:
                            print(f"Killing monitored process (PID: {proc.pid})")
                            proc.kill()
                        else:
                            print(f"Executing kill command: {args.kill_command}")
                            # Use shell so "$USER" env var works in defaults
                            subprocess.run(args.kill_command, shell=True, check=True)

                        if args.grace > 0:
                            print(f"Waiting {args.grace}s grace period before exit...")
                            time.sleep(args.grace)
                    except Exception as e:
                        print(f"Failed to execute kill command: {e}", file=sys.stderr)
                else:
                    print("(dry-run) Skipping kill execution")

                print(f"[{get_date(time.time())}] Monitor exiting after inactivity timeout.")
                break
    except KeyboardInterrupt:
        print(f"\n[{get_date(time.time())}] Monitor interrupted by user. Exiting.")
        if proc and proc.returncode is None:
            print("Attempting to terminate the monitored command...")
            proc.terminate()
            try:
                proc.wait(timeout=args.grace)
                print("Command terminated.")
            except subprocess.TimeoutExpired:
                print("Command did not terminate gracefully. Forcing kill.")
                proc.kill()
    finally:
        if proc and proc.returncode is None:
            print("Ensuring monitored process is terminated before exit.")
            proc.kill()


if __name__ == "__main__":
    main()
