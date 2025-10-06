#!/usr/bin/env python3
"""
check_nan.py — Monitor text output files for NaN/Inf and terminate a job if found.

Typical usage examples:
  # Scan *.out files every 15s; if NaN found, scancel the current PBS job
  python check_nan.py --outputs "logs/*.out" --check 15 --pbs

  # Scan recursively; send SIGTERM to a PID on detection (then SIGKILL after grace)
  python check_nan.py --outputs "runs/**/stdout.txt" --recursive \
      --pid 123456 --signal TERM --grace 20

  # Provide an explicit kill command (e.g., for PBS)
  python check_nan.py --outputs "*.out" --kill-command "qdel $PBS_JOBID"

Notes:
- This script treats any case-insensitive occurrence of the tokens `nan` or `inf`
  as problematic (configurable via flags). It reads files incrementally to avoid
  re-scanning from the beginning on each poll.
"""
import argparse
import glob
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, Tuple

# --- Regex helpers -----------------------------------------------------------
NAN_RE = re.compile(r"(?<![A-Za-z0-9_])nan(?![A-Za-z0-9_])", re.IGNORECASE)
INF_RE = re.compile(r"(?<![A-Za-z0-9_])inf(?![A-Za-z0-9_])", re.IGNORECASE)

# --- Core logic --------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Monitor files for NaN/Inf and terminate a job if detected.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--outputs",
        required=True,
        help="Glob pattern for files to watch (quote to avoid shell expansion).",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Enable recursive globbing (uses glob recursive=True; ** patterns allowed).",
    )
    p.add_argument(
        "--check",
        type=int,
        default=15,
        help="Polling interval in seconds.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=0,
        help=(
            "Optional: exit with code 0 if no NaN/Inf found after this many seconds. "
            "0 disables the timeout."
        ),
    )
    p.add_argument(
        "--include-inf",
        action="store_true",
        help="Also treat 'inf' tokens as fatal (in addition to 'nan').",
    )
    p.add_argument(
        "--pid",
        type=int,
        default=0,
        help="If set, send a signal to this PID on detection (TERM by default).",
    )
    p.add_argument(
        "--signal",
        choices=["TERM", "KILL", "INT", "HUP"],
        default="TERM",
        help="Signal to send when using --pid.",
    )
    p.add_argument(
        "--grace",
        type=int,
        default=15,
        help="Seconds to wait before escalating to SIGKILL if --pid is used.",
    )
    p.add_argument(
        "--kill-command",
        default="",
        help="Arbitrary shell command to run on detection (e.g., 'qdel $PBS_JOBID').",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect and report but do not kill or run commands.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose progress messages.",
    )
    return p.parse_args()


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def vprint(verbose: bool, *msg):
    if verbose:
        print(f"[{now()}]", *msg, flush=True)


def list_files(pattern: str, recursive: bool) -> Dict[str, int]:
    # Return current files matching the pattern with their size in bytes
    files = glob.glob(pattern, recursive=recursive)
    return {f: os.path.getsize(f) for f in files if os.path.isfile(f)}


def scan_new_bytes(path: str, start: int) -> Tuple[int, str]:
    """Read bytes from a file starting at offset `start` and return (new_end, text)."""
    size = os.path.getsize(path)
    if start >= size:
        return size, ""
    with open(path, "r", errors="replace") as fh:
        fh.seek(start)
        data = fh.read()
    return size, data


def contains_bad_tokens(text: str, include_inf: bool) -> bool:
    if not text:
        return False
    if NAN_RE.search(text):
        return True
    if include_inf and INF_RE.search(text):
        return True
    return False


def try_kill(args: argparse.Namespace) -> None:
    """Attempt to terminate the job/process according to flags."""
    if args.dry_run:
        print("[DRY-RUN] Would terminate job (skipping actual kill).", flush=True)
        return

    did_something = False

    # 2) PID signaling
    if args.pid:
        sig = getattr(signal, f"SIG{args.signal}")
        print(f"Sending SIG{args.signal} to PID {args.pid}", flush=True)
        try:
            os.kill(args.pid, sig)
            did_something = True
        except ProcessLookupError:
            print(f"[WARN] PID {args.pid} does not exist.", flush=True)
        except PermissionError as e:
            print(f"[ERROR] No permission to signal PID {args.pid}: {e}", flush=True)
        except Exception as e:
            print(f"[ERROR] Failed to signal PID {args.pid}: {e}", flush=True)

        # Optional escalation to SIGKILL
        if args.grace > 0 and sig != signal.SIGKILL:
            time.sleep(args.grace)
            try:
                os.kill(args.pid, 0)
            except ProcessLookupError:
                pass  # process is gone
            else:
                print(f"Escalating to SIGKILL for PID {args.pid}", flush=True)
                try:
                    os.kill(args.pid, signal.SIGKILL)
                except Exception as e:
                    print(f"[ERROR] SIGKILL failed for PID {args.pid}: {e}", flush=True)

    # 3) Arbitrary kill command
    if args.kill_command:
        print(f"Running kill command: {args.kill_cmd}", flush=True)
        try:
            subprocess.run(args.kill_command, shell=True, check=False)
            did_something = True
        except Exception as e:
            print(f"[ERROR] Kill command failed: {e}", flush=True)

    if not did_something:
        print("[WARN] No kill action executed (provide --slurm, --pid, or --kill-command).", flush=True)


def main() -> int:
    args = parse_args()

    print(
        f"[{now()}] Monitoring for NaN{'/Inf' if args.include_inf else ''} in: {args.outputs}",
        flush=True,
    )

    offsets: Dict[str, int] = {}
    first_seen = time.time()

    while True:
        # Timeout condition (optional)
        if args.timeout and (time.time() - first_seen) >= args.timeout:
            print(f"[{now()}] Timeout reached ({args.timeout}s). No issues detected. Exiting.", flush=True)
            return 0

        # Discover files and initialize offsets for new files
        current = list_files(args.outputs, args.recursive)
        for path, size in current.items():
            if path not in offsets:
                offsets[path] = 0  # start reading from beginning for new files

        # Remove offsets for files that disappeared
        for tracked in list(offsets.keys()):
            if tracked not in current:
                del offsets[tracked]

        # Scan new bytes for each file
        for path in sorted(current.keys()):
            start = offsets.get(path, 0)
            end, chunk = scan_new_bytes(path, start)
            offsets[path] = end
            if not chunk:
                continue

            if contains_bad_tokens(chunk, include_inf=args.include_inf):
                print(f"[{now()}] Detected NaN{'/Inf' if args.include_inf else ''} in {path}.", flush=True)
                try_kill(args)
                return 2

        vprint(args.verbose, f"Scanned {len(current)} files. Sleeping {args.check}s…")
        time.sleep(args.check)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"[{now()}] Interrupted by user.")
        sys.exit(130)
