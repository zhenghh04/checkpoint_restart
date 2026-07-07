#!/usr/bin/env python3
"""
job_monitor.py — Claude-powered job monitoring agent.

Monitors HPC job output files for hangs and NaN/Inf values. Unlike the simple
polling scripts (check_hang.py, check_nan.py), this agent uses Claude to analyze
the job context and produce a human-readable diagnostic report on detection.

Usage:
  python job_monitor.py --outputs output.log:train.log --timeout 300 --check 10 \
      --kill-command "pkill -u $USER mpiexec" --report-file monitor_report.jsonl

Environment:
  ANTHROPIC_API_KEY  — required for Claude analysis
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic

MODEL = "claude-sonnet-4-6"
NAN_RE = re.compile(r"(?<![A-Za-z0-9_])nan(?![A-Za-z0-9_])", re.IGNORECASE)
INF_RE = re.compile(r"(?<![A-Za-z0-9_])inf(?![A-Za-z0-9_])", re.IGNORECASE)


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] [job-monitor] {msg}", flush=True)


def write_report(report_file: Optional[str], record: dict) -> None:
    record["timestamp"] = ts()
    if report_file:
        with open(report_file, "a") as f:
            f.write(json.dumps(record) + "\n")
    print(f"\n{'='*60}\n[REPORT] {json.dumps(record, indent=2)}\n{'='*60}\n", flush=True)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def tool_read_file_tail(path: str, lines: int = 50) -> str:
    try:
        p = Path(path)
        if not p.exists():
            return f"File not found: {path}"
        text = p.read_text(errors="replace")
        tail = "\n".join(text.splitlines()[-lines:])
        return tail or "(empty file)"
    except Exception as e:
        return f"Error reading {path}: {e}"


def tool_check_file_info(path: str) -> str:
    try:
        p = Path(path)
        if not p.exists():
            return json.dumps({"exists": False, "path": path})
        stat = p.stat()
        age = time.time() - stat.st_mtime
        return json.dumps({
            "exists": True,
            "path": path,
            "size_bytes": stat.st_size,
            "mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "age_seconds": round(age, 1),
        })
    except Exception as e:
        return f"Error stating {path}: {e}"


def tool_list_matching_files(pattern: str) -> str:
    try:
        matches = glob.glob(pattern, recursive=True)
        return json.dumps(matches[:30])
    except Exception as e:
        return f"Error globbing {pattern}: {e}"


def tool_run_safe_command(command: str) -> str:
    """Run read-only diagnostic commands only."""
    allowed_prefixes = ("ps ", "top ", "df ", "du ", "ls ", "cat ", "tail ", "head ",
                        "hostname", "date", "uptime", "free ", "nvidia-smi", "xpu-smi",
                        "rocm-smi", "qstat", "squeue", "sacct")
    if not any(command.startswith(p) for p in allowed_prefixes):
        return f"Command not allowed for safety: {command}"
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=15
        )
        out = result.stdout.strip() or result.stderr.strip()
        return out[:3000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Command timed out"
    except Exception as e:
        return f"Error: {e}"


TOOLS = [
    {
        "name": "read_file_tail",
        "description": "Read the last N lines of a file to see recent job output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to file"},
                "lines": {"type": "integer", "description": "Number of lines from end (default 50)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "check_file_info",
        "description": "Get file modification time, size, and age to detect stale outputs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to file"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_matching_files",
        "description": "List files matching a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern (e.g. 'logs/*.out')"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "run_safe_command",
        "description": "Run a read-only diagnostic command (ps, df, nvidia-smi, etc.).",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
            },
            "required": ["command"],
        },
    },
]


def dispatch_tool(name: str, inputs: dict) -> str:
    if name == "read_file_tail":
        return tool_read_file_tail(inputs["path"], inputs.get("lines", 50))
    elif name == "check_file_info":
        return tool_check_file_info(inputs["path"])
    elif name == "list_matching_files":
        return tool_list_matching_files(inputs["pattern"])
    elif name == "run_safe_command":
        return tool_run_safe_command(inputs["command"])
    return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# Claude agentic analysis
# ---------------------------------------------------------------------------

def analyze_with_claude(client: anthropic.Anthropic, issue_type: str, context: dict) -> str:
    """Run a Claude agentic loop to analyze the detected issue."""
    watched = context.get("watched_files", [])
    prompt = (
        f"You are monitoring an HPC job on an Exascale computing system.\n"
        f"Issue detected: **{issue_type}**\n\n"
        f"Context:\n{json.dumps(context, indent=2)}\n\n"
        f"Please analyze the situation using the provided tools to:\n"
        f"1. Read the last few lines of these output files to understand what the job was doing: {watched}\n"
        f"2. Check file modification times to confirm hang duration\n"
        f"3. Run `ps aux | head -20` or similar to check running processes\n"
        f"4. Provide a concise diagnostic report: what happened, likely cause, recommended action."
    )

    messages = [{"role": "user", "content": prompt}]
    max_turns = 6

    for _ in range(max_turns):
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = dispatch_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            break

    return "Analysis complete (no further output from model)."


# ---------------------------------------------------------------------------
# Monitoring state
# ---------------------------------------------------------------------------

def get_file_mtimes(paths: list[Path]) -> dict:
    result = {}
    for p in paths:
        try:
            if p.exists():
                result[str(p)] = p.stat().st_mtime
        except Exception:
            pass
    return result


def scan_nan(paths: list[Path], offsets: dict, include_inf: bool) -> Optional[tuple[str, str]]:
    """Return (path, snippet) if NaN/Inf detected in new bytes, else None."""
    for p in paths:
        try:
            if not p.exists():
                continue
            size = p.stat().st_size
            start = offsets.get(str(p), 0)
            if start >= size:
                continue
            with open(p, "r", errors="replace") as fh:
                fh.seek(start)
                chunk = fh.read()
            offsets[str(p)] = size
            if NAN_RE.search(chunk) or (include_inf and INF_RE.search(chunk)):
                snippet = chunk[-500:] if len(chunk) > 500 else chunk
                return str(p), snippet
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Claude-powered job monitoring agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--outputs", default="output.log", help="Colon-separated output files to watch")
    parser.add_argument("--timeout", type=int, default=300, help="Hang timeout in seconds")
    parser.add_argument("--check", type=int, default=10, help="Poll interval in seconds")
    parser.add_argument("--kill-command", dest="kill_command", default="", help="Shell command to kill the job")
    parser.add_argument("--include-inf", action="store_true", help="Also treat Inf as fatal")
    parser.add_argument("--report-file", dest="report_file", default="", help="JSONL file to write reports to")
    parser.add_argument("--dry-run", action="store_true", help="Detect but do not kill")
    parser.add_argument("--no-claude", action="store_true", help="Skip Claude analysis (basic mode only)")
    args = parser.parse_args()

    paths = [Path(p.strip()) for p in args.outputs.split(":") if p.strip()]
    report_file = args.report_file or None

    client = None
    if not args.no_claude:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            log("ANTHROPIC_API_KEY not set — running in basic mode (no Claude analysis)")
            args.no_claude = True
        else:
            client = anthropic.Anthropic(api_key=api_key)

    log(f"Job monitor started. Watching: {[str(p) for p in paths]}")
    log(f"Hang timeout: {args.timeout}s | Poll: {args.check}s | Claude: {not args.no_claude}")

    nan_offsets: dict = {}
    last_mtimes = get_file_mtimes(paths)
    last_change = max(last_mtimes.values()) if last_mtimes else time.time()
    start_time = time.time()

    while True:
        time.sleep(args.check)
        now = time.time()

        # --- Hang detection ---
        current_mtimes = get_file_mtimes(paths)
        latest = max(current_mtimes.values()) if current_mtimes else last_change
        if latest > last_change:
            last_change = latest

        idle = now - last_change
        runtime = now - start_time
        log(f"Runtime: {runtime:.0f}s | Idle: {idle:.0f}s / {args.timeout}s")

        if idle >= args.timeout:
            context = {
                "issue": "hang",
                "watched_files": [str(p) for p in paths],
                "idle_seconds": round(idle, 1),
                "runtime_seconds": round(runtime, 1),
                "file_mtimes": {k: datetime.fromtimestamp(v).strftime("%H:%M:%S")
                                for k, v in current_mtimes.items()},
            }
            log(f"HANG DETECTED: no output for {idle:.0f}s")

            analysis = ""
            if not args.no_claude:
                log("Calling Claude for diagnostic analysis...")
                try:
                    analysis = analyze_with_claude(client, "JOB HANG", context)
                except Exception as e:
                    analysis = f"Claude analysis failed: {e}"

            write_report(report_file, {
                "event": "hang_detected",
                "idle_seconds": round(idle),
                "analysis": analysis,
            })

            if args.kill_command and not args.dry_run:
                log(f"Executing kill command: {args.kill_command}")
                subprocess.run(args.kill_command, shell=True)

            sys.exit(2)

        # --- NaN detection ---
        hit = scan_nan(paths, nan_offsets, args.include_inf)
        if hit:
            path_hit, snippet = hit
            context = {
                "issue": "nan_detected",
                "watched_files": [str(p) for p in paths],
                "detected_in": path_hit,
                "snippet": snippet,
                "runtime_seconds": round(runtime, 1),
            }
            log(f"NaN/Inf DETECTED in {path_hit}")

            analysis = ""
            if not args.no_claude:
                log("Calling Claude for diagnostic analysis...")
                try:
                    analysis = analyze_with_claude(client, "NaN/Inf DETECTED", context)
                except Exception as e:
                    analysis = f"Claude analysis failed: {e}"

            write_report(report_file, {
                "event": "nan_detected",
                "file": path_hit,
                "snippet": snippet[:200],
                "analysis": analysis,
            })

            if args.kill_command and not args.dry_run:
                log(f"Executing kill command: {args.kill_command}")
                subprocess.run(args.kill_command, shell=True)

            sys.exit(2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted by user. Exiting.")
        sys.exit(0)
