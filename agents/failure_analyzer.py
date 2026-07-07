#!/usr/bin/env python3
"""
failure_analyzer.py — Claude-powered failure analysis and restart decision agent.

Analyzes a failed HPC job and classifies the failure into one of three categories:
  - system_fault:        transient hardware/network issue → resubmit
  - numerical_divergence: NaN/Inf or loss explosion       → resubmit (possibly with tuning)
  - application_bug:     segfault, assertion, logic error  → quit

Prints a JSON verdict to stdout:
  {"decision": "resubmit"|"quit", "failure_type": "...", "confidence": 0-1,
   "diagnosis": "...", "recommended_action": "..."}

Exit codes:
  0 → resubmit
  1 → quit
  2 → analysis failed (caller should default to resubmit for safety)

Usage:
  verdict=$(python agents/failure_analyzer.py \
      --exit-code $EXIT_CODE \
      --outputs "output.log:train.log" \
      --trial $RUN --max-trials $MAX_TRIALS)
  decision=$(echo "$verdict" | python3 -c "import sys,json; print(json.load(sys.stdin)['decision'])")

Environment:
  ANTHROPIC_API_KEY  — required; without it the agent falls back to heuristics only
"""
import argparse
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

# Heuristic patterns for quick pre-classification (no Claude needed)
SYSTEM_FAULT_PATTERNS = [
    r"node\s+fail",
    r"killed by signal 9",
    r"oom.kill",
    r"fabric error",
    r"network error",
    r"drpc.*failed",
    r"communication error",
    r"mpi.*abort",
    r"lost contact",
    r"timeout.*expired",
    r"hardware error",
]

NUMERICAL_PATTERNS = [
    r"(?<![A-Za-z])nan(?![A-Za-z])",
    r"(?<![A-Za-z])inf(?![A-Za-z])",
    r"loss.*explod",
    r"gradient.*overflow",
    r"numerical.*instabilit",
]

BUG_PATTERNS = [
    r"assertion.*fail",
    r"segmentation fault",
    r"core dumped",
    r"undefined behavior",
    r"out of bounds",
    r"RuntimeError",
    r"AttributeError",
    r"TypeError",
    r"KeyError",
    r"IndexError",
    r"syntax error",
    r"import error",
    r"module not found",
]


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _run(cmd: str, timeout: int = 10) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout.strip() or r.stderr.strip())[:3000]
    except Exception as e:
        return f"error: {e}"


def read_tail(path: str, lines: int = 80) -> str:
    try:
        p = Path(path)
        if not p.exists():
            return f"(file not found: {path})"
        text = p.read_text(errors="replace")
        return "\n".join(text.splitlines()[-lines:]) or "(empty)"
    except Exception as e:
        return f"error reading {path}: {e}"


def heuristic_classify(logs: str) -> Optional[str]:
    """Quick regex-based pre-classification. Returns None if ambiguous."""
    logs_lower = logs.lower()
    bug_score = sum(1 for p in BUG_PATTERNS if re.search(p, logs_lower))
    sys_score = sum(1 for p in SYSTEM_FAULT_PATTERNS if re.search(p, logs_lower))
    nan_score = sum(1 for p in NUMERICAL_PATTERNS if re.search(p, logs_lower))

    # Only return a classification if one category clearly dominates
    scores = {"application_bug": bug_score, "system_fault": sys_score,
               "numerical_divergence": nan_score}
    best = max(scores, key=scores.get)
    if scores[best] >= 2 and sum(scores.values()) > 0:
        # Dominant category wins if it has at least 2 signals and > 60% share
        total = sum(scores.values())
        if scores[best] / total >= 0.6:
            return best
    return None


# ---------------------------------------------------------------------------
# Tools for Claude
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
HEALTH_CHECK_SCRIPT = REPO_ROOT / "system_monitoring" / "run_health_checks.py"
HEALTH_CHECK_YAML = REPO_ROOT / "system_monitoring" / "health_checks.yaml"

TOOLS = [
    {
        "name": "run_node_health_checks",
        "description": (
            "Run hardware health microkernels to distinguish system faults from application bugs. "
            "Available groups: memory (mem_and_gpu_row, fast, no MPI), "
            "injection_bisection (MPI network test), smoke (hello_world_scale). "
            "Run 'memory' first; only run 'injection_bisection' if you suspect network issues."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "groups": {
                    "type": "string",
                    "description": "Comma-separated check groups to run (e.g. 'memory,smoke')",
                },
                "checks": {
                    "type": "string",
                    "description": "Comma-separated specific check IDs (overrides groups)",
                },
            },
        },
    },
    {
        "name": "read_file_tail",
        "description": "Read the last N lines of a job output/error file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "lines": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_log_pattern",
        "description": "Search a file for lines matching a regex pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "pattern": {"type": "string", "description": "Python regex"},
                "context_lines": {"type": "integer", "description": "Lines of context around match"},
            },
            "required": ["path", "pattern"],
        },
    },
    {
        "name": "run_diagnostic",
        "description": "Run a safe read-only diagnostic command.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "submit_verdict",
        "description": "Submit the final analysis verdict.",
        "input_schema": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["resubmit", "quit"],
                    "description": "'resubmit' for system/numerical issues; 'quit' for application bugs",
                },
                "failure_type": {
                    "type": "string",
                    "enum": ["system_fault", "numerical_divergence", "application_bug", "unknown"],
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence in the classification, 0.0 to 1.0",
                },
                "diagnosis": {
                    "type": "string",
                    "description": "One-paragraph explanation of what went wrong",
                },
                "recommended_action": {
                    "type": "string",
                    "description": "Concise recommended next step",
                },
            },
            "required": ["decision", "failure_type", "confidence", "diagnosis", "recommended_action"],
        },
    },
]


def run_health_check_tool(groups: str = "", checks: str = "") -> str:
    """Invoke the YAML-driven health check runner."""
    if not HEALTH_CHECK_SCRIPT.exists():
        return f"Health check script not found: {HEALTH_CHECK_SCRIPT}"

    cmd = [sys.executable, str(HEALTH_CHECK_SCRIPT),
           "--config", str(HEALTH_CHECK_YAML)]
    if checks:
        cmd += ["--checks", checks]
    elif groups:
        cmd += ["--groups", groups]
    else:
        # Default: run memory group only (fast, no MPI)
        cmd += ["--groups", "memory"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        out = (result.stdout + result.stderr).strip()
        return out[:4000] if out else "(no output from health checks)"
    except subprocess.TimeoutExpired:
        return "Health checks timed out (>120s)"
    except Exception as e:
        return f"Health check error: {e}"


def dispatch_tool(name: str, inputs: dict) -> tuple[str, Optional[dict]]:
    """Returns (tool_output_str, verdict_dict_or_None)."""
    if name == "run_node_health_checks":
        return run_health_check_tool(
            groups=inputs.get("groups", "memory"),
            checks=inputs.get("checks", ""),
        ), None
    elif name == "read_file_tail":
        return read_tail(inputs["path"], inputs.get("lines", 80)), None

    elif name == "search_log_pattern":
        path = inputs["path"]
        pattern = inputs["pattern"]
        ctx = inputs.get("context_lines", 2)
        try:
            p = Path(path)
            if not p.exists():
                return f"File not found: {path}", None
            lines = p.read_text(errors="replace").splitlines()
            results = []
            for i, line in enumerate(lines):
                if re.search(pattern, line, re.IGNORECASE):
                    start = max(0, i - ctx)
                    end = min(len(lines), i + ctx + 1)
                    block = lines[start:end]
                    results.append(f"--- match at line {i+1} ---\n" + "\n".join(block))
            return "\n\n".join(results[:10]) if results else f"No matches for '{pattern}' in {path}", None
        except Exception as e:
            return f"Error: {e}", None

    elif name == "run_diagnostic":
        cmd = inputs.get("command", "")
        allowed = ("ps ", "top ", "df ", "du ", "ls ", "free ", "uptime",
                   "nvidia-smi", "xpu-smi", "rocm-smi", "cat /proc",
                   "dmesg", "tail ", "head ", "grep ", "wc ")
        if not any(cmd.startswith(p) for p in allowed):
            return f"Command not allowed: {cmd}", None
        return _run(cmd), None

    elif name == "submit_verdict":
        return "Verdict submitted.", inputs

    return f"Unknown tool: {name}", None


def analyze_with_claude(client: anthropic.Anthropic, context: dict) -> dict:
    """Run Claude agentic loop to classify failure and produce verdict."""
    outputs = context.get("outputs", [])
    exit_code = context.get("exit_code", -1)
    trial = context.get("trial", 1)
    max_trials = context.get("max_trials", 10)

    prompt = (
        f"You are analyzing a failed HPC job to decide whether to resubmit or quit.\n\n"
        f"Job context:\n"
        f"  - Exit code: {exit_code}\n"
        f"  - Trial: {trial} of {max_trials}\n"
        f"  - Output files: {outputs}\n"
        f"  - Heuristic pre-classification: {context.get('heuristic', 'ambiguous')}\n\n"
        f"Steps:\n"
        f"1. Read the tail of each output file to find error messages.\n"
        f"2. Search for specific error patterns (segfault, NaN, MPI abort, etc.).\n"
        f"3. If logs are ambiguous, run node health checks:\n"
        f"   - run_node_health_checks(groups='memory') for GPU/CPU memory health (fast)\n"
        f"   - run_node_health_checks(groups='injection_bisection') if you suspect network issues (needs MPI)\n"
        f"4. Classify the failure as one of:\n"
        f"   - system_fault (transient hardware/network issue → resubmit)\n"
        f"   - numerical_divergence (NaN/Inf/loss explosion → resubmit)\n"
        f"   - application_bug (segfault, assertion, Python exception → quit)\n"
        f"   Tip: if health checks PASS, the failure is likely an application bug.\n"
        f"        if health checks FAIL, it is a system fault.\n"
        f"5. Call submit_verdict with your final decision.\n\n"
        f"Important: if trial >= max_trials, always set decision='quit' regardless of type."
    )

    messages = [{"role": "user", "content": prompt}]
    verdict = None
    max_turns = 8

    for _ in range(max_turns):
        response = client.messages.create(
            model=MODEL, max_tokens=2048, tools=TOOLS, messages=messages,
        )

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    output, v = dispatch_tool(block.name, block.input)
                    if v is not None:
                        verdict = v
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    })
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            if verdict is not None:
                break
        else:
            break

    return verdict or {
        "decision": "resubmit",
        "failure_type": "unknown",
        "confidence": 0.3,
        "diagnosis": "Could not determine failure cause. Defaulting to resubmit.",
        "recommended_action": "Review logs manually.",
    }


# ---------------------------------------------------------------------------
# Heuristic-only fallback
# ---------------------------------------------------------------------------

DECISION_MAP = {
    "system_fault": "resubmit",
    "numerical_divergence": "resubmit",
    "application_bug": "quit",
}


def heuristic_verdict(exit_code: int, outputs: list[str], trial: int, max_trials: int) -> dict:
    combined_logs = ""
    for path in outputs:
        combined_logs += read_tail(path, 100) + "\n"

    failure_type = heuristic_classify(combined_logs) or "unknown"
    decision = DECISION_MAP.get(failure_type, "resubmit")
    if trial >= max_trials:
        decision = "quit"

    return {
        "decision": decision,
        "failure_type": failure_type,
        "confidence": 0.5 if failure_type != "unknown" else 0.2,
        "diagnosis": f"Heuristic classification (no Claude). Exit code: {exit_code}. "
                     f"Pattern match: {failure_type}.",
        "recommended_action": "quit" if decision == "quit" else "resubmit on healthy nodes",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a failed job and decide resubmit vs quit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--exit-code", dest="exit_code", type=int, required=True,
                        help="Exit code of the failed job")
    parser.add_argument("--outputs", default="output.log",
                        help="Colon-separated output files to analyze")
    parser.add_argument("--trial", type=int, default=1, help="Current trial number")
    parser.add_argument("--max-trials", dest="max_trials", type=int, default=10)
    parser.add_argument("--report-file", dest="report_file", default="",
                        help="Append verdict to this JSONL file")
    parser.add_argument("--no-claude", action="store_true", help="Use heuristics only")
    args = parser.parse_args()

    outputs = [p.strip() for p in args.outputs.split(":") if p.strip()]

    # Force quit if max trials reached
    if args.trial >= args.max_trials:
        verdict = {
            "decision": "quit",
            "failure_type": "max_trials_reached",
            "confidence": 1.0,
            "diagnosis": f"Trial {args.trial} reached max_trials={args.max_trials}.",
            "recommended_action": "Maximum retries reached. Job will not be resubmitted.",
        }
        verdict["timestamp"] = ts()
        print(json.dumps(verdict))
        if args.report_file:
            with open(args.report_file, "a") as f:
                f.write(json.dumps(verdict) + "\n")
        sys.exit(1)

    # Pre-classify with heuristics
    combined_logs = "".join(read_tail(p, 100) for p in outputs)
    heuristic = heuristic_classify(combined_logs)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if args.no_claude or not api_key:
        verdict = heuristic_verdict(args.exit_code, outputs, args.trial, args.max_trials)
    else:
        try:
            client = anthropic.Anthropic(api_key=api_key)
            context = {
                "exit_code": args.exit_code,
                "outputs": outputs,
                "trial": args.trial,
                "max_trials": args.max_trials,
                "heuristic": heuristic or "ambiguous",
            }
            verdict = analyze_with_claude(client, context)
        except Exception as e:
            verdict = heuristic_verdict(args.exit_code, outputs, args.trial, args.max_trials)
            verdict["diagnosis"] += f" (Claude failed: {e})"

    verdict["timestamp"] = ts()
    print(json.dumps(verdict))

    if args.report_file:
        with open(args.report_file, "a") as f:
            f.write(json.dumps(verdict) + "\n")

    sys.exit(0 if verdict["decision"] == "resubmit" else 1)


if __name__ == "__main__":
    main()
