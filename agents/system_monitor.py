#!/usr/bin/env python3
"""
system_monitor.py — Claude-powered system metrics monitoring agent.

Continuously collects GPU, CPU, and memory metrics. Detects anomalies
and uses Claude to generate human-readable health summaries.

Usage:
  python system_monitor.py --check 30 --report-interval 300 \
      --report-file sys_report.jsonl --gpu-type auto

Environment:
  ANTHROPIC_API_KEY  — required for Claude analysis
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from typing import Optional

import anthropic

MODEL = "claude-sonnet-4-6"


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] [sys-monitor] {msg}", flush=True)


def write_report(report_file: Optional[str], record: dict) -> None:
    record["timestamp"] = ts()
    if report_file:
        with open(report_file, "a") as f:
            f.write(json.dumps(record) + "\n")
    print(f"\n{'='*60}\n[SYS REPORT] {json.dumps(record, indent=2)}\n{'='*60}\n", flush=True)


# ---------------------------------------------------------------------------
# Metric collectors
# ---------------------------------------------------------------------------

def _run(cmd: str, timeout: int = 10) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout.strip() or r.stderr.strip())[:4000]
    except Exception as e:
        return f"error: {e}"


def collect_cpu_memory() -> dict:
    """Collect CPU and memory metrics using /proc or psutil."""
    metrics: dict = {}

    # CPU load from /proc/loadavg
    try:
        loadavg = open("/proc/loadavg").read().split()
        metrics["load_avg_1m"] = float(loadavg[0])
        metrics["load_avg_5m"] = float(loadavg[1])
        metrics["load_avg_15m"] = float(loadavg[2])
    except Exception:
        pass

    # Memory from /proc/meminfo
    try:
        meminfo = {}
        for line in open("/proc/meminfo"):
            parts = line.split()
            if len(parts) >= 2:
                meminfo[parts[0].rstrip(":")] = int(parts[1])
        total = meminfo.get("MemTotal", 0)
        avail = meminfo.get("MemAvailable", 0)
        if total > 0:
            metrics["mem_total_gb"] = round(total / 1024 / 1024, 1)
            metrics["mem_used_gb"] = round((total - avail) / 1024 / 1024, 1)
            metrics["mem_used_pct"] = round(100 * (total - avail) / total, 1)
    except Exception:
        pass

    # Try psutil as fallback
    if "mem_used_pct" not in metrics:
        try:
            import psutil
            vm = psutil.virtual_memory()
            metrics["mem_total_gb"] = round(vm.total / 1e9, 1)
            metrics["mem_used_gb"] = round(vm.used / 1e9, 1)
            metrics["mem_used_pct"] = round(vm.percent, 1)
            metrics["load_avg_1m"] = os.getloadavg()[0]
        except Exception:
            pass

    return metrics


def collect_nvidia_gpu() -> list[dict]:
    """Query nvidia-smi for GPU metrics."""
    fields = "index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw"
    out = _run(f"nvidia-smi --query-gpu={fields} --format=csv,noheader,nounits")
    if out.startswith("error") or not out:
        return []
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        try:
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "gpu_util_pct": float(parts[2]),
                "mem_util_pct": float(parts[3]),
                "mem_used_mb": float(parts[4]),
                "mem_total_mb": float(parts[5]),
                "temp_c": float(parts[6]),
                "power_w": float(parts[7]) if parts[7] not in ("N/A", "[N/A]") else None,
            })
        except (ValueError, IndexError):
            pass
    return gpus


def collect_intel_gpu() -> list[dict]:
    """Query xpu-smi or intel_gpu_top for Intel GPU metrics."""
    gpus = []
    # Try xpu-smi first
    out = _run("xpu-smi dump -m 0,1,2,3 -d 0 -n 1 2>/dev/null || true")
    if out and "error" not in out.lower() and out.strip():
        gpus.append({"source": "xpu-smi", "raw": out[:500]})
        return gpus
    # Try intel_gpu_top with brief output
    out = _run("timeout 3 intel_gpu_top -J -s 1000 2>/dev/null | head -40 || true")
    if out and "error" not in out.lower() and out.strip():
        try:
            data = json.loads(out.split("\n")[0])
            gpus.append({"source": "intel_gpu_top", "data": data})
        except Exception:
            gpus.append({"source": "intel_gpu_top", "raw": out[:500]})
    return gpus


def collect_amd_gpu() -> list[dict]:
    """Query rocm-smi for AMD GPU metrics."""
    out = _run("rocm-smi --showuse --showmemuse --showtemp --json 2>/dev/null || rocm-smi --json 2>/dev/null")
    if out.startswith("error") or not out:
        return []
    try:
        data = json.loads(out)
        return [{"source": "rocm-smi", "data": data}]
    except Exception:
        return [{"source": "rocm-smi", "raw": out[:500]}]


def detect_gpu_type() -> str:
    """Auto-detect available GPU tools."""
    if _run("which nvidia-smi 2>/dev/null"):
        test = _run("nvidia-smi -L 2>/dev/null")
        if test and "GPU" in test:
            return "nvidia"
    if _run("which xpu-smi 2>/dev/null") or _run("which intel_gpu_top 2>/dev/null"):
        return "intel"
    if _run("which rocm-smi 2>/dev/null"):
        return "amd"
    return "none"


def collect_gpu(gpu_type: str) -> list[dict]:
    if gpu_type == "nvidia":
        return collect_nvidia_gpu()
    elif gpu_type == "intel":
        return collect_intel_gpu()
    elif gpu_type == "amd":
        return collect_amd_gpu()
    return []


def collect_disk_io() -> dict:
    """Collect I/O wait from /proc/diskstats or iostat."""
    out = _run("iostat -x 1 1 2>/dev/null | tail -5")
    return {"iostat": out[:300]} if out and "error" not in out else {}


def collect_network() -> dict:
    """Check network interface stats."""
    out = _run("cat /proc/net/dev 2>/dev/null | head -10")
    return {"net_dev": out[:400]} if out and "error" not in out else {}


def collect_all_metrics(gpu_type: str) -> dict:
    return {
        "cpu_memory": collect_cpu_memory(),
        "gpus": collect_gpu(gpu_type),
        "disk_io": collect_disk_io(),
        "network": collect_network(),
        "hostname": _run("hostname").split()[0] if _run("hostname") else "unknown",
    }


# ---------------------------------------------------------------------------
# Anomaly detection (threshold-based, no Claude needed)
# ---------------------------------------------------------------------------

def detect_anomalies(metrics: dict, history: deque, thresholds: dict) -> list[str]:
    anomalies = []
    cpu_mem = metrics.get("cpu_memory", {})

    if cpu_mem.get("mem_used_pct", 0) > thresholds["mem_pct"]:
        anomalies.append(f"High memory usage: {cpu_mem['mem_used_pct']}%")

    gpus = metrics.get("gpus", [])
    for gpu in gpus:
        util = gpu.get("gpu_util_pct")
        if util is not None and util < thresholds["gpu_util_min"] and len(history) > 3:
            anomalies.append(f"Low GPU utilization on GPU {gpu.get('index', '?')}: {util}%")

    return anomalies


# ---------------------------------------------------------------------------
# Claude agentic analysis
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "run_diagnostic_command",
        "description": "Run a safe read-only diagnostic command on the node.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command (ps, df, nvidia-smi, etc.)"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "get_process_list",
        "description": "Get a list of top processes by CPU/memory usage.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sort_by": {"type": "string", "description": "Sort by 'cpu' or 'mem' (default: cpu)"},
            },
        },
    },
]


def dispatch_tool(name: str, inputs: dict) -> str:
    allowed = ("ps ", "top ", "df ", "du ", "ls ", "free ", "uptime",
               "nvidia-smi", "xpu-smi", "rocm-smi", "hostname", "cat /proc",
               "iostat", "vmstat", "dmesg", "mpstat", "sar ")
    if name == "run_diagnostic_command":
        cmd = inputs.get("command", "")
        if not any(cmd.startswith(p) for p in allowed):
            return f"Command not allowed: {cmd}"
        return _run(cmd)
    elif name == "get_process_list":
        sort_by = inputs.get("sort_by", "cpu")
        flag = "-%cpu" if sort_by == "cpu" else "-%mem"
        return _run(f"ps aux --sort={flag} | head -20")
    return f"Unknown tool: {name}"


def analyze_with_claude(client: anthropic.Anthropic, trigger: str, metrics: dict,
                        history: list, anomalies: list) -> str:
    prompt = (
        f"You are monitoring system health for an HPC job on an Exascale computing node.\n"
        f"Trigger: {trigger}\n\n"
        f"Current metrics:\n{json.dumps(metrics, indent=2)}\n\n"
        f"Anomalies detected: {anomalies}\n\n"
        f"Please:\n"
        f"1. Use the tools to gather more diagnostic information if needed\n"
        f"2. Assess the system health (GPU utilization, memory pressure, I/O)\n"
        f"3. Identify any bottlenecks or problems that could affect the HPC job\n"
        f"4. Provide a concise health summary and recommendations."
    )

    messages = [{"role": "user", "content": prompt}]
    max_turns = 5

    for _ in range(max_turns):
        response = client.messages.create(
            model=MODEL, max_tokens=2048, tools=TOOLS, messages=messages,
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

    return "Analysis complete."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Claude-powered system metrics monitoring agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--check", type=int, default=30, help="Metric collection interval in seconds")
    parser.add_argument("--report-interval", type=int, default=300,
                        help="Claude summary report interval in seconds (0=anomaly-only)")
    parser.add_argument("--report-file", dest="report_file", default="", help="JSONL report file")
    parser.add_argument("--gpu-type", dest="gpu_type", default="auto",
                        choices=["auto", "nvidia", "intel", "amd", "none"],
                        help="GPU type (auto-detects if unset)")
    parser.add_argument("--mem-threshold", type=float, default=90.0,
                        help="Memory usage % to trigger anomaly alert")
    parser.add_argument("--gpu-util-min", type=float, default=5.0,
                        help="GPU utilization % below which an alert is triggered")
    parser.add_argument("--no-claude", action="store_true", help="Disable Claude analysis")
    args = parser.parse_args()

    report_file = args.report_file or None
    thresholds = {"mem_pct": args.mem_threshold, "gpu_util_min": args.gpu_util_min}

    gpu_type = args.gpu_type
    if gpu_type == "auto":
        gpu_type = detect_gpu_type()
        log(f"Auto-detected GPU type: {gpu_type}")

    client = None
    if not args.no_claude:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            log("ANTHROPIC_API_KEY not set — running in basic mode")
            args.no_claude = True
        else:
            client = anthropic.Anthropic(api_key=api_key)

    log(f"System monitor started. GPU: {gpu_type} | Poll: {args.check}s | "
        f"Report interval: {args.report_interval}s | Claude: {not args.no_claude}")

    history: deque = deque(maxlen=20)
    last_report_time = time.time()

    while True:
        metrics = collect_all_metrics(gpu_type)
        history.append({"time": ts(), "metrics": metrics})
        anomalies = detect_anomalies(metrics, history, thresholds)

        if anomalies:
            log(f"Anomalies: {anomalies}")
            analysis = ""
            if not args.no_claude:
                log("Calling Claude for anomaly analysis...")
                try:
                    analysis = analyze_with_claude(
                        client, "anomaly_detected", metrics, list(history)[-5:], anomalies
                    )
                except Exception as e:
                    analysis = f"Claude analysis failed: {e}"
            write_report(report_file, {
                "event": "anomaly",
                "anomalies": anomalies,
                "metrics": metrics,
                "analysis": analysis,
            })

        # Periodic summary report
        now = time.time()
        if args.report_interval > 0 and (now - last_report_time) >= args.report_interval:
            log("Generating periodic system health report...")
            analysis = ""
            if not args.no_claude:
                try:
                    analysis = analyze_with_claude(
                        client, "periodic_report", metrics, list(history)[-10:], []
                    )
                except Exception as e:
                    analysis = f"Claude analysis failed: {e}"
            write_report(report_file, {
                "event": "periodic_report",
                "metrics": metrics,
                "analysis": analysis,
            })
            last_report_time = now
        else:
            # Short log without full report
            cpu_mem = metrics.get("cpu_memory", {})
            gpus = metrics.get("gpus", [])
            gpu_str = ""
            if gpus and isinstance(gpus[0], dict) and "gpu_util_pct" in gpus[0]:
                utils = [f"{g.get('gpu_util_pct', '?')}%" for g in gpus]
                gpu_str = f" | GPU util: {', '.join(utils)}"
            log(f"CPU load: {cpu_mem.get('load_avg_1m', '?')} | "
                f"Mem: {cpu_mem.get('mem_used_pct', '?')}%{gpu_str}")

        time.sleep(args.check)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted. Exiting.")
        sys.exit(0)
