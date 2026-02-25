#!/usr/bin/env python3
"""Simple dashboard server for visualizing node health JSON data."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional


def iso_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def parse_timestamp(value: Optional[str]) -> Optional[dt.datetime]:
    if not value or not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return dt.datetime.fromisoformat(raw)
    except Exception:
        return None


def normalize_status(record: Dict) -> str:
    status = str(record.get("status", "")).strip().lower()
    if status in {"healthy", "unhealthy", "warning", "unknown"}:
        return status

    if "healthy" in record:
        return "healthy" if bool(record["healthy"]) else "unhealthy"

    score = record.get("score")
    if isinstance(score, (int, float)):
        if score >= 0.8:
            return "healthy"
        if score >= 0.5:
            return "warning"
        return "unhealthy"

    return "unknown"


def normalize_node(record: Dict) -> str:
    for key in ("node", "hostname", "host", "name"):
        if key in record and str(record[key]).strip():
            return str(record[key]).strip()
    return "unknown"


@dataclass
class NodeState:
    node: str
    status: str
    timestamp: Optional[str]
    source_file: str
    details: Dict


class HealthDataStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._lock = threading.Lock()
        self._node_map: Dict[str, NodeState] = {}
        self._source_files: List[str] = []
        self._last_refresh_utc = iso_now()

    def refresh(self) -> None:
        files = sorted(self.data_dir.glob("*.json"))
        next_map: Dict[str, NodeState] = {}
        source_names: List[str] = []

        for path in files:
            source_names.append(path.name)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue

            rows = []
            if isinstance(payload, list):
                rows = [r for r in payload if isinstance(r, dict)]
            elif isinstance(payload, dict):
                if isinstance(payload.get("nodes"), list):
                    rows = [r for r in payload["nodes"] if isinstance(r, dict)]
                else:
                    rows = [payload]

            for row in rows:
                node = normalize_node(row)
                status = normalize_status(row)
                ts_raw = row.get("timestamp") or row.get("time")
                ts = str(ts_raw).strip() if ts_raw is not None else None

                candidate = NodeState(
                    node=node,
                    status=status,
                    timestamp=ts,
                    source_file=path.name,
                    details=row,
                )

                prev = next_map.get(node)
                if prev is None:
                    next_map[node] = candidate
                    continue

                prev_dt = parse_timestamp(prev.timestamp)
                curr_dt = parse_timestamp(candidate.timestamp)
                if prev_dt is None and curr_dt is not None:
                    next_map[node] = candidate
                elif prev_dt is not None and curr_dt is not None and curr_dt >= prev_dt:
                    next_map[node] = candidate

        with self._lock:
            self._node_map = next_map
            self._source_files = source_names
            self._last_refresh_utc = iso_now()

    def snapshot(self) -> Dict:
        with self._lock:
            nodes = list(self._node_map.values())
            files = list(self._source_files)
            last_refresh = self._last_refresh_utc

        status_counts = {"healthy": 0, "warning": 0, "unhealthy": 0, "unknown": 0}
        for n in nodes:
            status_counts[n.status] = status_counts.get(n.status, 0) + 1

        return {
            "last_refresh_utc": last_refresh,
            "data_dir": str(self.data_dir),
            "source_files": files,
            "summary": {
                "total_nodes": len(nodes),
                "healthy": status_counts.get("healthy", 0),
                "warning": status_counts.get("warning", 0),
                "unhealthy": status_counts.get("unhealthy", 0),
                "unknown": status_counts.get("unknown", 0),
            },
            "nodes": [
                {
                    "node": n.node,
                    "status": n.status,
                    "timestamp": n.timestamp,
                    "source_file": n.source_file,
                    "details": n.details,
                }
                for n in sorted(nodes, key=lambda x: x.node)
            ],
        }


def build_index_html(refresh_seconds: int) -> str:
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Node Health Dashboard</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --panel: #ffffff;
      --ink: #1f2937;
      --muted: #64748b;
      --healthy: #0f766e;
      --warning: #b45309;
      --unhealthy: #b91c1c;
      --unknown: #475569;
      --accent: #0ea5e9;
    }}
    body {{ margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif; background: radial-gradient(circle at 15% 10%, #dbeafe, transparent 35%), var(--bg); color: var(--ink); }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
    .head {{ display: flex; justify-content: space-between; align-items: baseline; gap: 12px; flex-wrap: wrap; }}
    .title {{ font-size: 28px; font-weight: 700; letter-spacing: 0.01em; }}
    .meta {{ color: var(--muted); font-size: 14px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-top: 16px; }}
    .card {{ background: var(--panel); border-radius: 12px; padding: 14px; box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08); }}
    .card .k {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }}
    .card .v {{ font-size: 28px; font-weight: 700; line-height: 1.1; margin-top: 6px; }}
    .healthy .v {{ color: var(--healthy); }}
    .warning .v {{ color: var(--warning); }}
    .unhealthy .v {{ color: var(--unhealthy); }}
    .unknown .v {{ color: var(--unknown); }}
    .toolbar {{ margin-top: 18px; display: flex; gap: 10px; flex-wrap: wrap; }}
    input, select {{ border: 1px solid #cbd5e1; border-radius: 8px; padding: 8px 10px; font-size: 14px; background: white; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 14px; background: var(--panel); border-radius: 12px; overflow: hidden; box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08); }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e2e8f0; text-align: left; font-size: 14px; }}
    th {{ background: #f8fafc; font-weight: 600; color: #334155; }}
    tr:last-child td {{ border-bottom: none; }}
    .pill {{ display: inline-block; font-size: 12px; font-weight: 700; border-radius: 999px; padding: 3px 8px; color: white; text-transform: uppercase; }}
    .pill.healthy {{ background: var(--healthy); }}
    .pill.warning {{ background: var(--warning); }}
    .pill.unhealthy {{ background: var(--unhealthy); }}
    .pill.unknown {{ background: var(--unknown); }}
    .small {{ color: var(--muted); font-size: 12px; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"head\">
      <div class=\"title\">Node Health Dashboard</div>
      <div class=\"meta\" id=\"meta\">Loading...</div>
    </div>

    <div class=\"cards\">
      <div class=\"card\"><div class=\"k\">Total Nodes</div><div class=\"v\" id=\"total\">0</div></div>
      <div class=\"card healthy\"><div class=\"k\">Healthy</div><div class=\"v\" id=\"healthy\">0</div></div>
      <div class=\"card warning\"><div class=\"k\">Warning</div><div class=\"v\" id=\"warning\">0</div></div>
      <div class=\"card unhealthy\"><div class=\"k\">Unhealthy</div><div class=\"v\" id=\"unhealthy\">0</div></div>
      <div class=\"card unknown\"><div class=\"k\">Unknown</div><div class=\"v\" id=\"unknown\">0</div></div>
    </div>

    <div class=\"toolbar\">
      <input id=\"search\" placeholder=\"Filter nodes...\" />
      <select id=\"status\">
        <option value=\"all\">All status</option>
        <option value=\"healthy\">Healthy</option>
        <option value=\"warning\">Warning</option>
        <option value=\"unhealthy\">Unhealthy</option>
        <option value=\"unknown\">Unknown</option>
      </select>
    </div>

    <table>
      <thead>
        <tr>
          <th>Node</th>
          <th>Status</th>
          <th>Timestamp</th>
          <th>Source</th>
        </tr>
      </thead>
      <tbody id=\"rows\"></tbody>
    </table>
  </div>

<script>
const refreshMs = {max(1000, refresh_seconds * 1000)};
let cache = [];

function esc(s) {{
  return String(s ?? '').replace(/[&<>\"]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}}[c]));
}}

function pill(status) {{
  return `<span class=\"pill ${{esc(status)}}\">${{esc(status)}}</span>`;
}}

function renderRows() {{
  const q = document.getElementById('search').value.trim().toLowerCase();
  const status = document.getElementById('status').value;
  const rows = document.getElementById('rows');

  const filtered = cache.filter(n => {{
    const matchesQ = !q || (n.node || '').toLowerCase().includes(q);
    const matchesS = status === 'all' || n.status === status;
    return matchesQ && matchesS;
  }});

  rows.innerHTML = filtered.map(n => `
    <tr>
      <td>${{esc(n.node)}}</td>
      <td>${{pill(n.status)}}</td>
      <td><span class=\"small\">${{esc(n.timestamp || '')}}</span></td>
      <td><span class=\"small\">${{esc(n.source_file || '')}}</span></td>
    </tr>
  `).join('');
}}

async function refresh() {{
  try {{
    const res = await fetch('/api/health');
    const data = await res.json();

    cache = data.nodes || [];

    document.getElementById('total').textContent = data.summary?.total_nodes ?? 0;
    document.getElementById('healthy').textContent = data.summary?.healthy ?? 0;
    document.getElementById('warning').textContent = data.summary?.warning ?? 0;
    document.getElementById('unhealthy').textContent = data.summary?.unhealthy ?? 0;
    document.getElementById('unknown').textContent = data.summary?.unknown ?? 0;

    const fileCount = (data.source_files || []).length;
    document.getElementById('meta').textContent = `Last refresh: ${{data.last_refresh_utc || ''}} | Sources: ${{fileCount}}`;

    renderRows();
  }} catch (err) {{
    document.getElementById('meta').textContent = `Failed to refresh: ${{err}}`;
  }}
}}

document.getElementById('search').addEventListener('input', renderRows);
document.getElementById('status').addEventListener('change', renderRows);
refresh();
setInterval(refresh, refreshMs);
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    store: HealthDataStore = None  # type: ignore
    refresh_seconds: int = 10

    def _send_json(self, data: Dict, status: int = 200) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_html(self, html: str, status: int = 200) -> None:
        payload = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/index.html"}:
            self._send_html(build_index_html(self.refresh_seconds))
            return

        if self.path == "/api/health":
            self.store.refresh()
            self._send_json(self.store.snapshot())
            return

        if self.path == "/api/ping":
            self._send_json({"ok": True, "time": iso_now()})
            return

        self._send_json({"error": "not found"}, status=404)


def run_server(data_dir: Path, host: str, port: int, refresh_seconds: int) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    store = HealthDataStore(data_dir)
    store.refresh()

    DashboardHandler.store = store
    DashboardHandler.refresh_seconds = refresh_seconds

    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"Dashboard started on http://{host}:{port}")
    print(f"Reading JSON data from: {data_dir}")
    print("Endpoints: /, /api/health, /api/ping")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a node-health dashboard from JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default="system_monitoring/data",
        help="Directory containing node health JSON files.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host bind address.")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port.")
    parser.add_argument(
        "--refresh-seconds",
        type=int,
        default=10,
        help="UI/API refresh interval in seconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_server(
        data_dir=Path(args.data_dir).resolve(),
        host=args.host,
        port=args.port,
        refresh_seconds=max(1, args.refresh_seconds),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
