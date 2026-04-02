"""
Radix Tree Visualizer for SGLang Workers

Visualizes the radix tree structure on each SGLang worker.
Since SGLang doesn't expose a direct API for radix tree visualization,
this module provides utilities to:
1. Estimate tree structure from cache statistics
2. Generate visual representations of cache state
3. Export tree data for external visualization tools
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


def _get_no_proxy_session() -> requests.Session:
    """Create a requests session that bypasses proxy for internal hosts."""
    session = requests.Session()
    # Disable proxy for this session
    session.trust_env = False
    return session


# Global session for all requests
_session = _get_no_proxy_session()


@dataclass
class CacheSnapshot:
    """Snapshot of cache state from a worker."""
    worker_url: str
    timestamp: float
    cache_hit_rate: float
    token_usage: float
    num_cached_tokens: int
    num_total_tokens: int
    num_running_reqs: int
    num_waiting_reqs: int
    gen_throughput: float
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TreeVisualization:
    """Data structure for tree visualization."""
    worker_url: str
    snapshots: List[CacheSnapshot] = field(default_factory=list)
    request_history: List[Dict[str, Any]] = field(default_factory=list)


class RadixTreeVisualizer:
    """Visualizer for radix tree cache state across workers."""

    def __init__(self, worker_urls: List[str], output_dir: str = "logs/visualizations"):
        self.worker_urls = worker_urls
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.visualizations: Dict[str, TreeVisualization] = {
            url: TreeVisualization(worker_url=url) for url in worker_urls
        }

    def _get_worker_prometheus_metrics(self, worker_url: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        try:
            resp = _session.get(f"{worker_url}/metrics", timeout=10)
            if resp.status_code != 200:
                return metrics
            for line in resp.text.split("\n"):
                if line.startswith("#") or not line.strip():
                    continue
                if "prompt_tokens_total" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        metrics["prompt_tokens_total"] = float(parts[-1])
                elif "cached_tokens_total" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        metrics["cached_tokens_total"] = float(parts[-1])
        except Exception:
            return metrics
        return metrics

    def _cache_hit_rate_from_prometheus(self, prom: Dict[str, float]) -> float:
        prompt_tokens = prom.get("prompt_tokens_total", 0.0)
        cached_tokens = prom.get("cached_tokens_total", 0.0)
        total = prompt_tokens + cached_tokens
        return (cached_tokens / total) if total > 0 else 0.0

    def capture_snapshot(self, worker_url: str) -> Optional[CacheSnapshot]:
        """Capture current cache state from a worker."""
        try:
            prom = self._get_worker_prometheus_metrics(worker_url)
            resp = _session.get(f"{worker_url}/server_info", timeout=10)
            if resp.status_code != 200:
                return None

            data = resp.json()

            state: Dict[str, Any] = data if isinstance(data, dict) else {}

            # Extract cache-related metrics
            cache_hit_rate = self._cache_hit_rate_from_prometheus(prom)
            token_usage = float(state.get("token_usage", 0.0) or 0.0)
            max_total_num_tokens = int(state.get("max_total_num_tokens", 0) or 0)
            num_cached_tokens = int(token_usage * max_total_num_tokens) if max_total_num_tokens > 0 else 0
            snapshot = CacheSnapshot(
                worker_url=worker_url,
                timestamp=time.time(),
                cache_hit_rate=cache_hit_rate,
                token_usage=token_usage,
                num_cached_tokens=num_cached_tokens,
                num_total_tokens=max_total_num_tokens,
                num_running_reqs=state.get("num_running_reqs", 0),
                num_waiting_reqs=state.get("num_waiting_reqs", 0),
                gen_throughput=state.get("gen_throughput", 0.0),
                raw_data=state,
            )

            self.visualizations[worker_url].snapshots.append(snapshot)
            return snapshot

        except Exception as e:
            print(f"Error capturing snapshot from {worker_url}: {e}")
            return None

    def capture_all_snapshots(self) -> Dict[str, CacheSnapshot]:
        """Capture snapshots from all workers."""
        snapshots = {}
        for url in self.worker_urls:
            snapshot = self.capture_snapshot(url)
            if snapshot:
                snapshots[url] = snapshot
        return snapshots

    def record_request(
        self,
        worker_url: str,
        trace_id: str,
        agent_id: str,
        input_tokens: int,
        cache_hit_tokens: int,
        output_tokens: int,
    ):
        """Record a request for visualization."""
        if worker_url in self.visualizations:
            self.visualizations[worker_url].request_history.append({
                "timestamp": time.time(),
                "trace_id": trace_id,
                "agent_id": agent_id,
                "input_tokens": input_tokens,
                "cache_hit_tokens": cache_hit_tokens,
                "output_tokens": output_tokens,
            })

    def generate_ascii_tree(self, worker_url: str) -> str:
        """Generate ASCII representation of cache state."""
        if worker_url not in self.visualizations:
            return f"No data for {worker_url}"

        viz = self.visualizations[worker_url]
        if not viz.snapshots:
            return f"No snapshots for {worker_url}"

        latest = viz.snapshots[-1]

        lines = []
        lines.append(f"╔{'═' * 60}╗")
        lines.append(f"║ Worker: {worker_url:<50} ║")
        lines.append(f"╠{'═' * 60}╣")
        lines.append(f"║ Cache Hit Rate: {latest.cache_hit_rate:>6.2%}                              ║")
        lines.append(f"║ Token Usage:    {latest.token_usage:>6.2%}                              ║")
        lines.append(f"║ Cached Tokens:  {latest.num_cached_tokens:>10,}                        ║")
        lines.append(f"║ Total Capacity: {latest.num_total_tokens:>10,}                        ║")
        lines.append(f"║ Running Reqs:   {latest.num_running_reqs:>10}                        ║")
        lines.append(f"║ Waiting Reqs:   {latest.num_waiting_reqs:>10}                        ║")
        lines.append(f"║ Throughput:     {latest.gen_throughput:>10.2f} tok/s                 ║")
        lines.append(f"╠{'═' * 60}╣")

        # Show request distribution by trace
        trace_counts: Dict[str, int] = {}
        for req in viz.request_history:
            tid = req["trace_id"]
            trace_counts[tid] = trace_counts.get(tid, 0) + 1

        lines.append(f"║ Request Distribution by Trace:                             ║")
        for tid, count in sorted(trace_counts.items()):
            bar_len = min(count * 2, 40)
            bar = "█" * bar_len
            lines.append(f"║   {tid:<12}: {bar:<40} ({count:>3}) ║")

        # Show request distribution by agent
        agent_counts: Dict[str, int] = {}
        for req in viz.request_history:
            aid = req["agent_id"]
            agent_counts[aid] = agent_counts.get(aid, 0) + 1

        lines.append(f"╠{'═' * 60}╣")
        lines.append(f"║ Request Distribution by Agent:                             ║")
        for aid, count in sorted(agent_counts.items()):
            bar_len = min(count * 2, 40)
            bar = "█" * bar_len
            lines.append(f"║   {aid:<12}: {bar:<40} ({count:>3}) ║")

        lines.append(f"╚{'═' * 60}╝")

        return "\n".join(lines)

    def generate_cache_usage_bar(self, worker_url: str, width: int = 50) -> str:
        """Generate a visual bar showing cache usage."""
        if worker_url not in self.visualizations:
            return ""

        viz = self.visualizations[worker_url]
        if not viz.snapshots:
            return ""

        latest = viz.snapshots[-1]
        usage = latest.token_usage
        filled = int(usage * width)
        empty = width - filled

        bar = f"[{'█' * filled}{'░' * empty}] {usage:.1%}"
        return bar

    def print_all_workers(self):
        """Print visualization for all workers."""
        print("\n" + "=" * 80)
        print("RADIX TREE CACHE VISUALIZATION")
        print("=" * 80)

        for url in self.worker_urls:
            print(self.generate_ascii_tree(url))
            print()

    def export_to_json(self, filename: str = None) -> str:
        """Export visualization data to JSON."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"tree_viz_{timestamp}.json"

        filepath = self.output_dir / filename

        export_data = {}
        for url, viz in self.visualizations.items():
            export_data[url] = {
                "snapshots": [
                    {
                        "timestamp": s.timestamp,
                        "cache_hit_rate": s.cache_hit_rate,
                        "token_usage": s.token_usage,
                        "num_cached_tokens": s.num_cached_tokens,
                        "num_total_tokens": s.num_total_tokens,
                        "num_running_reqs": s.num_running_reqs,
                        "num_waiting_reqs": s.num_waiting_reqs,
                        "gen_throughput": s.gen_throughput,
                    }
                    for s in viz.snapshots
                ],
                "request_history": viz.request_history,
            }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"Exported visualization data to: {filepath}")
        return str(filepath)

    def generate_html_report(self, filename: str = None) -> str:
        """Generate an HTML report with interactive visualizations."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"tree_report_{timestamp}.html"

        filepath = self.output_dir / filename

        # Prepare data for charts
        worker_data = []
        for url, viz in self.visualizations.items():
            if viz.snapshots:
                latest = viz.snapshots[-1]
                worker_data.append({
                    "url": url,
                    "cache_hit_rate": latest.cache_hit_rate,
                    "token_usage": latest.token_usage,
                    "num_requests": len(viz.request_history),
                })

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Radix Tree Cache Visualization</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .chart-container {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .worker-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2196F3;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
        }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background-color: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            transition: width 0.3s ease;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Radix Tree Cache Visualization</h1>
        <p style="text-align: center; color: #666;">Generated at: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>

        <div class="chart-container">
            <h2>Cache Hit Rate by Worker</h2>
            <canvas id="cacheHitChart"></canvas>
        </div>

        <div class="chart-container">
            <h2>Token Usage by Worker</h2>
            <canvas id="tokenUsageChart"></canvas>
        </div>

        <div class="chart-container">
            <h2>Request Distribution</h2>
            <canvas id="requestChart"></canvas>
        </div>

        <h2>Worker Details</h2>
"""

        for url, viz in self.visualizations.items():
            if viz.snapshots:
                latest = viz.snapshots[-1]
                html_content += f"""
        <div class="worker-card">
            <h3>{url}</h3>
            <div class="metric">
                <div class="metric-value">{latest.cache_hit_rate:.1%}</div>
                <div class="metric-label">Cache Hit Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{latest.token_usage:.1%}</div>
                <div class="metric-label">Token Usage</div>
            </div>
            <div class="metric">
                <div class="metric-value">{len(viz.request_history)}</div>
                <div class="metric-label">Total Requests</div>
            </div>
            <div class="metric">
                <div class="metric-value">{latest.gen_throughput:.1f}</div>
                <div class="metric-label">Throughput (tok/s)</div>
            </div>
            <div style="margin-top: 15px;">
                <label>Cache Usage:</label>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {latest.token_usage * 100}%"></div>
                </div>
            </div>
        </div>
"""

        # Add Chart.js scripts
        labels = [d["url"].split("//")[1] for d in worker_data]
        cache_rates = [d["cache_hit_rate"] * 100 for d in worker_data]
        token_usages = [d["token_usage"] * 100 for d in worker_data]
        request_counts = [d["num_requests"] for d in worker_data]

        html_content += f"""
    </div>

    <script>
        // Cache Hit Rate Chart
        new Chart(document.getElementById('cacheHitChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(labels)},
                datasets: [{{
                    label: 'Cache Hit Rate (%)',
                    data: {json.dumps(cache_rates)},
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100
                    }}
                }}
            }}
        }});

        // Token Usage Chart
        new Chart(document.getElementById('tokenUsageChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(labels)},
                datasets: [{{
                    label: 'Token Usage (%)',
                    data: {json.dumps(token_usages)},
                    backgroundColor: 'rgba(75, 192, 192, 0.6)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100
                    }}
                }}
            }}
        }});

        // Request Distribution Chart
        new Chart(document.getElementById('requestChart'), {{
            type: 'pie',
            data: {{
                labels: {json.dumps(labels)},
                datasets: [{{
                    data: {json.dumps(request_counts)},
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.6)',
                        'rgba(54, 162, 235, 0.6)',
                        'rgba(255, 206, 86, 0.6)',
                        'rgba(75, 192, 192, 0.6)',
                        'rgba(153, 102, 255, 0.6)'
                    ]
                }}]
            }}
        }});
    </script>
</body>
</html>
"""

        with open(filepath, "w") as f:
            f.write(html_content)

        print(f"Generated HTML report: {filepath}")
        return str(filepath)

    def reset(self):
        """Reset all visualization data."""
        self.visualizations = {
            url: TreeVisualization(worker_url=url) for url in self.worker_urls
        }


if __name__ == "__main__":
    # Test the visualizer
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-urls-file", type=str, default="logs/worker_urls.txt")
    args = parser.parse_args()

    # Load worker URLs
    with open(args.worker_urls_file, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(urls)} worker URLs")

    visualizer = RadixTreeVisualizer(urls)

    print("\nCapturing snapshots...")
    snapshots = visualizer.capture_all_snapshots()

    print("\nGenerating visualizations...")
    visualizer.print_all_workers()

    print("\nExporting data...")
    visualizer.export_to_json()
    visualizer.generate_html_report()
