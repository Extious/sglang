"""
Metrics Collector for SGLang Multi-Agent Experiments

Collects cache hit rate, TTFT, and other metrics from SGLang workers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
class RequestMetrics:
    """Metrics for a single request."""
    request_id: str
    trace_id: str
    agent_id: str
    worker_url: str
    ttft: Optional[float] = None  # Time to first token (seconds)
    total_latency: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit_tokens: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class WorkerMetrics:
    """Aggregated metrics for a single worker."""
    worker_url: str
    total_requests: int = 0
    total_input_tokens: int = 0
    total_cache_hit_tokens: int = 0
    total_output_tokens: int = 0
    ttft_values: List[float] = field(default_factory=list)
    latency_values: List[float] = field(default_factory=list)

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.total_input_tokens + self.total_cache_hit_tokens
        if total == 0:
            return 0.0
        return self.total_cache_hit_tokens / total

    @property
    def avg_ttft(self) -> float:
        """Calculate average TTFT."""
        if not self.ttft_values:
            return 0.0
        return sum(self.ttft_values) / len(self.ttft_values)

    @property
    def avg_latency(self) -> float:
        """Calculate average latency."""
        if not self.latency_values:
            return 0.0
        return sum(self.latency_values) / len(self.latency_values)


class MetricsCollector:
    """Collects and aggregates metrics from SGLang workers."""

    def __init__(self, worker_urls: List[str]):
        self.worker_urls = worker_urls
        self.request_metrics: List[RequestMetrics] = []
        self.worker_metrics: Dict[str, WorkerMetrics] = {
            url: WorkerMetrics(worker_url=url) for url in worker_urls
        }
        self._baseline_stats: Dict[str, Dict] = {}

    def flush_all_caches(self) -> Dict[str, bool]:
        """Flush KV cache on all workers."""
        results = {}
        for url in self.worker_urls:
            try:
                resp = _session.post(f"{url}/flush_cache", timeout=30)
                results[url] = resp.status_code == 200
                if results[url]:
                    print(f"  [OK] Flushed cache on {url}")
                else:
                    print(f"  [FAIL] Failed to flush cache on {url}: {resp.text[:200]}")
            except Exception as e:
                results[url] = False
                print(f"  [ERROR] Error flushing cache on {url}: {e}")
        return results

    def get_worker_server_info(self, worker_url: str) -> Optional[Dict[str, Any]]:
        """Get server info from a worker."""
        try:
            resp = _session.get(f"{worker_url}/server_info", timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"  [ERROR] Error getting server info from {worker_url}: {e}")
        return None

    def get_worker_prometheus_metrics(self, worker_url: str) -> Optional[Dict[str, float]]:
        """Get Prometheus metrics from a worker for accurate cache hit rate."""
        try:
            resp = _session.get(f"{worker_url}/metrics", timeout=10)
            if resp.status_code == 200:
                metrics = {}
                for line in resp.text.split('\n'):
                    if line.startswith('#') or not line.strip():
                        continue
                    # Parse metric line: metric_name{labels} value
                    if 'prompt_tokens_total' in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            metrics['prompt_tokens_total'] = float(parts[-1])
                    elif 'cached_tokens_total' in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            metrics['cached_tokens_total'] = float(parts[-1])
                    elif 'generation_tokens_total' in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            metrics['generation_tokens_total'] = float(parts[-1])
                    elif 'time_to_first_token_seconds_sum' in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            metrics['ttft_sum'] = float(parts[-1])
                    elif 'time_to_first_token_seconds_count' in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            metrics['ttft_count'] = float(parts[-1])
                return metrics
        except Exception as e:
            print(f"  [ERROR] Error getting prometheus metrics from {worker_url}: {e}")
        return None

    def get_all_prometheus_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get Prometheus metrics from all workers."""
        results = {}
        for url in self.worker_urls:
            metrics = self.get_worker_prometheus_metrics(url)
            if metrics:
                results[url] = metrics
        return results

    def calculate_cache_hit_rate_from_prometheus(self, metrics: Dict[str, float]) -> float:
        """Calculate cache hit rate from Prometheus metrics."""
        prompt_tokens = metrics.get('prompt_tokens_total', 0)
        cached_tokens = metrics.get('cached_tokens_total', 0)
        total = prompt_tokens + cached_tokens
        if total > 0:
            return cached_tokens / total
        return 0.0

    def get_all_server_info(self) -> Dict[str, Dict[str, Any]]:
        """Get server info from all workers."""
        results = {}
        for url in self.worker_urls:
            info = self.get_worker_server_info(url)
            if info:
                results[url] = info
        return results

    def capture_baseline_stats(self):
        """Capture baseline stats before running experiments."""
        print("Capturing baseline stats from all workers...")
        self._baseline_stats = {}
        for url in self.worker_urls:
            # Use Prometheus metrics for accurate baseline
            prom_metrics = self.get_worker_prometheus_metrics(url)
            if prom_metrics:
                self._baseline_stats[url] = {
                    "prompt_tokens_total": prom_metrics.get("prompt_tokens_total", 0),
                    "cached_tokens_total": prom_metrics.get("cached_tokens_total", 0),
                    "generation_tokens_total": prom_metrics.get("generation_tokens_total", 0),
                    "ttft_sum": prom_metrics.get("ttft_sum", 0),
                    "ttft_count": prom_metrics.get("ttft_count", 0),
                }
                print(f"  [OK] Captured baseline for {url}")
            else:
                # Fallback to server_info
                info = self.get_worker_server_info(url)
                if info:
                    if isinstance(info, list) and len(info) > 0:
                        state = info[0]
                        self._baseline_stats[url] = {
                            "cache_hit_rate": state.get("cache_hit_rate", 0.0),
                            "token_usage": state.get("token_usage", 0.0),
                        }
                    else:
                        self._baseline_stats[url] = info
                    print(f"  [OK] Captured baseline for {url} (fallback)")

    def get_current_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get current stats from all workers using Prometheus metrics."""
        stats = {}
        for url in self.worker_urls:
            # Use Prometheus metrics for accurate cache hit rate
            prom_metrics = self.get_worker_prometheus_metrics(url)
            baseline = self._baseline_stats.get(url, {})

            if prom_metrics:
                # Calculate delta from baseline
                prompt_tokens = prom_metrics.get("prompt_tokens_total", 0) - baseline.get("prompt_tokens_total", 0)
                cached_tokens = prom_metrics.get("cached_tokens_total", 0) - baseline.get("cached_tokens_total", 0)
                total = prompt_tokens + cached_tokens

                cache_hit_rate = cached_tokens / total if total > 0 else 0.0

                # Calculate average TTFT from Prometheus
                ttft_sum_delta = prom_metrics.get("ttft_sum", 0) - baseline.get("ttft_sum", 0)
                ttft_count_delta = prom_metrics.get("ttft_count", 0) - baseline.get("ttft_count", 0)
                avg_ttft = ttft_sum_delta / ttft_count_delta if ttft_count_delta > 0 else 0.0

                stats[url] = {
                    "cache_hit_rate": cache_hit_rate,
                    "prompt_tokens": prompt_tokens,
                    "cached_tokens": cached_tokens,
                    "avg_ttft_prometheus": avg_ttft,
                    "token_usage": 0.0,  # Will get from server_info if needed
                }

            # Also get server_info for other metrics
            info = self.get_worker_server_info(url)
            if info:
                if isinstance(info, list) and len(info) > 0:
                    state = info[0]
                    if url not in stats:
                        stats[url] = {}
                    stats[url]["token_usage"] = state.get("token_usage", 0.0)
                    stats[url]["num_running_reqs"] = state.get("num_running_reqs", 0)
                    stats[url]["num_waiting_reqs"] = state.get("num_waiting_reqs", 0)
                    stats[url]["gen_throughput"] = state.get("gen_throughput", 0.0)

        return stats

    def record_request(self, metrics: RequestMetrics):
        """Record metrics for a single request."""
        self.request_metrics.append(metrics)

        # Update worker aggregates
        if metrics.worker_url in self.worker_metrics:
            wm = self.worker_metrics[metrics.worker_url]
            wm.total_requests += 1
            wm.total_input_tokens += metrics.input_tokens
            wm.total_cache_hit_tokens += metrics.cache_hit_tokens
            wm.total_output_tokens += metrics.output_tokens
            if metrics.ttft is not None:
                wm.ttft_values.append(metrics.ttft)
            wm.latency_values.append(metrics.total_latency)

    def reset_metrics(self):
        """Reset all collected metrics."""
        self.request_metrics = []
        self.worker_metrics = {
            url: WorkerMetrics(worker_url=url) for url in self.worker_urls
        }
        self._baseline_stats = {}

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics."""
        worker_summaries = {}
        for url, wm in self.worker_metrics.items():
            worker_summaries[url] = {
                "total_requests": wm.total_requests,
                "cache_hit_rate": wm.cache_hit_rate,
                "avg_ttft": wm.avg_ttft,
                "avg_latency": wm.avg_latency,
                "total_input_tokens": wm.total_input_tokens,
                "total_cache_hit_tokens": wm.total_cache_hit_tokens,
                "total_output_tokens": wm.total_output_tokens,
            }

        # Overall stats
        total_requests = sum(wm.total_requests for wm in self.worker_metrics.values())
        total_input = sum(wm.total_input_tokens for wm in self.worker_metrics.values())
        total_hit = sum(wm.total_cache_hit_tokens for wm in self.worker_metrics.values())
        all_ttft = [t for wm in self.worker_metrics.values() for t in wm.ttft_values]
        all_latency = [l for wm in self.worker_metrics.values() for l in wm.latency_values]

        return {
            "total_requests": total_requests,
            "overall_cache_hit_rate": total_hit / (total_input + total_hit) if (total_input + total_hit) > 0 else 0.0,
            "avg_ttft": sum(all_ttft) / len(all_ttft) if all_ttft else 0.0,
            "avg_latency": sum(all_latency) / len(all_latency) if all_latency else 0.0,
            "worker_summaries": worker_summaries,
        }

    def print_summary(self):
        """Print a formatted summary of metrics."""
        summary = self.get_summary()

        print("\n" + "=" * 80)
        print("METRICS SUMMARY")
        print("=" * 80)
        print(f"Total Requests: {summary['total_requests']}")
        print(f"Overall Cache Hit Rate: {summary['overall_cache_hit_rate']:.4f}")
        print(f"Average TTFT: {summary['avg_ttft']:.4f}s")
        print(f"Average Latency: {summary['avg_latency']:.4f}s")

        print("\n" + "-" * 80)
        print("PER-WORKER METRICS")
        print("-" * 80)

        for url, ws in summary["worker_summaries"].items():
            print(f"\n{url}:")
            print(f"  Requests: {ws['total_requests']}")
            print(f"  Cache Hit Rate: {ws['cache_hit_rate']:.4f}")
            print(f"  Avg TTFT: {ws['avg_ttft']:.4f}s")
            print(f"  Avg Latency: {ws['avg_latency']:.4f}s")
            print(f"  Input Tokens: {ws['total_input_tokens']}")
            print(f"  Cache Hit Tokens: {ws['total_cache_hit_tokens']}")


def load_worker_urls(file_path: str) -> List[str]:
    """Load worker URLs from file."""
    with open(file_path, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    return urls


if __name__ == "__main__":
    # Test the metrics collector
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-urls-file", type=str,
                        default="logs/worker_urls.txt")
    args = parser.parse_args()

    urls = load_worker_urls(args.worker_urls_file)
    print(f"Loaded {len(urls)} worker URLs")

    collector = MetricsCollector(urls)

    print("\nGetting current server info...")
    stats = collector.get_current_stats()
    for url, info in stats.items():
        print(f"\n{url}:")
        for k, v in info.items():
            print(f"  {k}: {v}")
