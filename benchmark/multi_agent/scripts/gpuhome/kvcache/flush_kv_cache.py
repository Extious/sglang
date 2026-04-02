"""
CLI script to flush KV cache on SGLang workers and (optionally) the
router's radix tree.

Usage:
    python script/kvcache/flush_kv_cache.py --worker-urls-file logs/worker_urls.txt
    python script/kvcache/flush_kv_cache.py --worker-urls http://gpu18:8000 http://gpu19:8000
    python script/kvcache/flush_kv_cache.py --worker-urls-file logs/worker_urls.txt \
        --router-url http://gpu01:30000
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List

import requests


def _get_no_proxy_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


_session = _get_no_proxy_session()


def _load_worker_urls(path: str) -> List[str]:
    urls: List[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def _flush_worker(url: str, timeout_s: int, retries: int, retry_delay_s: float) -> bool:
    for attempt in range(retries + 1):
        try:
            resp = _session.post(f"{url}/flush_cache", timeout=timeout_s)
            if resp.status_code == 200:
                return True
            err_text = (resp.text or "").strip()
            if err_text:
                print(f"[FAIL] {url} status={resp.status_code} msg={err_text[:200]}")
            else:
                print(f"[FAIL] {url} status={resp.status_code}")
        except Exception as e:
            print(f"[ERROR] {url} error={e}")
        if attempt < retries:
            time.sleep(retry_delay_s)
    return False


def _flush_router_tree(url: str, timeout_s: int, retries: int, retry_delay_s: float) -> bool:
    for attempt in range(retries + 1):
        try:
            resp = _session.post(f"{url}/flush_tree", timeout=timeout_s)
            if resp.status_code == 200:
                return True
            err_text = (resp.text or "").strip()
            if err_text:
                print(f"[FAIL] router {url} status={resp.status_code} msg={err_text[:200]}")
            else:
                print(f"[FAIL] router {url} status={resp.status_code}")
        except Exception as e:
            print(f"[ERROR] router {url} error={e}")
        if attempt < retries:
            time.sleep(retry_delay_s)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Flush KV cache on SGLang workers."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--worker-urls-file",
        type=str,
        help="Path to a text file with one worker URL per line.",
    )
    group.add_argument(
        "--worker-urls",
        nargs="+",
        type=str,
        help="Worker URLs directly, e.g. http://gpu18:8000",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds (default: 30).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Number of retries per worker (default: 0).",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="Seconds to wait between retries (default: 1.0).",
    )
    parser.add_argument(
        "--router-url",
        type=str,
        default=None,
        help="Router URL to flush its radix tree, e.g. http://gpu01:30000",
    )
    parser.add_argument(
        "--router-url-file",
        type=str,
        default=None,
        help="Path to a text file containing the router URL.",
    )

    args = parser.parse_args()

    if args.worker_urls_file:
        worker_urls = _load_worker_urls(args.worker_urls_file)
    else:
        worker_urls = args.worker_urls

    if not worker_urls:
        print("Error: no worker URLs provided.", file=sys.stderr)
        return 1

    print(f"Workers: {worker_urls}")
    all_ok = True
    for url in worker_urls:
        ok = _flush_worker(url, args.timeout, args.retries, args.retry_delay)
        if ok:
            print(f"[OK] {url} cache flushed")
        else:
            all_ok = False

    if not all_ok:
        print("One or more workers failed to flush.")
        return 1

    print("All workers flushed successfully.")

    router_url = args.router_url
    if not router_url and args.router_url_file:
        try:
            urls = _load_worker_urls(args.router_url_file)
            if urls:
                router_url = urls[0]
        except FileNotFoundError:
            print(f"WARNING: router-url-file not found: {args.router_url_file}",
                  file=sys.stderr)

    if router_url:
        print(f"Flushing router radix tree: {router_url}")
        if _flush_router_tree(router_url, args.timeout, args.retries, args.retry_delay):
            print(f"[OK] router {router_url} tree flushed")
        else:
            print("Router tree flush failed.")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
