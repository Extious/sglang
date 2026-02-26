"""CLI script to capture and visualize radix trees from SGLang workers.

Usage:
    python script/visualize_radix_tree.py --worker-urls-file logs/worker_urls.txt
    python script/visualize_radix_tree.py --worker-urls http://gpu18:8000 http://gpu19:8000
    python script/visualize_radix_tree.py --worker-urls-file logs/worker_urls.txt --strategy my_exp --max-nodes 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the benchmark/multi_agent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.server.prompt_tree_visualizer import PromptTreeVisualizer


def parse_worker_urls_file(path: str) -> list[str]:
    urls = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def main():
    parser = argparse.ArgumentParser(
        description="Capture and visualize radix trees from SGLang workers."
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
        "--output-dir",
        type=str,
        default="results/radix_tree",
        help="Directory to write output files (default: results/radix_tree).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="default",
        help="Strategy name label for the capture (default: 'default').",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=2000,
        help="Maximum number of nodes to fetch per tree (default: 2000).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=64,
        help="Maximum tree depth (default: 64).",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Exclude decoded text from the response.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Only export JSON, skip HTML report.",
    )
    parser.add_argument(
        "--no-strict-sync",
        action="store_true",
        help="Do not wait for HiCache async events before dumping radix tree.",
    )
    parser.add_argument(
        "--sync-timeout-s",
        type=float,
        default=5.0,
        help="Max seconds to wait for strict HiCache sync (default: 5.0).",
    )

    args = parser.parse_args()

    if args.worker_urls_file:
        worker_urls = parse_worker_urls_file(args.worker_urls_file)
    else:
        worker_urls = args.worker_urls

    if not worker_urls:
        print("Error: no worker URLs provided.", file=sys.stderr)
        sys.exit(1)

    print(f"Workers: {worker_urls}")
    print(f"Output dir: {args.output_dir}")
    print(
        f"Strict sync: {str(not args.no_strict_sync).lower()} (timeout={args.sync_timeout_s}s)"
    )

    viz = PromptTreeVisualizer(
        worker_urls=worker_urls,
        output_dir=args.output_dir,
    )

    include_text = not args.no_text
    captured = viz.capture_all(
        strategy_name=args.strategy,
        include_text=include_text,
        max_nodes=args.max_nodes,
        max_depth=args.max_depth,
        strict_sync=not args.no_strict_sync,
        sync_timeout_s=args.sync_timeout_s,
    )

    # Print summary
    for url, info in captured["workers"].items():
        if info["success"]:
            total_nodes = sum(
                len(t.get("nodes", [])) for t in info["dp_trees"]
            )
            print(f"  {url}: OK, {len(info['dp_trees'])} dp rank(s), {total_nodes} total nodes")
        else:
            print(f"  {url}: FAILED - {info['error']}")

    json_path = viz.export_to_json(captured, "radix_tree.json")
    print(f"JSON saved: {json_path}")

    if not args.json_only:
        html_path = viz.generate_html_report(
            captured,
            filename="radix_tree.html",
            title=f"Radix Tree - {args.strategy}",
        )
        print(f"HTML saved: {html_path}")


if __name__ == "__main__":
    main()
