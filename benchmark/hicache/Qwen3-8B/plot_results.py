#!/usr/bin/env python3
import json
import os
import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.labelsize'] = 16
matplotlib.rcParams['axes.titlesize'] = 18
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['legend.fontsize'] = 14

def load_results(results_dir):
    results = {}
    jsonl_files = glob.glob(os.path.join(results_dir, "*.jsonl"))

    for filepath in jsonl_files:
        filename = os.path.basename(filepath)
        config_name = filename.split("_")[1] if "_" in filename else filename.replace(".jsonl", "")

        data = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    if 'summary' in obj:
                        data.append(obj['summary'])

        if data:
            results[config_name] = data

    return results

def plot_metrics(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    metrics_config = [
        {
            'key': 'throughput',
            'title': 'Throughput vs Request Rate',
            'ylabel': 'Throughput (req/s)',
            'xlabel': 'Request Rate (req/s)'
        },
        {
            'key': 'average_latency',
            'title': 'Average Latency vs Request Rate',
            'ylabel': 'Average Latency (s)',
            'xlabel': 'Request Rate (req/s)'
        },
        {
            'key': 'average_ttft',
            'title': 'Average TTFT vs Request Rate',
            'ylabel': 'Average TTFT (s)',
            'xlabel': 'Request Rate (req/s)'
        },
        {
            'key': 'cache_hit_rate',
            'title': 'Cache Hit Rate vs Request Rate',
            'ylabel': 'Cache Hit Rate',
            'xlabel': 'Request Rate (req/s)'
        },
        {
            'key': 'p90_latency',
            'title': 'P90 Latency vs Request Rate',
            'ylabel': 'P90 Latency (s)',
            'xlabel': 'Request Rate (req/s)'
        },
        {
            'key': 'output_token_throughput',
            'title': 'Output Token Throughput vs Request Rate',
            'ylabel': 'Output Token Throughput (tokens/s)',
            'xlabel': 'Request Rate (req/s)'
        }
    ]

    for metric in metrics_config:
        plt.figure(figsize=(10, 6))

        for config_name, data in results.items():
            request_rates = [d['request_rate'] for d in data]
            values = [d[metric['key']] for d in data]

            plt.plot(request_rates, values, marker='o', label=config_name, linewidth=2, markersize=6)

        plt.xlabel(metric['xlabel'], fontsize=16)
        plt.ylabel(metric['ylabel'], fontsize=16)
        plt.title(metric['title'], fontsize=18, fontweight='bold')
        plt.legend(fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_file = os.path.join(output_dir, f"{metric['key']}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_file}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    output_dir = os.path.join(script_dir, "plots")

    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return

    print(f"Loading results from: {results_dir}")
    results = load_results(results_dir)

    if not results:
        print("No results found!")
        return

    print(f"Found {len(results)} configurations:")
    for config_name in results.keys():
        print(f"  - {config_name}")

    print(f"\nGenerating plots...")
    plot_metrics(results, output_dir)
    print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
