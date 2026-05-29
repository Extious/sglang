from __future__ import annotations

from pathlib import Path

from .backup_cache_hits import plot_cache_hits
from .trace_log_profile import plot_runtime_profile


def write_default_plots(output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    figures_dir = output_path / "figures"
    plot_cache_hits(
        output_path / "cache_hits.csv",
        figures_dir / "backup_cache_hit_profile.png",
    )
    plot_runtime_profile(
        output_path / "request_detail.csv",
        figures_dir / "trace_log_profile.png",
    )
