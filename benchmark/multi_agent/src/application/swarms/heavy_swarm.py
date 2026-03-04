import csv
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "swarms")
)

try:
    import litellm
except ModuleNotFoundError:
    litellm = None

try:
    from swarms import HeavySwarm
except ModuleNotFoundError:
    HeavySwarm = None

MODEL_NAME = "openai/Qwen/Qwen3-4B-Instruct-2507"
ROUTER_HOST = os.getenv("ROUTER_HOST", "127.0.0.1")
ROUTER_PORT = os.getenv("ROUTER_PORT", "30000")
TIMING_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "agent_workspace"
    / "timing_reports"
)

AGENT_ORDER = [
    "question",
    "research",
    "analysis",
    "alternatives",
    "verification",
    "synthesis",
]

AGENT_ALIASES = {
    "question-agent": "question",
    "research-agent": "research",
    "analysis-agent": "analysis",
    "alternatives-agent": "alternatives",
    "verification-agent": "verification",
    "synthesis-agent": "synthesis",
    "question": "question",
    "research": "research",
    "analysis": "analysis",
    "alternatives": "alternatives",
    "verification": "verification",
    "synthesis": "synthesis",
}


def _normalize_base_url(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        text = f"http://{ROUTER_HOST}:{ROUTER_PORT}/v1"

    # Handle accidental multi-line / multi-value env content by taking first token.
    text = text.split()[0].split(",")[0].strip()
    if "://" not in text:
        text = f"http://{text}"

    parsed = urlparse(text)
    if not parsed.hostname:
        return f"http://{ROUTER_HOST}:{ROUTER_PORT}/v1"

    scheme = parsed.scheme or "http"
    host = parsed.hostname
    port = parsed.port
    path = parsed.path.rstrip("/")
    if not path or path == "/":
        path = "/v1"

    base = f"{scheme}://{host}"
    if port is not None:
        base = f"{base}:{port}"
    return f"{base}{path}"


def _canonical_agent_name(raw: Optional[str]) -> str:
    text = (raw or "").strip()
    if not text:
        return "unknown"
    normalized = text.lower().replace("_", "-")
    for alias, canonical in AGENT_ALIASES.items():
        if alias in normalized:
            return canonical
    return normalized


def _extract_agent_from_kwargs(kwargs: Dict[str, Any]) -> str:
    metadata = kwargs.get("metadata")
    if isinstance(metadata, dict):
        for key in ("agent_id", "agent", "name"):
            value = metadata.get(key)
            if value:
                return _canonical_agent_name(str(value))

    headers = kwargs.get("extra_headers")
    if isinstance(headers, dict):
        for key in (
            "x-sglang-agent-id",
            "x-agent-id",
            "x-agent",
        ):
            value = headers.get(key)
            if value:
                return _canonical_agent_name(str(value))

    for key in ("user", "safety_identifier"):
        value = kwargs.get(key)
        if value:
            return _canonical_agent_name(str(value))

    return "unknown"


def _to_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _value_from_obj(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _extract_cached_tokens(usage: Any) -> Optional[int]:
    details = _value_from_obj(usage, "prompt_tokens_details")
    if details is None:
        details = _value_from_obj(usage, "input_tokens_details")
    if details is not None:
        cached = _to_int_or_none(
            _value_from_obj(details, "cached_tokens")
        )
        if cached is not None:
            return cached

    for key in (
        "cached_tokens",
        "cache_read_input_tokens",
        "cache_hit_tokens",
    ):
        cached = _to_int_or_none(_value_from_obj(usage, key))
        if cached is not None:
            return cached

    return None


def _extract_usage_tokens(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)

    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    elif usage is None and hasattr(response, "model_dump"):
        try:
            usage = response.model_dump().get("usage")
        except Exception:
            usage = None

    if usage is None:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "cached_tokens": None,
            "cache_hit_ratio": None,
        }

    if isinstance(usage, dict):
        prompt = _to_int_or_none(usage.get("prompt_tokens"))
        completion = _to_int_or_none(
            usage.get("completion_tokens")
        )
        total = _to_int_or_none(usage.get("total_tokens"))
    else:
        prompt = _to_int_or_none(
            getattr(usage, "prompt_tokens", None)
        )
        completion = _to_int_or_none(
            getattr(usage, "completion_tokens", None)
        )
        total = _to_int_or_none(getattr(usage, "total_tokens", None))

    if total is None and prompt is not None and completion is not None:
        total = prompt + completion

    cached = _extract_cached_tokens(usage)
    # Treat None as 0 for cached_tokens (SGLang returns None when cached_tokens=0)
    if cached is None:
        cached = 0

    cache_hit_ratio = None
    if prompt is not None and prompt > 0:
        cache_hit_ratio = float(cached) / float(prompt)

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "cached_tokens": cached,
        "cache_hit_ratio": cache_hit_ratio,
    }


class TimingRecorder:
    def __init__(self, task_index: int, task: str):
        self.task_index = task_index
        self.task = task
        self._lock = threading.Lock()
        self._run_counter: Dict[str, int] = defaultdict(int)
        self._request_records: List[Dict[str, Any]] = []

    def record_request(
        self,
        agent: str,
        latency_seconds: float,
        response: Any = None,
        error: str = "",
    ) -> None:
        canonical_agent = _canonical_agent_name(agent)
        usage = _extract_usage_tokens(response)
        with self._lock:
            self._run_counter[canonical_agent] += 1
            run_index = self._run_counter[canonical_agent]
            self._request_records.append(
                {
                    "task_index": self.task_index,
                    "task": self.task,
                    "agent": canonical_agent,
                    "run_index": run_index,
                    "latency_seconds": float(latency_seconds),
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage[
                        "completion_tokens"
                    ],
                    "total_tokens": usage["total_tokens"],
                    "cached_tokens": usage["cached_tokens"],
                    "cache_hit_ratio": usage["cache_hit_ratio"],
                    "error": error or "",
                }
            )

    def build_task_payload(
        self, task_duration_seconds: float
    ) -> Dict[str, Any]:
        with self._lock:
            request_rows = list(self._request_records)

        per_agent = {}
        for agent_name in AGENT_ORDER + ["unknown"]:
            rows = [
                row for row in request_rows if row["agent"] == agent_name
            ]
            if not rows and agent_name == "unknown":
                continue

            latency_sum = sum(row["latency_seconds"] for row in rows)
            prompt_sum = sum(
                row["prompt_tokens"] or 0 for row in rows
            )
            completion_sum = sum(
                row["completion_tokens"] or 0 for row in rows
            )
            total_sum = sum(row["total_tokens"] or 0 for row in rows)
            has_cache_measurement = any(
                row.get("cached_tokens") is not None for row in rows
            )
            cached_sum = (
                sum(row["cached_tokens"] or 0 for row in rows)
                if has_cache_measurement
                else None
            )
            cache_hit_ratio = None
            if (
                has_cache_measurement
                and cached_sum is not None
                and prompt_sum > 0
            ):
                cache_hit_ratio = float(cached_sum) / float(prompt_sum)

            per_agent[agent_name] = {
                "request_count": len(rows),
                "latency_seconds": latency_sum,
                "prompt_tokens": prompt_sum,
                "completion_tokens": completion_sum,
                "total_tokens": total_sum,
                "cached_tokens": cached_sum,
                "cache_hit_ratio": cache_hit_ratio,
            }

        return {
            "task_index": self.task_index,
            "task": self.task,
            "task_duration_seconds": float(task_duration_seconds),
            "requests": request_rows,
            "per_agent": per_agent,
        }


LLM_BASE_URL = _normalize_base_url(
    os.getenv("LLM_BASE_URL", f"http://{ROUTER_HOST}:{ROUTER_PORT}/v1")
)
LLM_API_KEY = (
    os.getenv("LLM_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("OPENROUTER_API_KEY")
    or "sk-local"
)

# Route local router traffic directly: disable external proxies.
for key in (
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
):
    os.environ[key] = ""

# Must set no_proxy before client initialization.
router_host = urlparse(LLM_BASE_URL).hostname or ""
for key in ("no_proxy", "NO_PROXY"):
    existing = [
        item.strip()
        for item in os.environ.get(key, "").split(",")
        if item.strip()
    ]
    if router_host and router_host not in existing:
        existing.append(router_host)
    for fixed in ("127.0.0.1", "localhost", "0.0.0.0", "::1"):
        if fixed not in existing:
            existing.append(fixed)
    os.environ[key] = ",".join(existing)

if litellm is not None:
    litellm.model_cost[MODEL_NAME] = {
        "max_tokens": 32768,
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
    }
    if MODEL_NAME not in litellm.model_list:
        litellm.model_list.append(MODEL_NAME)

os.environ["OPENAI_BASE_URL"] = LLM_BASE_URL
os.environ["OPENAI_API_KEY"] = LLM_API_KEY

print(f"[heavy_swarm] OPENAI_BASE_URL={LLM_BASE_URL}")
print(f"[heavy_swarm] OPENAI_API_KEY set={bool(LLM_API_KEY)}")

TASK_PROCESSES = max(
    1, int(os.getenv("HEAVY_SWARM_TASK_PROCESSES", "6"))
)
TASK_LIMIT = max(1, int(os.getenv("HEAVY_SWARM_TASK_LIMIT", "80")))
SHOW_DASHBOARD = (
    os.getenv("HEAVY_SWARM_SHOW_DASHBOARD", "0").strip() == "1"
)
ENABLE_TIMING_REPORTS = (
    os.getenv("HEAVY_SWARM_ENABLE_TIMING_REPORTS", "1").strip()
    == "1"
)

tasks = [
    "Analyze the impact of AI on healthcare",
    # --- Economy & Business ---
    "Analyze the impact of AI on retail and e-commerce",
    "Analyze the impact of AI on manufacturing and automation",
    "Analyze the impact of AI on supply chain management",
    "Analyze the impact of AI on marketing and advertising",
    "Analyze the impact of AI on human resources and recruitment",
    # --- Society & Law ---
    "Analyze the impact of AI on the legal system",
    "Analyze the impact of AI on public safety and surveillance",
    "Analyze the impact of AI on privacy and data security",
    "Analyze the impact of AI on journalism and media",
    "Analyze the impact of AI on cybersecurity",
    # --- Science & Environment ---
    "Analyze the impact of AI on environmental sustainability",
    "Analyze the impact of AI on agriculture and food production",
    "Analyze the impact of AI on energy management",
    "Analyze the impact of AI on space exploration",
    "Analyze the impact of AI on pharmaceutical drug discovery",
    # --- Culture & Creative ---
    "Analyze the impact of AI on the entertainment industry",
    "Analyze the impact of AI on visual arts and design",
    "Analyze the impact of AI on music composition and production",
    "Analyze the impact of AI on gaming and interactive media",
    "Analyze the impact of AI on language translation and linguistics",
    # --- Education & Learning ---
    "Analyze the impact of AI on higher education and online learning",
    "Analyze the impact of AI on K-12 education and personalized learning",
    "Analyze the impact of AI on skill development and professional training",
    "Analyze the impact of AI on academic research and scientific discovery",
    # --- Finance & Banking ---
    "Analyze the impact of AI on banking and financial services",
    "Analyze the impact of AI on investment and trading",
    "Analyze the impact of AI on fraud detection and risk management",
    "Analyze the impact of AI on insurance industry",
    # --- Transportation & Logistics ---
    "Analyze the impact of AI on autonomous vehicles and transportation",
    "Analyze the impact of AI on drone delivery and logistics",
    "Analyze the impact of AI on traffic management and urban planning",
    "Analyze the impact of AI on aviation and pilot assistance",
    # --- Communication & Social ---
    "Analyze the impact of AI on social media and content moderation",
    "Analyze the impact of AI on customer service and chatbots",
    "Analyze the impact of AI on natural language processing",
    "Analyze the impact of AI on accessibility and assistive technologies",
    # --- Real Estate & Construction ---
    "Analyze the impact of AI on real estate and property valuation",
    "Analyze the impact of AI on architecture and building design",
    "Analyze the impact of AI on construction management and safety",
    # --- Sports & Recreation ---
    "Analyze the impact of AI on sports analytics and performance optimization",
    "Analyze the impact of AI on fitness tracking and health monitoring",
    "Analyze the impact of AI on sports training and coaching",
    # --- Fashion & Retail ---
    "Analyze the impact of AI on fashion design and trend forecasting",
    "Analyze the impact of AI on personal styling and recommendation systems",
    "Analyze the impact of AI on inventory management in retail",
    # --- Travel & Hospitality ---
    "Analyze the impact of AI on travel booking and recommendation engines",
    "Analyze the impact of AI on hotel management and customer experience",
    "Analyze the impact of AI on tourism and destination planning",
    # --- Manufacturing & Quality ---
    "Analyze the impact of AI on quality control and defect detection",
    "Analyze the impact of AI on predictive maintenance in factories",
    "Analyze the impact of AI on robotics and industrial automation",
    # --- Government & Public Sector ---
    "Analyze the impact of AI on government efficiency and public services",
    "Analyze the impact of AI on tax compliance and revenue management",
    "Analyze the impact of AI on disaster response and emergency management",
    # --- Ethics & Philosophy ---
    "Analyze the ethical implications of AI decision-making systems",
    "Analyze the impact of AI on human autonomy and free will",
    "Analyze the impact of AI on bias and fairness in decision systems",
    # --- Additional Healthcare Applications ---
    "Analyze the impact of AI on medical imaging and diagnostics",
    "Analyze the impact of AI on personalized medicine and treatment plans",
    "Analyze the impact of AI on mental health and psychological support",
    "Analyze the impact of AI on elderly care and nursing assistance",
    # --- Additional Environmental Applications ---
    "Analyze the impact of AI on climate change prediction and modeling",
    "Analyze the impact of AI on wildlife conservation and monitoring",
    "Analyze the impact of AI on water resource management",
    "Analyze the impact of AI on renewable energy optimization",
    # --- Additional Scientific Research Applications ---
    "Analyze the impact of AI on genomics and genetic research",
    "Analyze the impact of AI on materials science and discovery",
    "Analyze the impact of AI on particle physics and cosmology",
    "Analyze the impact of AI on neuroscience and brain research",
    # --- Additional Social & Cultural Applications ---
    "Analyze the impact of AI on language preservation and revitalization",
    "Analyze the impact of AI on cultural heritage preservation",
    "Analyze the impact of AI on social inequality and digital divide",
    "Analyze the impact of AI on human creativity and expression",
    "Analyze the impact of AI on interpersonal communication and relationships",
    # --- Additional Education Applications ---
    "Analyze the impact of AI on special education and inclusive learning",
    "Analyze the impact of AI on educational assessment and testing",
    "Analyze the impact of AI on lifelong learning and adult education",
    "Analyze the impact of AI on educational administration and policy",
    "Analyze the impact of AI on language learning and acquisition",
    # --- Additional Finance Applications ---
    "Analyze the impact of AI on cryptocurrency and blockchain technology",
    "Analyze the impact of AI on financial inclusion and microfinance",
    "Analyze the impact of AI on central banking and monetary policy",
    "Analyze the impact of AI on financial literacy and education",
    "Analyze the impact of AI on wealth management and financial planning",
    # -- Additional Transportation Applications ---
    "Analyze the impact of AI on public transportation and transit systems",
    "Analyze the impact of AI on maritime transportation and shipping",
    "Analyze the impact of AI on logistics and supply chain optimization",
    "Analyze the impact of AI on traffic safety and accident prevention",
    "Analyze the impact of AI on urban mobility and smart cities",
]

def _build_swarm() -> HeavySwarm:
    if HeavySwarm is None:
        raise ModuleNotFoundError(
            "swarms package is unavailable. Install runtime "
            "dependencies before executing HeavySwarm tasks."
        )
    return HeavySwarm(
        name="Research Team",
        description="Multi-agent analysis system",
        worker_model_name=MODEL_NAME,
        question_agent_model_name=MODEL_NAME,
        show_dashboard=SHOW_DASHBOARD,
        llm_base_url=LLM_BASE_URL,
        llm_api_key=LLM_API_KEY,
        max_workers=4,
        streaming_on=True,  # Enable streaming for faster fault detection
    )


def _run_one_task(task_item: Tuple[int, str]) -> Dict[str, Any]:
    task_index, task = task_item
    swarm = _build_swarm()
    recorder = TimingRecorder(task_index=task_index, task=task)

    from swarms.utils import litellm_wrapper

    original_completion = litellm_wrapper.completion

    def timed_completion(*args, **kwargs):
        agent = _extract_agent_from_kwargs(kwargs)
        start = time.perf_counter()
        try:
            response = original_completion(*args, **kwargs)
            recorder.record_request(
                agent=agent,
                latency_seconds=time.perf_counter() - start,
                response=response,
            )
            return response
        except Exception as exc:
            recorder.record_request(
                agent=agent,
                latency_seconds=time.perf_counter() - start,
                response=None,
                error=str(exc),
            )
            raise

    litellm_wrapper.completion = timed_completion
    task_start = time.perf_counter()
    try:
        result = swarm.run(task)
    finally:
        litellm_wrapper.completion = original_completion

    timing = recorder.build_task_payload(
        task_duration_seconds=time.perf_counter() - task_start
    )

    return {
        "task_index": task_index,
        "task": task,
        "result": result,
        "timing": timing,
    }


def _write_csv(
    path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_timing_tables(
    results: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    detailed_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    sorted_results = sorted(
        results, key=lambda item: item["task_index"]
    )
    for result in sorted_results:
        timing = result.get("timing", {})
        task_index = result.get("task_index")
        task_text = result.get("task", "")
        task_duration = float(
            timing.get("task_duration_seconds", 0.0)
        )
        per_agent = timing.get("per_agent", {})
        requests = timing.get("requests", [])

        for request_row in requests:
            detailed_rows.append(
                {
                    "task_index": task_index,
                    "task": task_text,
                    "agent": request_row.get("agent", "unknown"),
                    "run_index": request_row.get("run_index", 0),
                    "latency_seconds": request_row.get(
                        "latency_seconds", 0.0
                    ),
                    "prompt_tokens": request_row.get(
                        "prompt_tokens", ""
                    ),
                    "completion_tokens": request_row.get(
                        "completion_tokens", ""
                    ),
                    "total_tokens": request_row.get(
                        "total_tokens", ""
                    ),
                    "cached_tokens": request_row.get(
                        "cached_tokens", ""
                    ),
                    "cache_hit_ratio": request_row.get(
                        "cache_hit_ratio", ""
                    ),
                    "error": request_row.get("error", ""),
                }
            )

        summary_row: Dict[str, Any] = {
            "task_index": task_index,
            "task": task_text,
            "task_duration_seconds": task_duration,
        }
        for agent in AGENT_ORDER:
            metrics = per_agent.get(agent, {})
            summary_row[f"{agent}_seconds"] = metrics.get(
                "latency_seconds", 0.0
            )
            summary_row[f"{agent}_prefill_tokens"] = metrics.get(
                "prompt_tokens", 0
            )
            summary_row[f"{agent}_decode_tokens"] = metrics.get(
                "completion_tokens", 0
            )
            summary_row[f"{agent}_total_tokens"] = metrics.get(
                "total_tokens", 0
            )
            summary_row[f"{agent}_cached_tokens"] = metrics.get(
                "cached_tokens", ""
            )
            cache_hit_ratio = metrics.get("cache_hit_ratio")
            summary_row[f"{agent}_cache_hit_ratio"] = (
                float(cache_hit_ratio)
                if cache_hit_ratio is not None
                else ""
            )
        summary_rows.append(summary_row)

    return detailed_rows, summary_rows


def _load_summary_rows_from_csv(path: Path) -> List[Dict[str, Any]]:
    def _num(value: str, as_int: bool = False) -> Any:
        text = (value or "").strip()
        if text == "":
            return 0 if as_int else 0.0
        try:
            if as_int:
                return int(float(text))
            return float(text)
        except ValueError:
            return 0 if as_int else 0.0

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            one: Dict[str, Any] = {
                "task_index": _num(
                    row.get("task_index", "0"), as_int=True
                ),
                "task": row.get("task", ""),
                "task_duration_seconds": _num(
                    row.get("task_duration_seconds", "0")
                ),
            }
            for agent in AGENT_ORDER:
                one[f"{agent}_seconds"] = _num(
                    row.get(f"{agent}_seconds", "0")
                )
                one[f"{agent}_prefill_tokens"] = _num(
                    row.get(f"{agent}_prefill_tokens", "0"),
                    as_int=True,
                )
                one[f"{agent}_decode_tokens"] = _num(
                    row.get(f"{agent}_decode_tokens", "0"),
                    as_int=True,
                )
                one[f"{agent}_total_tokens"] = _num(
                    row.get(f"{agent}_total_tokens", "0"),
                    as_int=True,
                )
                one[f"{agent}_cached_tokens"] = _num(
                    row.get(f"{agent}_cached_tokens", "0"),
                    as_int=True,
                )
                one[f"{agent}_cache_hit_ratio"] = _num(
                    row.get(f"{agent}_cache_hit_ratio", "0")
                )
            rows.append(one)

    rows.sort(key=lambda item: item["task_index"])
    return rows


def _plot_line_report(
    summary_rows: List[Dict[str, Any]],
    png_path: Path,
    pdf_path: Path,
) -> bool:
    if not summary_rows:
        return False

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import numpy as np
    except Exception as exc:
        print(
            "[heavy_swarm] Skip plotting line report "
            f"(matplotlib unavailable): {exc}"
        )
        return False

    paper_style = {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
        ],
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 1.0,
        "grid.color": "#c7c7c7",
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.75,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    plt.rcParams.update(paper_style)

    agent_styles = {
        "question": ("#1f77b4", "o"),
        "research": ("#d99600", "s"),
        "analysis": ("#1b9e77", "^"),
        "alternatives": ("#d95f02", "D"),
        "verification": ("#cc79a7", "v"),
        "synthesis": ("#4ea8de", "o"),
    }

    task_indices = [row["task_index"] for row in summary_rows]
    task_durations = [
        float(row["task_duration_seconds"]) for row in summary_rows
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    fig.suptitle(
        "HeavySwarm Runtime and Token Profiling (Line Charts)",
        fontsize=19,
        fontweight="bold",
    )

    ax_prefill = axes[0, 0]
    for agent in AGENT_ORDER:
        color, marker = agent_styles[agent]
        y_values = [
            row.get(f"{agent}_prefill_tokens", 0) for row in summary_rows
        ]
        ax_prefill.plot(
            task_indices,
            y_values,
            label=agent.capitalize(),
            color=color,
            marker=marker,
            linewidth=1.6,
            markersize=3.2,
        )
    ax_prefill.set_title(
        "(A) Agent Prefill Tokens by Task",
        fontsize=12,
        fontweight="bold",
    )
    ax_prefill.set_xlabel("Task Index", fontsize=11, fontweight="bold")
    ax_prefill.set_ylabel("Tokens", fontsize=11, fontweight="bold")
    ax_prefill.grid(True, axis="both")

    ax_decode = axes[0, 1]
    for agent in AGENT_ORDER:
        color, marker = agent_styles[agent]
        y_values = [
            row.get(f"{agent}_decode_tokens", 0) for row in summary_rows
        ]
        ax_decode.plot(
            task_indices,
            y_values,
            label=agent.capitalize(),
            color=color,
            marker=marker,
            linewidth=1.6,
            markersize=3.2,
        )
    ax_decode.set_title(
        "(B) Agent Decode Tokens by Task",
        fontsize=12,
        fontweight="bold",
    )
    ax_decode.set_xlabel("Task Index", fontsize=11, fontweight="bold")
    ax_decode.set_ylabel("Tokens", fontsize=11, fontweight="bold")
    ax_decode.grid(True, axis="both")

    ax_runtime = axes[1, 0]
    for agent in AGENT_ORDER:
        color, marker = agent_styles[agent]
        y_values = [
            row.get(f"{agent}_seconds", 0.0) for row in summary_rows
        ]
        ax_runtime.plot(
            task_indices,
            y_values,
            label=agent.capitalize(),
            color=color,
            marker=marker,
            linewidth=1.6,
            markersize=3.2,
        )
    ax_runtime.set_title(
        "(C) Agent Runtime by Task",
        fontsize=12,
        fontweight="bold",
    )
    ax_runtime.set_xlabel("Task Index", fontsize=11, fontweight="bold")
    ax_runtime.set_ylabel("Time (s)", fontsize=11, fontweight="bold")
    ax_runtime.grid(True, axis="both")

    ax_completion = axes[1, 1]
    mean_duration = float(np.mean(task_durations))
    median_duration = float(np.median(task_durations))
    ax_completion.plot(
        task_indices,
        task_durations,
        color="#222222",
        marker="o",
        linewidth=1.8,
        markersize=3.2,
        label="Task completion time",
    )
    ax_completion.axhline(
        mean_duration,
        color="#8f8f8f",
        linestyle="--",
        linewidth=1.2,
        label=f"Mean={mean_duration:.1f}s",
    )
    ax_completion.axhline(
        median_duration,
        color="#8f8f8f",
        linestyle=":",
        linewidth=1.2,
        label=f"Median={median_duration:.1f}s",
    )
    ax_completion.set_title(
        "(D) Task Completion Time",
        fontsize=12,
        fontweight="bold",
    )
    ax_completion.set_xlabel(
        "Task Index", fontsize=11, fontweight="bold"
    )
    ax_completion.set_ylabel("Time (s)", fontsize=11, fontweight="bold")
    ax_completion.grid(True, axis="both")
    ax_completion.legend(loc="upper right")

    for ax in axes.flat:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.set_axisbelow(True)

    handles, labels = ax_runtime.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=True,
        fontsize=10,
        borderpad=0.35,
        handlelength=1.5,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.965))
    fig.savefig(png_path, dpi=500, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    svg_path = png_path.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_cache_hit_ratio_report(
    detailed_rows: List[Dict[str, Any]],
    png_path: Path,
    pdf_path: Path,
) -> bool:
    if not detailed_rows:
        return False

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import numpy as np
    except Exception as exc:
        print(
            "[heavy_swarm] Skip plotting cache-hit report "
            f"(matplotlib unavailable): {exc}"
        )
        return False

    agent_styles = {
        "question": ("#1f77b4", "o"),
        "research": ("#d99600", "s"),
        "analysis": ("#1b9e77", "^"),
        "alternatives": ("#d95f02", "D"),
        "verification": ("#cc79a7", "v"),
        "synthesis": ("#4ea8de", "o"),
        "unknown": ("#666666", "x"),
    }

    rows_by_agent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in detailed_rows:
        agent = _canonical_agent_name(str(row.get("agent", "unknown")))
        prompt = _to_int_or_none(row.get("prompt_tokens")) or 0
        raw_cached = _to_int_or_none(row.get("cached_tokens"))
        ratio_value = row.get("cache_hit_ratio")
        if ratio_value in ("", None) and raw_cached is None:
            continue

        cached = raw_cached or 0
        ratio = None
        if ratio_value not in ("", None):
            try:
                ratio = float(ratio_value)
            except (TypeError, ValueError):
                ratio = None
        if ratio is None and raw_cached is not None and prompt > 0:
            ratio = float(cached) / float(prompt)
        if ratio is None:
            ratio = 0.0
        ratio = max(0.0, ratio)

        rows_by_agent[agent].append(
            {
                "task_index": _to_int_or_none(row.get("task_index")) or 0,
                "run_index": _to_int_or_none(row.get("run_index")) or 0,
                "prompt_tokens": prompt,
                "cached_tokens": cached,
                "cache_hit_ratio": ratio,
            }
        )

    ordered_agents = [
        agent for agent in AGENT_ORDER if agent in rows_by_agent
    ]
    ordered_agents += sorted(
        agent
        for agent in rows_by_agent
        if agent not in AGENT_ORDER
    )
    if not ordered_agents:
        print(
            "[heavy_swarm] Skip plotting cache-hit report "
            "(no cache-hit fields in detailed rows)."
        )
        return False

    max_ratio = 1.0
    for rows in rows_by_agent.values():
        for row in rows:
            max_ratio = max(max_ratio, row["cache_hit_ratio"])
    y_upper = max(1.05, max_ratio * 1.05)

    n_cols = 2
    n_rows = (len(ordered_agents) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(13.5, max(3.6 * n_rows, 4.2)),
        sharey=True,
    )
    axes_arr = np.array(axes).reshape(-1)

    for idx, agent in enumerate(ordered_agents):
        ax = axes_arr[idx]
        rows = sorted(
            rows_by_agent[agent],
            key=lambda item: (item["task_index"], item["run_index"]),
        )
        x_values = list(range(1, len(rows) + 1))
        y_values = [item["cache_hit_ratio"] for item in rows]
        color, marker = agent_styles.get(agent, ("#444444", "o"))
        ax.plot(
            x_values,
            y_values,
            color=color,
            marker=marker,
            linewidth=1.5,
            markersize=3.6,
        )

        total_prompt = sum(item["prompt_tokens"] for item in rows)
        total_cached = sum(item["cached_tokens"] for item in rows)
        weighted_ratio = (
            float(total_cached) / float(total_prompt)
            if total_prompt > 0
            else 0.0
        )
        ax.axhline(
            weighted_ratio,
            color="#8a8a8a",
            linestyle="--",
            linewidth=1.0,
            label=f"Weighted avg={weighted_ratio:.1%}",
        )
        ax.set_title(
            f"{agent.capitalize()} (n={len(rows)})",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Request index", fontsize=10, fontweight="bold")
        ax.set_ylabel("Cache hit ratio", fontsize=10, fontweight="bold")
        ax.set_ylim(0.0, y_upper)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.grid(True, axis="both", linestyle=":", linewidth=0.6)
        ax.legend(loc="lower right", fontsize=8, frameon=True)

    for idx in range(len(ordered_agents), len(axes_arr)):
        fig.delaxes(axes_arr[idx])

    fig.suptitle(
        "HeavySwarm Cache Hit Ratio by Agent and Request",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.0, 1, 0.96))
    fig.savefig(png_path, dpi=500, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    svg_path = png_path.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return True


def _save_timing_reports(results: List[Dict[str, Any]]) -> None:
    TIMING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    detailed_rows, summary_rows = _build_timing_tables(results)

    detailed_csv = (
        TIMING_OUTPUT_DIR
        / f"agent_timing_detailed_{timestamp}.csv"
    )
    summary_csv = (
        TIMING_OUTPUT_DIR / f"task_timing_summary_{timestamp}.csv"
    )

    detailed_fieldnames = [
        "task_index",
        "task",
        "agent",
        "run_index",
        "latency_seconds",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "cache_hit_ratio",
        "error",
    ]
    summary_fieldnames = [
        "task_index",
        "task",
        "task_duration_seconds",
    ]
    for agent in AGENT_ORDER:
        summary_fieldnames.append(f"{agent}_seconds")
    for agent in AGENT_ORDER:
        summary_fieldnames.append(f"{agent}_prefill_tokens")
    for agent in AGENT_ORDER:
        summary_fieldnames.append(f"{agent}_decode_tokens")
    for agent in AGENT_ORDER:
        summary_fieldnames.append(f"{agent}_total_tokens")
    for agent in AGENT_ORDER:
        summary_fieldnames.append(f"{agent}_cached_tokens")
    for agent in AGENT_ORDER:
        summary_fieldnames.append(f"{agent}_cache_hit_ratio")

    _write_csv(
        detailed_csv,
        detailed_fieldnames,
        detailed_rows,
    )
    _write_csv(
        summary_csv,
        summary_fieldnames,
        summary_rows,
    )

    line_png = (
        TIMING_OUTPUT_DIR / f"timing_metrics_{timestamp}_line_report.png"
    )
    line_pdf = (
        TIMING_OUTPUT_DIR / f"timing_metrics_{timestamp}_line_report.pdf"
    )
    cache_png = (
        TIMING_OUTPUT_DIR
        / f"timing_metrics_{timestamp}_cache_hit_report.png"
    )
    cache_pdf = (
        TIMING_OUTPUT_DIR
        / f"timing_metrics_{timestamp}_cache_hit_report.pdf"
    )

    line_plotted = _plot_line_report(
        summary_rows=summary_rows,
        png_path=line_png,
        pdf_path=line_pdf,
    )
    cache_plotted = _plot_cache_hit_ratio_report(
        detailed_rows=detailed_rows,
        png_path=cache_png,
        pdf_path=cache_pdf,
    )

    raw_timing_json = TIMING_OUTPUT_DIR / f"timing_metrics_{timestamp}.json"
    with raw_timing_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(),
                "results": results,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[heavy_swarm] Timing CSV: {detailed_csv}")
    print(f"[heavy_swarm] Timing CSV: {summary_csv}")
    print(f"[heavy_swarm] Timing JSON: {raw_timing_json}")
    if line_plotted:
        print(f"[heavy_swarm] Timing report: {line_png}")
        print(f"[heavy_swarm] Timing report: {line_pdf}")
        print(
            f"[heavy_swarm] Timing report: {line_png.with_suffix('.svg')}"
        )
    if cache_plotted:
        print(f"[heavy_swarm] Timing report: {cache_png}")
        print(f"[heavy_swarm] Timing report: {cache_pdf}")
        print(
            f"[heavy_swarm] Timing report: {cache_png.with_suffix('.svg')}"
        )


def main() -> None:
    summary_csv_for_plot = os.getenv(
        "HEAVY_SWARM_SUMMARY_CSV_FOR_PLOT", ""
    ).strip()
    if summary_csv_for_plot:
        summary_csv_path = Path(summary_csv_for_plot).expanduser()
        if not summary_csv_path.is_absolute():
            summary_csv_path = (Path.cwd() / summary_csv_path).resolve()
        if not summary_csv_path.exists():
            print(
                "[heavy_swarm] Summary CSV not found: "
                f"{summary_csv_path}"
            )
            return

        summary_rows = _load_summary_rows_from_csv(summary_csv_path)
        if not summary_rows:
            print(
                "[heavy_swarm] Summary CSV is empty, skip plotting: "
                f"{summary_csv_path}"
            )
            return

        TIMING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        line_png = (
            TIMING_OUTPUT_DIR
            / f"timing_metrics_{timestamp}_line_report.png"
        )
        line_pdf = (
            TIMING_OUTPUT_DIR
            / f"timing_metrics_{timestamp}_line_report.pdf"
        )
        plotted = _plot_line_report(summary_rows, line_png, line_pdf)
        if plotted:
            print(f"[heavy_swarm] Replot line report: {line_png}")
            print(f"[heavy_swarm] Replot line report: {line_pdf}")
            print(
                f"[heavy_swarm] Replot line report: {line_png.with_suffix('.svg')}"
            )
        return

    selected_tasks = tasks[:TASK_LIMIT]
    if not selected_tasks:
        print("[heavy_swarm] No tasks selected.")
        return

    if litellm is None:
        print(
            "[heavy_swarm] Missing dependency: litellm. "
            "Install it before running HeavySwarm tasks."
        )
        return
    if HeavySwarm is None:
        print(
            "[heavy_swarm] Missing dependency: swarms package. "
            "Install it before running HeavySwarm tasks."
        )
        return

    indexed_tasks = list(enumerate(selected_tasks))
    workers = min(TASK_PROCESSES, len(indexed_tasks))
    print(
        f"[heavy_swarm] Running {len(indexed_tasks)} task(s) "
        f"with {workers} process(es)."
    )

    if workers == 1:
        results = [_run_one_task(task_item) for task_item in indexed_tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_run_one_task, indexed_tasks))

    results = sorted(results, key=lambda item: item["task_index"])

    output_path = os.getenv("HEAVY_SWARM_RESULTS_PATH", "").strip()
    if output_path:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        print(f"[heavy_swarm] Results written to {output_path}")

    if ENABLE_TIMING_REPORTS:
        _save_timing_reports(results)

    print(f"[heavy_swarm] Completed {len(results)} task(s).")


if __name__ == "__main__":
    main()
