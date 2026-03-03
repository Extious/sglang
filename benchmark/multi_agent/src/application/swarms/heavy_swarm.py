import os
import sys
from urllib.parse import urlparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "swarms"))

import litellm
from swarms import HeavySwarm

MODEL_NAME = "openai/Qwen/Qwen3-4B-Instruct-2507"
ROUTER_HOST = os.getenv("ROUTER_HOST", "127.0.0.1")
ROUTER_PORT = os.getenv("ROUTER_PORT", "30000")


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


LLM_BASE_URL = _normalize_base_url(os.getenv("LLM_BASE_URL", f"http://{ROUTER_HOST}:{ROUTER_PORT}/v1"))
LLM_API_KEY = (
    os.getenv("LLM_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("OPENROUTER_API_KEY")
    or "sk-local"
)

# Route local router traffic directly: disable external proxies.
for k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ[k] = ""

# Must set no_proxy before client initialization.
router_host = urlparse(LLM_BASE_URL).hostname or ""
for k in ("no_proxy", "NO_PROXY"):
    existing = [x.strip() for x in os.environ.get(k, "").split(",") if x.strip()]
    if router_host and router_host not in existing:
        existing.append(router_host)
    for fixed in ("127.0.0.1", "localhost", "0.0.0.0", "::1"):
        if fixed not in existing:
            existing.append(fixed)
    os.environ[k] = ",".join(existing)

litellm.model_cost[MODEL_NAME] = {"max_tokens": 32768, "input_cost_per_token": 0, "output_cost_per_token": 0}
if MODEL_NAME not in litellm.model_list:
    litellm.model_list.append(MODEL_NAME)

os.environ["OPENAI_BASE_URL"] = LLM_BASE_URL
os.environ["OPENAI_API_KEY"] = LLM_API_KEY

print(f"[heavy_swarm] OPENAI_BASE_URL={LLM_BASE_URL}")
print(f"[heavy_swarm] OPENAI_API_KEY set={bool(LLM_API_KEY)}")

swarm = HeavySwarm(
    name="Research Team",
    description="Multi-agent analysis system",
    worker_model_name=MODEL_NAME,
    question_agent_model_name=MODEL_NAME,
    show_dashboard=True,
    llm_base_url=LLM_BASE_URL,
    llm_api_key=LLM_API_KEY,
)

tasks = [
    "Analyze the impact of AI on healthcare",
    # --- 经济与商业 (Economy & Business) ---
    "Analyze the impact of AI on retail and e-commerce",
    "Analyze the impact of AI on manufacturing and automation",
    "Analyze the impact of AI on supply chain management",
    "Analyze the impact of AI on marketing and advertising",
    "Analyze the impact of AI on human resources and recruitment",

    # --- 社会与法律 (Society & Law) ---
    "Analyze the impact of AI on the legal system",
    "Analyze the impact of AI on public safety and surveillance",
    "Analyze the impact of AI on privacy and data security",
    "Analyze the impact of AI on journalism and media",
    "Analyze the impact of AI on cybersecurity",

    # --- 科学与环境 (Science & Environment) ---
    "Analyze the impact of AI on environmental sustainability",
    "Analyze the impact of AI on agriculture and food production",
    "Analyze the impact of AI on energy management",
    "Analyze the impact of AI on space exploration",
    "Analyze the impact of AI on pharmaceutical drug discovery",

    # --- 文化与创意 (Culture & Creative) ---
    "Analyze the impact of AI on the entertainment industry",
    "Analyze the impact of AI on visual arts and design",
    "Analyze the impact of AI on music composition and production",
    "Analyze the impact of AI on gaming and interactive media",
    "Analyze the impact of AI on language translation and linguistics",
]

results = [swarm.run(task) for task in tasks[:6]]
