import os
import sys

# Must set no_proxy BEFORE importing libraries that cache proxy settings
os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",gpu19"
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",gpu19"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "swarms"))

import litellm
from swarms import HeavySwarm

MODEL_NAME = "openai/Qwen/Qwen3-4B-Instruct-2507"
LLM_BASE_URL = "http://gpu19:8000/v1"
LLM_API_KEY = os.getenv("OPENROUTER_API_KEY")

litellm.model_cost[MODEL_NAME] = {"max_tokens": 32768, "input_cost_per_token": 0, "output_cost_per_token": 0}
if MODEL_NAME not in litellm.model_list:
    litellm.model_list.append(MODEL_NAME)

os.environ["OPENAI_BASE_URL"] = LLM_BASE_URL
if LLM_API_KEY:
    os.environ["OPENAI_API_KEY"] = LLM_API_KEY

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

results = [swarm.run(task) for task in tasks[:5]]