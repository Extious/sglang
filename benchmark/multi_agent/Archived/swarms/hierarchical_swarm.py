import os

# Must set no_proxy BEFORE importing libraries that cache proxy settings
os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",gpu14"
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",gpu14"

import litellm
from swarms import Agent

# Register custom model so litellm can resolve max_tokens
litellm.model_cost["openai/Qwen/Qwen3-4B-Instruct-2507"] = {
    "max_tokens": 32768,
    "input_cost_per_token": 0,
    "output_cost_per_token": 0,
    "supports_function_calling": True,
}
if "openai/Qwen/Qwen3-4B-Instruct-2507" not in litellm.model_list:
    litellm.model_list.append("openai/Qwen/Qwen3-4B-Instruct-2507")

LLM_BASE_URL = "http://gpu14:8000/v1"
LLM_API_KEY = os.getenv("OPENROUTER_API_KEY")

os.environ["OPENAI_BASE_URL"] = LLM_BASE_URL
if LLM_API_KEY:
    os.environ["OPENAI_API_KEY"] = LLM_API_KEY

from swarms import HeavySwarm
from swarms.utils.litellm_wrapper import LiteLLM
import swarms.structs.heavy_swarm as hierarchical_swarm_module
import swarms.structs.agent as agent_module
import swarms.utils.litellm_wrapper as litellm_wrapper

LLM_ARGS = {
    "base_url": LLM_BASE_URL,
    "drop_params": False,
}
if LLM_API_KEY:
    LLM_ARGS["api_key"] = LLM_API_KEY

class LiteLLMWithBase(LiteLLM):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("base_url", LLM_BASE_URL)
        kwargs.setdefault("drop_params", False)
        if LLM_API_KEY:
            kwargs.setdefault("api_key", LLM_API_KEY)
        super().__init__(*args, **kwargs)
        self._sglang_user = kwargs.get("user")
        self._sglang_headers = kwargs.get("headers") or kwargs.get("extra_headers")
        self._sglang_debug = os.getenv("SGLANG_AGENT_DEBUG") == "1"

    def _inject_kwargs(self, kwargs):
        if self._sglang_user and "user" not in kwargs:
            kwargs["user"] = self._sglang_user
        headers = self._sglang_headers
        if headers:
            kwargs.setdefault("headers", headers)
            kwargs.setdefault("extra_headers", headers)
        if self._sglang_debug:
            user_val = kwargs.get("user")
            headers_val = kwargs.get("headers") or kwargs.get("extra_headers")
            print(f"sglang agent debug user={user_val} headers={headers_val}")
        return kwargs

    def run(self, *args, **kwargs):
        kwargs = self._inject_kwargs(kwargs)
        return super().run(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        kwargs = self._inject_kwargs(kwargs)
        return super().__call__(*args, **kwargs)

hierarchical_swarm_module.LiteLLM = LiteLLMWithBase
agent_module.LiteLLM = LiteLLMWithBase
litellm_wrapper.LiteLLM = LiteLLMWithBase

_original_create_agents = HeavySwarm.create_agents

def _apply_agent_meta(agent, agent_name):
    headers = {"x-sglang-agent-id": agent_name}
    agent.llm_args = {
        **LLM_ARGS,
        "user": agent_name,
        "headers": headers,
        "extra_headers": headers,
        "drop_params": False,
    }
    llm = getattr(agent, "llm", None)
    if llm is None:
        return
    setattr(llm, "_sglang_user", agent_name)
    setattr(llm, "_sglang_headers", headers)
    for key in ("llm_args", "kwargs", "model_kwargs"):
        store = getattr(llm, key, None)
        if isinstance(store, dict):
            store.update(
                {
                    "user": agent_name,
                    "headers": headers,
                    "extra_headers": headers,
                    "drop_params": False,
                }
            )

def create_agents_with_base(self):
    agents = _original_create_agents(self)
    for agent in agents.values():
        agent_name = getattr(agent, "agent_name", None) or getattr(agent, "name", None)
        agent_name = agent_name or "agent"
        _apply_agent_meta(agent, agent_name)
        agent.llm_base_url = LLM_BASE_URL
        if LLM_API_KEY:
            agent.llm_api_key = LLM_API_KEY
    return agents

from swarms import Agent, HierarchicalSwarm

HierarchicalSwarm.create_agents = create_agents_with_base

# Define specialized worker agents
market_researcher = Agent(
    agent_name="MarketResearcher",
    system_prompt="Conduct comprehensive market research, analyze trends, and identify opportunities.",
    model_name="openai/Qwen/Qwen3-4B-Instruct-2507",
    top_p=None,
    max_loops=1,
    dynamic_temperature_enabled=True,
)

financial_planner = Agent(
    agent_name="FinancialPlanner",
    system_prompt="Analyze financial implications, create budgets, and project ROI.",
    model_name="openai/Qwen/Qwen3-4B-Instruct-2507",
    top_p=None,
    max_loops=1,
    dynamic_temperature_enabled=True,
)

technical_architect = Agent(
    agent_name="TechnicalArchitect",
    system_prompt="Design technical solutions, evaluate technology stacks, and plan implementation.",
    model_name="openai/Qwen/Qwen3-4B-Instruct-2507",
    top_p=None,
    max_loops=1,
    dynamic_temperature_enabled=True,
)

legal_reviewer = Agent(
    agent_name="LegalReviewer",
    system_prompt="Review legal implications, contracts, and compliance requirements.",
    model_name="openai/Qwen/Qwen3-4B-Instruct-2507",
    top_p=None,
    max_loops=1,
    dynamic_temperature_enabled=True,
)

risk_assessor = Agent(
    agent_name="RiskAssessor",
    system_prompt="Identify potential risks, assess impact, and develop mitigation strategies.",
    model_name="openai/Qwen/Qwen3-4B-Instruct-2507",
    top_p=None,
    max_loops=1,
    dynamic_temperature_enabled=True,
)

# Create the hierarchical swarm
project_swarm = HierarchicalSwarm(
    name="ProductLaunchTeam",
    description="A comprehensive team for planning and executing a new product launch",
    director_model_name="openai/Qwen/Qwen3-4B-Instruct-2507",
    agents=[
        market_researcher,
        financial_planner,
        technical_architect,
        legal_reviewer,
        risk_assessor
    ],
    max_loops=3,  # Allow for feedback and refinement
    verbose=True
)

# Display the hierarchy visualization
project_swarm.display_hierarchy()

# Run a complex project requiring coordinated expertise
project_plan = project_swarm.run(
    "Plan the launch of a revolutionary AI-powered fitness tracking wearable device. "
    "Consider market positioning, technical requirements, financial projections, "
    "legal compliance, and risk management. Create a comprehensive 6-month launch strategy."
)

print("=== Final Project Plan ===")
print(project_plan)