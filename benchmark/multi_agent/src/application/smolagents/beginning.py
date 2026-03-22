import os

os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",hkbugpusrv12"
os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",hkbugpusrv12"

from smolagents import CodeAgent, InferenceClientModel, OpenAIServerModel

model_id = "Qwen/Qwen3.5-9B"

model = OpenAIServerModel(model_id=model_id, api_base="http://hkbugpusrv12:30000/v1", api_key="EMPTY_API_KEY")
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)