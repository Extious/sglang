import os

EXA_API_KEY  = os.getenv('EXA_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if EXA_API_KEY and OPENAI_API_KEY:
    os.environ['EXA_API_KEY'] = EXA_API_KEY
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    print('✅ API keys set successfully!')
else:
    raise ValueError('Please enter both EXA and OPENAI API keys')



from crewai import Agent, Crew, Task, Process, LLM
from crewai_tools import EXASearchTool
from IPython.display import display, Markdown

# Initialize LLM and tools
llm = LLM(model="gpt-4o", api_key=OPENAI_API_KEY, max_tokens=16_384)
tool = EXASearchTool(api_key=EXA_API_KEY)

# Create collaborative agents
researcher = Agent(
    role="Research Specialist",
    goal="Find accurate, up-to-date information on any topic",
    backstory="""You're a meticulous researcher with expertise in finding
    reliable sources and fact-checking information across various domains.
    Use the tool to do in-depth research.
    """,
    allow_delegation=True,
    llm=llm,
    tools=[tool],
    verbose=True
)

writer = Agent(
    role="Content Writer",
    goal="Create engaging, well-structured content",
    backstory="""You're an experienced New York Times article writer who excels at transforming
    research content into compelling, readable content for different audiences. in the style of NYT articles.""",
    allow_delegation=True,
    llm=llm,
    verbose=True
)

editor = Agent(
    role="Content Editor",
    goal="Ensure content quality and consistency",
    backstory="""You're an experienced editor with an eye for detail,
    ensuring content meets high standards for clarity and accuracy of NYT articles.""",
    allow_delegation=True,
    llm=llm,
    verbose=True
)

# Create a task that encourages collaboration
article_task = Task(
    description="""Write a comprehensive 2000-word article about {topic} for this {year}.
    The article should be structured like a New York Time article. Make sure the byline author is
    this 'CrewAI Agent and Tony Kipkemboi'.
    Collaborate with your teammates to ensure accuracy and quality.""",
    expected_output="A well-researched, engaging 2000-word article with proper structure and citations.",
    agent=writer  # Writer leads, but can delegate research to researcher
)

# Create collaborative crew
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[article_task],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff(inputs={"topic": input("Enter topic to write an article about: "), "year":"2025"})

