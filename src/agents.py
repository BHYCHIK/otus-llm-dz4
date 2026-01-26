from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
import os

load_dotenv('.env')

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    print('tool calling')
    return f"It is snowy in {city}!"

llm = ChatOpenAI(
    base_url=os.getenv('API_BASE_URL'),
    model='qwen-3-32b',
    temperature=0.1,
    api_key='APIKEY')

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="I am wether predictor",
    
)

# Run the agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

for message in result['messages']:
    print(message.content)