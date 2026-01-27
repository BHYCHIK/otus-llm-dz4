from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import MessagesState, StateGraph

from dotenv import load_dotenv
import os

from langgraph.prebuilt import ToolNode, tools_condition

from tools.rss_collector import rss_collector
from tools.vkpost import vkpost

load_dotenv('.env')

llm = ChatOpenAI(
    base_url=os.getenv('API_BASE_URL'),
    model='qwen-3-32b',
    temperature=0.1,
    api_key=os.getenv('API_KEY'))

tools = [rss_collector.last_ai_articles_tool, vkpost.post_to_vk_tool]

llm_with_tools = llm.bind_tools(tools)

memory = MemorySaver()

def call_model(state: MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": response}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges("agent", tools_condition)

app = workflow.compile(checkpointer=memory)

sys_msg = SystemMessage(content="""Ты технический директор холдинга. Пишешь для социальных сетей.
            "Твой профессиональный интерес управление, искусственный интеллект и надежность." +
            "Только качественные тексты без маркетингового буллшита.""")

def main():
    config: RunnableConfig = {
        'configurable': {'thread_id': 1}
    }

    inputs: MessagesState = {
        'messages': [sys_msg,
                     HumanMessage(content='Напиши статью про искусственный интеллект. Используй последние статьи и новости. Проверь внешние статьи и новости на качество. При написании статьи избегай код. Только русский текст без программирования. Если ссылаешься на внешнюю статью, то прикладывай гиперссылку. Добавь хэштегов и опубликуй пост.'
                                  )]
    }

    for event in app.stream(inputs, config=config):
        if "agent" in event:
            print(".", end="", flush=True)
        if "tools" in event:
            print(f"[Используем тул]", end="", flush=True)

    snapshot = app.get_state(config)
    if snapshot.values["messages"]:
        last_message = snapshot.values["messages"][-1]
        if hasattr(last_message, "content"):
            print(f"\n\n🤖 Ассистент:\n{last_message.content}")

if __name__ == "__main__":
    main()