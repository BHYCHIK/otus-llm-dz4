from typing import TypedDict

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from dotenv import load_dotenv
import os

from pydantic import BaseModel, Field

from tools.rss_collector import rss_collector
from tools.vkpost import vkpost

load_dotenv('.env')

class State(TypedDict):
    original_prompt: str
    auditory: str
    plan_of_article: str
    original_articles: str
    result: str

llm = ChatOpenAI(
    base_url=os.getenv('API_BASE_URL'),
    model='qwen-3-32b',
    temperature=0.1,
    api_key=os.getenv('API_KEY'))

tools = [rss_collector.last_ai_articles_tool, vkpost.post_to_vk_tool]

llm_with_tools = llm.bind_tools(tools)

memory = MemorySaver()

planner_sys_msg = SystemMessage(content="""Ты состовляешь планы для технических статей""")

class PlannerResponse(BaseModel):
    PlanOfArticle: str = Field(description='План статьи')

def articles_fetcher_call(state: State):
    print('articles_fetcher_call')
    user_message = HumanMessage(f'Получи нужное для уровня аудитории {state['auditory']} колличество статей, из которых можно будет составить свою статью. Используй last_ai_articles_tool. Tool вернет тебе json. Верни только его, ничего не добавляя и не убирая.')
    agent = create_agent(llm_with_tools, tools=tools)
    response = agent.invoke({'messages':[user_message]})
    return {
        'original_articles': response['messages'][-1].content,
    }

def planner_agent_call(state: State):
    print('planner')
    system_message = SystemMessage(f"Ты должен составить план технической статьи об исскуственном интеллекте для следующего уровня подготовки аудитории {state['auditory']}")
    user_message = HumanMessage(f"Составь план переписанной статьи на основе следующих базовых статей: {state['original_articles']}")
    response = llm.with_structured_output(PlannerResponse).invoke([system_message, user_message])
    return {
        'plan_of_article': response.PlanOfArticle,
    }

def copyrighter_agent_call(state: State):
    print('copyrighter_agent_call')
    system_message = SystemMessage(content="""Ты технический директор холдинга. Пишешь для социальных сетей.
                Твой профессиональный интерес управление, искусственный интеллект и надежность.
                Только качественные тексты без маркетингового буллшита.""")
    user_message = HumanMessage(f"""Напиши статью для уровня аудитории: {state['auditory']}")
                                План статьи:\n{state['plan_of_article']}\n\n
                                Не креативь. Обязательно возьми материалы возьми из этих статей:\n{state['original_articles']}
                                В новой статье сохрани ссылки на оригинал.
                                """)
    print(user_message.content)
    response = llm.invoke([system_message, user_message])
    return {
        'result': response.content,
    }

class RoleAndNews(BaseModel):
    Auditory: str = Field(description='Уровень аудитории')

#TODO add config and context
def level_define_agent_call(state: State):
    print('level_define')
    user_message = HumanMessage(f"""Определи уровень аудитории по пользовательскому вводу из вариантов:
                                    - Ничего не знает про IT
                                    - Программист без знаний ML
                                    - Junior ML Engineer
                                    - Middle ML Engineer
                                    - Senior ML Engineer

                                    Пользовательский ввод: {state['original_prompt']}""")
    response = llm_with_tools.with_structured_output(RoleAndNews).invoke([user_message])
    res = {
        'auditory': response.Auditory,
    }
    return res

workflow = StateGraph(State)
workflow.add_node('level_define', level_define_agent_call)
workflow.add_node('planner', planner_agent_call)
workflow.add_node('copyrighter', copyrighter_agent_call)
workflow.add_node('articles_fetcher', articles_fetcher_call)

workflow.add_edge(START, "level_define")
workflow.add_edge("level_define", "articles_fetcher")
workflow.add_edge("articles_fetcher", "planner")
workflow.add_edge("planner", "copyrighter")
workflow.add_edge("copyrighter", END)


app = workflow.compile(checkpointer=memory)

def main():
    config: RunnableConfig = {
        'configurable': {'thread_id': 1}
    }

    initial_prompt = 'Напиши статью про искусственный интеллект для моей жены senior ml специалиста.'

    app.invoke({'original_prompt': initial_prompt}, config=config)

    state = app.get_state(config)
    print(state)
    print(state.values['result'])

if __name__ == "__main__":
    main()