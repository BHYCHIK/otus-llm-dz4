from typing import TypedDict

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

from dotenv import load_dotenv
import os

from pydantic import BaseModel, Field

from few_shots import get_examples_of_plans
from tools.rss_collector import rss_collector
from tools.vkpost import vkpost

load_dotenv('.env')

MAX_FIXES = 3

TEST_FAKE_FACT_DELETION = True

class State(TypedDict):
    original_prompt: str
    auditory: str
    plan_of_article: str
    original_articles: str
    result: str
    fix_num: int
    plan_to_fix: str
    published: bool

llm = ChatOpenAI(
    base_url=os.getenv('API_BASE_URL'),
    model='qwen-3-32b',
    temperature=0.3,
    api_key=os.getenv('API_KEY'))

tools = [rss_collector.last_ai_articles_tool, vkpost.post_to_vk_tool]

llm_with_tools = llm.bind_tools(tools)

memory = MemorySaver()

planner_sys_msg = SystemMessage(content="""Ты состовляешь планы для технических статей""")

class PlannerResponse(BaseModel):
    PlanOfArticle: str = Field(description='План статьи')

def articles_fetcher_call(state: State):
    print('articles_fetcher_call')
    user_message = HumanMessage(f'Получи нужное для уровня аудитории {state['auditory']} колличество статей, из которых можно будет составить свою статью. Используй last_ai_articles_tool. Tool вернет тебе текст, состоящий из нескольких статей. Сохрани их как есть, ничего не модифицируя.')
    agent = create_agent(llm_with_tools, tools=tools, name='articles_fetcher')
    response = agent.invoke({'messages':[user_message]})
    return {
        'original_articles': response['messages'][-1].content,
    }

def planner_agent_call(state: State):
    print('planner')
    system_message = SystemMessage(f"Ты должен составить план технической статьи об исскуственном интеллекте для следующего уровня подготовки аудитории {state['auditory']}")
    user_message = HumanMessage(f"""Составь план переписанной статьи на основе следующих базовых статей.
    
                                Примеры планов:
                                {get_examples_of_plans()}
                                
                                Базовые статьи: {state['original_articles']}""")
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
    response = llm.invoke([system_message, user_message])

    result = response.content
    if TEST_FAKE_FACT_DELETION:
        result = result + '. А еще вакцины вызывают рак, бесплодие. А в эпидемию COVID-19 людей чипировали'

    return {
        'result': result,
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

def router(state: State):
    if (state['fix_num'] < MAX_FIXES) and (state['plan_to_fix'] != ''):
        return 'quality_fixer'
    else:
        return 'end'

def quality_checker_agent_call(state: State):
    print('quality_checker_agent_call')
    system_message = SystemMessage(content="""
        Ты профессиональный редактор в журнале про искусственный интеллект.
        Ты выдаешь то, что нужно исправить в статьях или, если все хорошо,
        отдаешь на публикацию в виде поста в VK.
    """)
    user_message = HumanMessage(
        f"""Тебе на публикацию пришла статья.

        Если статья достаточно хороша для публикации:
         - сделай с ней пост в VK. В посте полностью и без изменений бери текст статьи. Ничего не меняй и не добавляй.
         - Ответь одним словом "posted"

        Если статья требует правок до публикации, то верни список правок, которые нужно внести. Только список правок, без упоминания правил. Ничего не игнорируй.

        Критерии качества статьи:
         - Нецензурная лексика
         - Политика
         - Искажение фактов
         - Отсутствие логики в статье
         - Недостаточно раскрыта тема

        Текст статьи (ее нужно сохранить как есть, ничего не меняя):
        {state['result']}""")

    agent = create_agent(llm_with_tools, tools=tools, name='quality_checker')
    content = agent.invoke({'messages': [system_message, user_message]})['messages'][-1].content

    if (content.find('posted') == -1) or (len(content) > 20):
        return {
            'plan_to_fix': content,
        }
    else:
        if content.find('posted') >= 0:
            return {
                'published': True,
                'plan_to_fix': '',
            }
        else:
            return {
                'published': False,
                'plan_to_fix': '',
            }

def quality_fixer_agent_call(state: State):
    print('quality_fixer')
    system_message = SystemMessage(content="""Ты технический директор холдинга. Пишешь для социальных сетей.
                Твой профессиональный интерес управление, искусственный интеллект и надежность.
                Только качественные тексты без маркетингового буллшита.""")
    user_message = HumanMessage(f"""Исправь статью для уровня аудитории: {state['auditory']}")
                                Список исправлений:\n{state['plan_to_fix']}\n\n
                                Статья, которую нужно исправить:\n{state['result']}
                                """)
    response = llm.invoke([system_message, user_message])
    return {
        'result': response.content,
        'fix_num': state['fix_num'] + 1,
    }

#TODO add gurdrails
workflow = StateGraph(State)
workflow.add_node('level_define', level_define_agent_call)
workflow.add_node('planner', planner_agent_call)
workflow.add_node('copyrighter', copyrighter_agent_call)
workflow.add_node('articles_fetcher', articles_fetcher_call)
workflow.add_node('quality_checker', quality_checker_agent_call)
workflow.add_node('quality_fixer', quality_fixer_agent_call)

workflow.add_edge(START, "level_define")
workflow.add_edge("level_define", "articles_fetcher")
workflow.add_edge("articles_fetcher", "planner")
workflow.add_edge("planner", "copyrighter")
workflow.add_edge("copyrighter", 'quality_checker')
workflow.add_conditional_edges('quality_checker', router, {
    'quality_fixer': 'quality_fixer',
    'end': END,
})
workflow.add_edge("quality_fixer", 'quality_checker')

app = workflow.compile(checkpointer=memory)

def main():
    config: RunnableConfig = {
        'configurable': {'thread_id': 1},
        'callbacks': [LangfuseCallbackHandler()],
    }

    initial_prompt = 'Напиши статью про искусственный интеллект для моей жены senior ml специалиста.'

    app.invoke({'original_prompt': initial_prompt, 'fix_num': 0}, config=config)

    state = app.get_state(config)
    if state.values['published']:
        print('Опубликовано')
    else:
        print('Не опубликовано')

if __name__ == "__main__":
    main()