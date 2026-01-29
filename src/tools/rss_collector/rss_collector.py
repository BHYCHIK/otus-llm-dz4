import feedparser
import requests
from langchain.tools import tool
from pydantic import BaseModel
from bs4 import BeautifulSoup
import time

_MAX_POSTS = 7

class OriginalArticleData(BaseModel):
    title: str
    link: str
    full_text: str
    summary: str

class OriginalArticles(BaseModel):
    articles: list[OriginalArticleData]

    def to_text(self) -> str:
        result = ''
        for index, article in enumerate(self.articles):
            result = result + f"""
            Статья номер {index + 1}.
            Ссылка на статью: {article.link}
            Название статьи: {article.title}
            Текст статьи: {article.full_text}\n\n\n"""

        return result

def _parse_habr_article(link: str) -> str:
    resp = requests.get(link, verify=False)
    if resp.status_code != 200:
        raise Exception(f"Cannot get article from habr, status {resp.status_code}")
    soup = BeautifulSoup(resp.content, 'html.parser')
    full_text = soup.find('div', class_='article-formatted-body').get_text()
    return full_text

@tool('last_ai_articles_tool')
def last_ai_articles_tool(articles_num: int):
    """
    Get text, title and link for last ai articles from habr.com

    Args:
        articles_num (int): number of articles to return from 1 to 7

    Returns:
        str: text, title and link for last ai articles
    """
    print(f'Calling last_ai_articles_tool with {articles_num} articles')
    current_utc = time.gmtime()

    rss_feed = 'https://habr.com/ru/rss/hubs/artificial_intelligence/articles/?fl=ru'
    feed_content = requests.get(rss_feed, verify=False)
    feed = feedparser.parse(feed_content.content)
    articles = []
    for entry in feed['entries']:
        if ((entry['published_parsed'].tm_year == current_utc.tm_year) and
                (entry['published_parsed'].tm_yday == current_utc.tm_yday - 1)):


            article = OriginalArticleData(
                title=entry['title'],
                link=entry['link'],
                summary=entry['summary'],
                full_text=_parse_habr_article(entry['link']),
            )

            articles.append(article)

            if len(articles) >= min(articles_num, _MAX_POSTS):
                break

    return OriginalArticles(articles=articles).to_text()