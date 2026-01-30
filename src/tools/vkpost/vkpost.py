from langchain.tools import tool
import requests
import os

API_VERSION = "5.131"

def _post(post: str):
    token = os.getenv("VK_KEY")
    user_id = os.getenv("VK_OWNER_ID")
    url = "https://api.vk.com/method/wall.post"
    params = {
        "access_token": token,
        "v": API_VERSION,
        "owner_id": user_id,
        "message": post,
        "from_group": 1,
    }

    resp = requests.post(url, data=params)

    if resp.status_code != 200:
        raise Exception(f"not 200 status code to vk_api: {resp.status_code}", resp)

    resp_json = resp.json()

    if "error" in resp_json:
        raise Exception(f'Error from vkapi: {resp_json['error']}')
    else:
        print("Опубликовано, post_id:", resp_json["response"]["post_id"])

@tool
def post_to_vk_tool(post: str):
    """
    Tool, which can make posts to vk.

    Args:
        post (str): The post text to post to vk.
    """
    return _post(post)