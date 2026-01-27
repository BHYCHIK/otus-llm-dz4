from langchain.tools import tool
import vk_api
import os

@tool
def post_to_vk_tool(post: str):
    """
    Tool, which can make posts to vk.

    Args:
        post (str): The post text to post to vk.
    """
    print("Posting to vk...")
    vk_session = vk_api.VkApi(token=os.environ["VK_TOKEN"])
    vk_session.auth()

    vk = vk_session.get_api()
    vk.methods.post("wall.post", {"message": post})

    vk.wall.post(message=post)