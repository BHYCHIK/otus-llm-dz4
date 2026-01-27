from langchain.tools import tool

@tool
def post_to_vk_tool(post: str):
    """
    Tool, which can make posts to vk.

    Args:
        post (str): The post text to post to vk.
    """
    print("Posting to vk...")
    print(post)