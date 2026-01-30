from tools.vkpost import vkpost
from dotenv import load_dotenv

load_dotenv('.env')

vkpost = vkpost._post("This is test post")