import os
from dotenv import load_dotenv
load_dotenv()   # 读取同目录下的 .env，把 key 注入环境变量

# --- 检查1：DeepSeek 连通性 ---
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0,
)
print("DeepSeek 回复:", llm.invoke("用一句话介绍你自己").content)

# --- 检查2：Langfuse Trace 上报 ---
from langfuse.langchain import CallbackHandler

handler = CallbackHandler()
print("Trace 测试:", llm.invoke("1+1等于几？", config={"callbacks": [handler]}).content)
print("完成！现在去 cloud.langfuse.com 的 Traces 页面刷新看看")
