import os
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from tools.csv_tools import get_csv_overview
from tools.analysis_tools import describe_column, groupby_aggregate, filter_and_stats, plot_bar, recall_history
from agent.prompts import SYSTEM_PROMPT

TOOLS = [get_csv_overview, describe_column, groupby_aggregate, filter_and_stats, plot_bar, recall_history]

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0,
).bind_tools(TOOLS)

def build_app(csv_path: str, checkpointer=None):
    """构建 Agent。csv_path 填进 Prompt；checkpointer 明天讲记忆时启用"""

    def agent_with_csv(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT.format(csv_path=csv_path))]
        messages += list(state["messages"])
        return {"messages": [llm.invoke(messages)]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent_with_csv)    # 决策：LLM 想干嘛
    g.add_node("tools", ToolNode(TOOLS))   # 执行：真正干活
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)  # 路由：要看 LLM 的回答类型
    g.add_edge("tools", "agent")           # 闭环：工具结果喂回 LLM
    return g.compile(checkpointer=checkpointer)
