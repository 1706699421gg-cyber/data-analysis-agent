import sys
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.sqlite import SqliteSaver
from agent.graph import build_app
from memory.vector_store import save_conclusion

def main():
    if len(sys.argv) < 3:
        print('用法: python main.py <csv路径> "<你的问题>"')
        sys.exit(1)
    csv_path, question = sys.argv[1], sys.argv[2]
    thread_id = sys.argv[3] if len(sys.argv) > 3 else "default"
    prompt_version = sys.argv[4] if len(sys.argv) > 4 else "v2"

    handler = CallbackHandler()
    with SqliteSaver.from_conn_string("agent.db") as checkpointer:
        app = build_app(csv_path, checkpointer=checkpointer, prompt_version=prompt_version)
        result = app.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"callbacks": [handler], 
                    "recursion_limit": 10, 
                    "thread_id": thread_id
                    },   #跑完10轮强制报错终止
        )
    print("\n===== Agent 回答 =====")
    answer = result["messages"][-1].content
    print(answer)
    save_conclusion(thread_id, csv_path, answer)
    print("（结论已存入长期记忆）")


if __name__ == "__main__":
    main()
