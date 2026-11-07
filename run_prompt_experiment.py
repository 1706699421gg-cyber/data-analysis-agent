import csv
from pathlib import Path
from langfuse import get_client
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, ToolMessage   # ★ ToolMessage 新增
from langfuse.langchain import CallbackHandler
from agent.graph import build_app

CSV_PATH = "data/sales.csv"

QUESTIONS = [
    ("Q1", "总销售额是多少？"),
    ("Q2", "哪个产品的总销售额最高？"),
    ("Q3", "8月份的销售额是多少？"),
    ("Q4", "列'销售额2'的平均值是多少？"),
    ("Q5", "今天天气怎么样？"),
    ("Q6", "帮我分析一下这份数据"),
]

def tool_seq(messages):
    seq = []
    for m in messages:
        for tc in (getattr(m, "tool_calls", None) or []):
            seq.append(tc["name"])
    return seq

def tool_returns(messages):                              # ★ 新增函数（1a 第2步）
    parts = []
    for m in messages:
        if isinstance(m, ToolMessage):
            c = m.content if isinstance(m.content, str) else str(m.content)
            parts.append(c[:300])
    return " || ".join(parts)

def main():
    rows = []
    for version in ("v1", "v2"):
        app = build_app(CSV_PATH, prompt_version=version)
        for qid, question in QUESTIONS:
            handler = CallbackHandler()
            print(f"[{version} {qid}] 运行中：{question}")
            try:
                result = app.invoke(
                    {"messages": [HumanMessage(content=question)]},
                    config={
                        "callbacks": [handler],
                        "recursion_limit": 15,
                        "run_name": f"{version}_{qid}",
                        "metadata": {"langfuse_tags": [f"prompt-{version}"]},
                    },
                )
                answer = result["messages"][-1].content
                seq = " -> ".join(tool_seq(result["messages"])) or "(未调用工具)"
                returns = tool_returns(result["messages"])      # ★ 改动1a：try 内，紧跟 seq 之后
            except Exception as e:
                answer = f"(运行异常: {type(e).__name__})"
                seq = "(异常中断)"
                returns = ""                                    # ★ 改动1b：except 内兜底
            rows.append({"version": version, "question_id": qid, "question": question,
                         "tool_sequence": seq, "answer": answer, "manual_assessment": "",
                         "tool_returns": returns, "number_audit": ""})   # ★ 改动2a：字典末尾加两个键
            handler.flush()

            client = get_client()   # langfuse 5.x 的 CallbackHandler 没有 flush()，flush 走全局 client
            if client is not None:
                client.flush()

    Path("docs").mkdir(exist_ok=True)
    out = Path("docs/prompt_experiment.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["version", "question_id", "question",
                                          "tool_sequence", "tool_returns", "number_audit",
                                          "answer", "manual_assessment"])  # ★ 改动2b：列表插入两列
        w.writeheader()
        w.writerows(rows)
    print(f"\n完成：{out} 共{len(rows)}行")

if __name__ == "__main__":
    main()
