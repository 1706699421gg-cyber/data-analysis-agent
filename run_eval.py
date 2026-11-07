import argparse
import csv
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, ToolMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from agent.graph import build_app

CSV_PATH = "data/sales.csv"
CASES_PATH = "docs/eval_cases.json"
OUT_PATH = "docs/eval_results.csv"


def tool_seq(messages):
    seq = []
    for m in messages:
        for tc in (getattr(m, "tool_calls", None) or []):
            seq.append(tc["name"])
    return seq


def tool_returns(messages):
    parts = []
    for m in messages:
        if isinstance(m, ToolMessage):
            c = m.content if isinstance(m.content, str) else str(m.content)
            parts.append(c[:300])
    return " || ".join(parts)


def run_one(app, question, run_name, handler):
    """运行单轮，返回 (answer, tool_sequence, tool_returns)"""
    result = app.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={
            "callbacks": [handler],
            "recursion_limit": 15,
            "run_name": run_name,
            "metadata": {"langfuse_tags": ["day6-eval"]},
        },
    )
    answer = result["messages"][-1].content
    seq = " -> ".join(tool_seq(result["messages"])) or "(未调用工具)"
    returns = tool_returns(result["messages"])
    return answer, seq, returns, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条用例")
    parser.add_argument("--prompt-version", default="v2", help="使用的 prompt 版本，默认 v2")
    args = parser.parse_args()

    with open(CASES_PATH, encoding="utf-8") as f:
        cases = json.load(f)
    if args.limit is not None:
        cases = cases[: args.limit]

    app = build_app(CSV_PATH, prompt_version=args.prompt_version)  # 不传 checkpointer，无持久化记忆

    rows = []
    for case in cases:
        cid = case["case_id"]
        rounds = case["rounds"]
        ctype = case["type"]

        if len(rounds) == 1:
            question = rounds[0]
            handler = CallbackHandler()
            print(f"[{cid}] 运行中：{question}")
            try:
                answer, seq, returns, _ = run_one(app, question, cid, handler)
            except Exception as e:
                answer = f"(运行异常: {type(e).__name__})"
                seq = "(异常中断)"
                returns = ""
            rows.append({
                "case_id": cid, "type": ctype, "round": 1,
                "question": question, "tool_sequence": seq,
                "tool_returns": returns, "answer": answer,
                "judge_suggestion": "", "number_audit": "", "manual_assessment": "",
            })
            client = get_client()
            if client is not None:
                client.flush()
        else:
            # 多轮：手动累积消息，两轮各记一行
            prev_messages = None
            for idx, question in enumerate(rounds, start=1):
                sub_id = f"{cid}-R{idx}"
                handler = CallbackHandler()
                print(f"[{sub_id}] 运行中：{question}")
                try:
                    if idx == 1:
                        result = app.invoke(
                            {"messages": [HumanMessage(content=question)]},
                            config={
                                "callbacks": [handler],
                                "recursion_limit": 15,
                                "run_name": sub_id,
                                "metadata": {"langfuse_tags": ["day6-eval"]},
                            },
                        )
                        prev_messages = list(result["messages"])
                    else:
                        # 第二轮：第一轮全部消息 + 新问题
                        result = app.invoke(
                            {"messages": prev_messages + [HumanMessage(content=question)]},
                            config={
                                "callbacks": [handler],
                                "recursion_limit": 15,
                                "run_name": sub_id,
                                "metadata": {"langfuse_tags": ["day6-eval"]},
                            },
                        )
                        prev_messages = list(result["messages"])
                    answer = result["messages"][-1].content
                    seq = " -> ".join(tool_seq(result["messages"])) or "(未调用工具)"
                    returns = tool_returns(result["messages"])
                except Exception as e:
                    answer = f"(运行异常: {type(e).__name__})"
                    seq = "(异常中断)"
                    returns = ""
                rows.append({
                    "case_id": sub_id, "type": ctype, "round": idx,
                    "question": question, "tool_sequence": seq,
                    "tool_returns": returns, "answer": answer,
                    "judge_suggestion": "", "number_audit": "", "manual_assessment": "",
                })
                client = get_client()
                if client is not None:
                    client.flush()

    Path("docs").mkdir(exist_ok=True)
    fieldnames = ["case_id", "type", "round", "question", "tool_sequence",
                  "tool_returns", "answer", "judge_suggestion", "number_audit",
                  "manual_assessment"]
    with open(OUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n完成：{OUT_PATH} 共{len(rows)}行")


if __name__ == "__main__":
    main()
