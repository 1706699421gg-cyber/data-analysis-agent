import argparse
import csv
import json
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

RESULTS_PATH = "docs/eval_results.csv"
CASES_PATH = "docs/eval_cases.json"

SYSTEM_PROMPT = """你是一名严格的评测裁判。下面给出一道数据分析题的：问题、Agent回答、调用的工具序列、工具返回内容、以及该题的合格判据（expect）。

请按以下三档量表打分，只输出一行，格式：<等级> | <一句理由>

量表定义：
- 合格：行为达标，且关键数字可溯源到工具返回，或衍生计算（差值/占比/均值等）正确无误。
- 部分合格：行为基本达标，但存在可归因的小瑕疵，例如数字有误、格式不规范、引用不完整等。
- 不合格：结果错误且无法归因到工具，或行为越界（编造数据、硬答题外话、拒答不当等）。

要求：
1. 理由控制在一句话内，指出关键依据。
2. 优先核对数字是否能在 tool_returns 中找到，或能由 tool_returns 中的数字正确推导。
3. 不要输出多余内容，只输出一行。"""


def build_judge_prompt(question, answer, tool_sequence, tool_returns, expect):
    returns = (tool_returns or "")[:1500]
    return (
        f"【问题】{question}\n"
        f"【合格判据】{expect}\n"
        f"【工具序列】{tool_sequence}\n"
        f"【工具返回】\n{returns}\n"
        f"【Agent回答】\n{answer}\n"
    )


def load_cases():
    with open(CASES_PATH, encoding="utf-8") as f:
        cases = json.load(f)
    # 多轮 case 的 expect 对所有 round 通用；建立 case_id(含 -R) -> expect 映射
    mapping = {}
    for c in cases:
        cid = c["case_id"]
        mapping[cid] = c["expect"]
        if len(c["rounds"]) > 1:
            for i in range(1, len(c["rounds"]) + 1):
                mapping[f"{cid}-R{i}"] = c["expect"]
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=0,
    )

    expect_map = load_cases()

    rows = []
    with open(RESULTS_PATH, encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames
        rows = list(r)

    target = rows if args.limit is None else rows[: args.limit]
    judged = 0
    for row in target:
        if row.get("judge_suggestion", "").strip():
            print(f"[{row['case_id']}] 跳过（已有判分）")
            continue
        cid = row["case_id"]
        expect = expect_map.get(cid, "")
        prompt = build_judge_prompt(
            row["question"], row["answer"], row["tool_sequence"],
            row["tool_returns"], expect,
        )
        try:
            resp = llm.invoke([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ])
            verdict = resp.content.strip().splitlines()[0]
        except Exception as e:
            verdict = f"不合格 | 评卷异常: {type(e).__name__}"
        row["judge_suggestion"] = verdict
        judged += 1
        print(f"[{cid}] {verdict}")

    with open(RESULTS_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n完成：本次新判 {judged} 行，结果已写回 {RESULTS_PATH}")


if __name__ == "__main__":
    main()
