import re
import csv
from pathlib import Path

CSV_PATH = "docs/eval_results.csv"


def normalize(text):
    """去掉中英文逗号和空格，便于数字溯源比对"""
    if not text:
        return ""
    return (text.replace(",", "")
                .replace("，", "")
                .replace(" ", "")
                .replace("\u3000", "")
                .replace("\n", "")
                .replace("\t", ""))


def extract_numbers(text):
    """从文本中提取整数部分 >= 3 位的数字，过滤百分比噪音"""
    nums = []
    # 匹配带千分位的数字（如 116,132）或纯数字（>=3 位）
    pattern = re.compile(r'(\d{1,3}(?:,\d{3})+|\d{3,})')
    for m in pattern.finditer(text or ""):
        raw = m.group(0)
        num_str = raw.replace(",", "")
        # 百分比直接跳过（如 40.7%、10%）
        end = m.end()
        if end < len(text) and text[end] in "%％":
            continue
        if len(num_str) >= 3:
            nums.append(num_str)
    return nums


def audit_row(answer, tool_returns):
    norm_returns = normalize(tool_returns)
    nums = extract_numbers(answer)
    if not nums:
        return "OK"
    missed = [n for n in nums if n not in norm_returns]
    if not missed:
        return "OK"
    return "待人工判读: " + ", ".join(missed)


def main():
    rows = []
    fieldnames = None
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames
        for row in r:
            row["number_audit"] = audit_row(row.get("answer", ""), row.get("tool_returns", ""))
            rows.append(row)

    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    ok = sum(1 for r in rows if r["number_audit"] == "OK")
    total = len(rows)
    print(f"审计完成：{CSV_PATH}，共 {total} 行，OK {ok} 行，待人工判读 {total - ok} 行")
    print("说明：本脚本只是筛选器，衍生数字（差值、占比、均值等）的正确性需由人工判定。")


if __name__ == "__main__":
    main()
