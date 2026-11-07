import csv
from collections import defaultdict
from pathlib import Path

RESULTS_PATH = "docs/eval_results.csv"
REPORT_PATH = "docs/eval_report.md"


def parse_verdict(judge_suggestion):
    """从 judge_suggestion 中提取等级（合格/部分合格/不合格）"""
    s = (judge_suggestion or "").strip()
    for v in ("合格", "部分合格", "不合格"):
        if s.startswith(v):
            return v
    return "未判分"


def main():
    rows = []
    with open(RESULTS_PATH, encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        rows = list(r)

    total = len(rows)
    by_type = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row)

    def pass_rate(rs):
        if not rs:
            return 0, 0, 0
        ok = sum(1 for r in rs if parse_verdict(r["judge_suggestion"]) == "合格")
        return ok, len(rs), ok / len(rs) * 100

    lines = ["# 评测报告", ""]
    ok, n, rate = pass_rate(rows)
    lines.append(f"## 总览")
    lines.append(f"- 总用例数：{n}")
    lines.append(f"- 合格数：{ok}")
    lines.append(f"- 总通过率：{rate:.1f}%")
    lines.append("")

    lines.append("## 按类型分组通过率")
    lines.append("| 类型 | 合格 / 总数 | 通过率 |")
    lines.append("|---|---|---|")
    for t in ("正向", "边界", "多轮"):
        rs = by_type.get(t, [])
        ok_t, n_t, rate_t = pass_rate(rs)
        lines.append(f"| {t} | {ok_t} / {n_t} | {rate_t:.1f}% |")
    lines.append("")

    lines.append("## 非合格用例清单")
    non_pass = [r for r in rows if parse_verdict(r["judge_suggestion"]) != "合格"]
    if not non_pass:
        lines.append("（无）")
    else:
        lines.append("| case_id | 等级 | judge 理由 |")
        lines.append("|---|---|---|")
        for r in non_pass:
            v = parse_verdict(r["judge_suggestion"])
            reason = (r["judge_suggestion"] or "").split("|", 1)
            reason_text = reason[1].strip() if len(reason) > 1 else r["judge_suggestion"]
            lines.append(f"| {r['case_id']} | {v} | {reason_text} |")
    lines.append("")

    Path("docs").mkdir(exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"报告已生成：{REPORT_PATH}")


if __name__ == "__main__":
    main()
