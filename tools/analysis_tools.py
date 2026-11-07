import pandas as pd
from langchain_core.tools import tool
from tools.csv_tools import smart_read_csv
import operator
from pathlib import Path

@tool
def describe_column(csv_path: str, column: str) -> str:
    """对指定数值列做描述统计：计数/均值/标准差/最小值/四分位数/最大值。column 必须是CSV中真实存在的列名。"""
    df = smart_read_csv(csv_path)
    if column not in df.columns:
        return f"错误：列 '{column}' 不存在。可用列：{list(df.columns)}"
    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"错误：列 '{column}' 不是数值列（实际类型 {df[column].dtype}），无法做描述统计"
    return df[column].describe().to_string()


@tool
def groupby_aggregate(csv_path: str, group_col: str, value_col: str, agg: str) -> str:
    """按 group_col 分组，对 value_col 做聚合统计。agg 只能是 'sum'/'mean'/'max'/'min'/'count' 之一。返回每个分组的统计结果，按值降序排列。"""
    df = smart_read_csv(csv_path)
    for c, name in [(group_col, "group_col"), (value_col, "value_col")]:
        if c not in df.columns:
            return f"错误：{name}='{c}' 不存在。可用列：{list(df.columns)}"
    if agg not in ("sum", "mean", "max", "min", "count"):
        return f"错误：agg 只支持 sum/mean/max/min/count，收到 '{agg}'"
    if agg != "count" and not pd.api.types.is_numeric_dtype(df[value_col]):
        return f"错误：value_col='{value_col}' 不是数值列，无法做 {agg} 聚合"
    result = df.groupby(group_col)[value_col].agg(agg).sort_values(ascending=False)
    return result.to_string()

@tool
def recall_history(query: str) -> str:
    """回忆历史分析会话的结论。当用户提到'之前''上次''对比一下'等涉及过往分析的内容时，调用此工具。"""
    from memory.vector_store import recall
    results = recall(query)
    return "\n---\n".join(results) if results else "没有找到相关的历史分析记录"

@tool
def filter_and_stats(csv_path: str, column: str, op: str, value: float, stat_col: str) -> str:
    """条件筛选后统计：筛选 column 满足条件的行（op 只能是 '>'/'<'/'=='/'>='/'<='），对 stat_col 求总和与均值。column 和 stat_col 都必须是数值列。"""
    ops = {">": operator.gt, "<": operator.lt, "==": operator.eq,
           ">=": operator.ge, "<=": operator.le}
    df = smart_read_csv(csv_path)
    for c in (column, stat_col):
        if c not in df.columns:
            return f"错误：列 '{c}' 不存在。可用列：{list(df.columns)}"
    if op not in ops:
        return f"错误：op 只支持 > < == >= <=，收到 '{op}'"
    try:
        sub = df[ops[op](df[column], value)]
    except TypeError:
        return f"错误：列 '{column}' 不是数值列，无法与数字比较"
    if sub.empty:
        return f"筛选后 0 行（条件：{column} {op} {value}），请调整筛选条件"
    return (f"匹配 {len(sub)} 行\n"
            f"{stat_col} 总和: {sub[stat_col].sum():.2f}\n"
            f"{stat_col} 均值: {sub[stat_col].mean():.2f}")

@tool
def plot_bar(csv_path: str, group_col: str, value_col: str, title: str) -> str:
    """绘制柱状图：按 group_col 分组统计 value_col 之和，保存到 outputs/ 目录并返回图片路径。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    df = smart_read_csv(csv_path)
    if group_col not in df.columns or value_col not in df.columns:
        return f"错误：可用列为 {list(df.columns)}"
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        return f"错误: value_col='{value_col}' 不是数值列，无法求和"
    data = df.groupby(group_col)[value_col].sum().sort_values(ascending=False)
    ax = data.plot(kind="bar", figsize=(10, 6))
    ax.set_title(title)
    ax.set_ylabel(value_col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    Path("outputs").mkdir(exist_ok=True)
    safe_name = "".join(ch for ch in title[:20] if ch not in '\\/:*?"<>|')
    out = f"outputs/{safe_name}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    return f"图表已保存: {out}，共 {len(data)} 个分组"
