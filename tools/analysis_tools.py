import pandas as pd
from langchain_core.tools import tool
from tools.csv_tools import smart_read_csv


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
