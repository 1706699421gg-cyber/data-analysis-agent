from pathlib import Path
import chardet
import pandas as pd
from langchain_core.tools import tool


def smart_read_csv(path: str) -> pd.DataFrame:
    """自动探测编码读CSV。今天当普通函数用，它的价值以后讲"""
    raw = Path(path).read_bytes()
    enc = (chardet.detect(raw[:100000]).get("encoding") or "utf-8").lower()
    if enc in ("gbk", "gb2312", "ascii"):
        enc = "gbk"
    try:
        return pd.read_csv(path, encoding=enc)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8", errors="replace")


@tool
def get_csv_overview(csv_path: str) -> str:
    """读取CSV文件并返回表结构概览：行数、列数、每列的名称/类型/缺失值数量/前2行示例。在分析任何数据之前，必须先调用此工具了解表结构。"""
    df = smart_read_csv(csv_path)
    lines = [f"表形状: {df.shape[0]} 行 x {df.shape[1]} 列"]
    for col in df.columns:
        lines.append(
            f"- {col} | 类型:{df[col].dtype} | 缺失:{df[col].isna().sum()} "
            f"| 示例:{df[col].dropna().head(2).tolist()}"
        )
    return "\n".join(lines)
