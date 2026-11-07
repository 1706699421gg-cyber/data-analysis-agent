import pandas as pd
df = pd.read_csv("data/sales.csv")
print("Q1 总销售额:", df["销售额"].sum())
print("Q2 各产品总额:\n", df.groupby("产品")["销售额"].sum().sort_values(ascending=False))
