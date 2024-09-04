from datasets import load_dataset

# 加载数据集
ds = load_dataset("heegyu/bbq", "Gender_identity")

# 查看test集中前5个样本
print(ds['test'][:5])

# 查看某一列的前5个样本，例如 'question' 列
print(ds['test']['question'][:5])

# 查看每列的所有值
for column in ds['test'].column_names:
    print(f"{column}: {ds['test'][column][:5]}")

# 查看test集的标签分布
from collections import Counter
label_distribution = Counter(ds['test']['label'])
print(label_distribution)
