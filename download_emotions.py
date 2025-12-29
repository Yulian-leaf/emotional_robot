from datasets import load_dataset

# 加载数据集（会自动下载到 datasets 缓存）
ds = load_dataset("Conna/eMotions")

# 保存为本地可重用格式
ds.save_to_disk("eMotions_local")
print("Saved to eMotions_local")