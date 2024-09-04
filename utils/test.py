import torch

# 檢查 CUDA 是否可用
print("CUDA available:", torch.cuda.is_available())

# 如果可用，列出可用的 GPU 數量
if torch.cuda.is_available():
    print("Number of GPUs:", torch.cuda.device_count())

    # 列出當前使用的 GPU 名稱
    print("Current GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
