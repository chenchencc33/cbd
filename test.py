import torch

# 检查 GPU 是否可用
print("GPU Available:", torch.cuda.is_available())
print("CUDA Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device found")
