import torch

# Checking the available backend
gpu_available = torch.backends.mps.is_available()
print(f"GPU available: {gpu_available}")