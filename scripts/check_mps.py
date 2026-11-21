import torch
import time

# Checking the available backend
gpu_available = torch.backends.mps.is_available()
print(f"GPU available: {gpu_available}")

# Initiating a matrix of 4096*4096 dimension
matrix = torch.rand(4096, 4096, device="cpu")
print(f"Matrix shape: {matrix.shape}")

# Running the matrix multiplication on CPU and print the time taken
start_time = time.monotonic()
result = matrix @ matrix
end_time = time.monotonic()
print(f"Time taken: {(end_time - start_time) * 1000} milliseconds")

# Running the matrix multiplication on GPU and print the time taken
start_time = time.monotonic()
result_gpu = matrix.to("mps") @ matrix.to("mps")
end_time = time.monotonic()
print(f"Time taken: {(end_time - start_time) * 1000} milliseconds")