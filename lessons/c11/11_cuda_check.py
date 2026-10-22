import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark

gpu = "cuda"

print(f"torch.cuda.memory_allocated = {torch.cuda.memory_allocated(gpu)}")
print(f"torch.cuda.memory_reserved = {torch.cuda.memory_reserved(gpu)}")

A = torch.randn(512, 512).to(device = gpu, dtype = torch.float32)
B = torch.randn(512, 512).to(device = gpu, dtype = torch.float32)

C = torch.matmul(A, B)
print(f"C.shape = {C.shape}")

print(f"A.all_bytes = {A.numel() * A.element_size()}")
print(f"B.all_bytes = {B.numel() * B.element_size()}")
print(f"C.all_bytes = {C.numel() * C.element_size()}")

print(f"torch.cuda.memory_allocated = {torch.cuda.memory_allocated(gpu)}")
print(f"torch.cuda.memory_reserved = {torch.cuda.memory_reserved(gpu)}")

del C
torch.cuda.synchronize(gpu)

print(f"torch.cuda.memory_allocated = {torch.cuda.memory_allocated(gpu)}")
print(f"torch.cuda.memory_reserved = {torch.cuda.memory_reserved(gpu)}")

torch.cuda.empty_cache()
print(f"torch.cuda.memory_allocated = {torch.cuda.memory_allocated(gpu)}")
print(f"torch.cuda.memory_reserved = {torch.cuda.memory_reserved(gpu)}")

global_dict = {"A":A, "B":B}
timer = benchmark.Timer("torch.matmul(A,B)", globals = global_dict, num_threads=1)
measurement = timer.blocked_autorange(min_run_time = 0.2)
median = measurement.median
print(f"median = {median}")




