import torch
import torch.nn as nn

gpu = "cuda"

X = torch.tensor([[1,2,4],[2,5,6],[4,3,8],[7,8,2]]).to(dtype = torch.float32, device = gpu)

bn = nn.BatchNorm1d(3)
bn_shape = bn.shape()
bn_running_mean = bn.running_mean.clone()
bn_running_var = bn.running_var.clone()
bn_num_batches_tracked = bn.num_batches_tracked.clone()

ln = nn.LayerNorm(3)

bn.to(gpu)
ln.to(gpu)

bn.train()
ln.train()

with torch.no_grad():
    Y_bn = bn(X)
    Y_ln = ln(X)

print(f"bn_running_mean() = {bn_running_mean}")
print(f"bn_running_var() = {bn_running_var}")
print(f"bn_num_batches_tracked() = {bn_num_batches_tracked}")

