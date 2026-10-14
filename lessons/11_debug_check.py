import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark

gpu = "cuda"
cpu = "cpu"

X = torch.tensor([[-2.],[-1.],[0.],[1.],[2.]], device=cpu, dtype = torch.float64)
Y = torch.tensor([[-3.],[-1.],[1.],[3.],[5.]], device=cpu, dtype = torch.float64)

w = torch.tensor([0.3], device = cpu, dtype = torch.float64, requires_grad = True)
b = torch.tensor([-0.2], device = cpu, dtype = torch.float64, requires_grad = True)

Loss_fn = nn.MSELoss()

def loss_for_check(w, b):
    Y_hat = X * w + b
    assert Y_hat.shape == Y.shape, f"shape must equal, Y_hat.shape = {Y_hat.shape}, Y.shape = {Y.shape}"
    loss = Loss_fn(Y_hat, Y)
    return loss

loss = loss_for_check(w, b)
loss.backward()

print(f"loss isfinite = {torch.isfinite(loss)}")
print(f"w.grad isfinite = {torch.isfinite(w.grad)}")
print(f"b.grad isfinite = {torch.isfinite(b.grad)}")

gradcheck = torch.autograd.gradcheck(loss_for_check, (w, b),eps = 1e-6, atol = 1e-5, rtol = 1e-3)
print(f"gradcheck = {gradcheck}")

print(f"all elements' bytes = {X.numel() * X.element_size()}")

global_dict = {"loss_for_check":loss_for_check, "w":w, "b":b}
timer = benchmark.Timer("loss_for_check(w,b)", globals = global_dict, num_threads = 1)

measurement = timer.blocked_autorange(min_run_time = 0.2) 
print(f"timer = {measurement}")
print(f"timer.median = {measurement.median}")
