import torch

x = torch.tensor([1.,2.,3.],dtype = torch.float32)
y = torch.tensor([2., 4., 6.], dtype = torch.float32)

w = torch.tensor([1.], dtype = torch.float32, requires_grad = True)
b = torch.tensor([0.], dtype = torch.float32, requires_grad = True)

for i in range(0,2):
    y_hat = x * w + b

    loss = torch.mean((y_hat-y)*(y_hat-y))
    loss.backward()

    print(f"loop:{i},w.grad = {w.grad}")
    print(f"loop:{i},b.grad = {b.grad}")

w.grad = None
b.grad = None

for i in range(1):
    y_hat = x * w + b

    loss = torch.mean((y_hat-y)*(y_hat-y))
    loss.backward()

    print(f"grad = None, loop:{i},w.grad = {w.grad}")
    print(f"grad = None, loop:{i},b.grad = {b.grad}")

w.grad = None
b.grad = None

y_hat = x * w + b

y_hat_detached = y_hat.detach()

print(f"y_hat = {y_hat}")
print(f"y_hat.shape = {y_hat.shape}")
print(f"y_hat.requires_grad = {y_hat.requires_grad}")
print(f"y_hat.grad_fn = {y_hat.grad_fn}")

print(f"y_hat_detached = {y_hat_detached}")
print(f"y_hat_detached.shape = {y_hat_detached.shape}")
print(f"y_hat_detached.requires_grad = {y_hat_detached.requires_grad}")
print(f"y_hat_detached.grad_fn = {y_hat_detached.grad_fn}")

loss = torch.mean((y_hat-y)*(y_hat-y))
loss.backward()

# print(f"w.grad = {w.grad}")
# print(f"b.grad = {b.grad}")
# print(f"w.requires_grad = {w.requires_grad}")
# print(f"b.requires_grad = {b.requires_grad}")
# print(f"w.grad_fn = {w.grad_fn}")
# print(f"b.grad_fn = {b.grad_fn}")






