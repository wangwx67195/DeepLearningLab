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

w.grad = None
b.grad = None

with torch.no_grad():
    y_no_grad = x * w + b
    print(f"y_o_grad = ", y_no_grad)
    print(f"y_no_grad.shape = ", y_no_grad.shape)
    print(f"y_no_grad.requires_grad = ", y_no_grad.requires_grad)
    print(f"y_no_grad.grad_fn = ", y_no_grad.grad_fn)

w.grad = None
b.grad = None

with torch.inference_mode(mode = True):
    y_infer = x * w + b
    print(f"y_infer = ", y_infer)
    print(f"y_infer.shape = ", y_infer.shape)
    print(f"y_infer.requires_grad = ", y_infer.requires_grad)
    print(f"y_infer.grad_fn = ", y_infer.grad_fn)

y_new = x * w + b
print(f"y_new = ", y_new)
print(f"y_new.shape = ", y_new.shape)
print(f"y_new.requires_grad = ", y_new.requires_grad)
print(f"y_new.grad_fn = ", y_new.grad_fn)




