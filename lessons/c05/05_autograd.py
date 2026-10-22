import torch 

x = torch.tensor([1.,2.,3.],device = "cpu", dtype = torch.float32, requires_grad = True)
y = torch.tensor([2.,4.,6.], device = "cpu", dtype = torch.float32, requires_grad = True)

w = torch.tensor([1.], device = "cpu", dtype = torch.float32, requires_grad = True)
b = torch.tensor([0.], device = "cpu", dtype = torch.float32, requires_grad = True)
y_hat = w * x + b
print("y_hat.shape = ", y_hat.shape)

L = torch.mean((y_hat - y)*(y_hat - y), dim = 0)

L.backward()
print("L.requires_grad = ", L.requires_grad)
print("L.is_leaf", L.is_leaf)
print("L.grad_fn", L.grad_fn)
print("L.grad", L.grad)

print("w.grad = ", w.grad)
print("w.requires_grad = ", w.requires_grad)
print("w.is_leaf = ", w.is_leaf)
print("w.grad_fn = ", w.grad_fn)

print("b.grad = ", b.grad)
print("b.requires_grad = ", b.requires_grad)
print("b.is_leaf = ", b.is_leaf)
print("b.grad_fn = ", b.grad_fn)

try: 
    print("x.grad = ", x.grad)
    print("x.requires_grad = ", x.requires_grad)
    print("x.is_leaf = ", x.is_leaf)
    print("x.grad_fn = ", x.grad_fn)

    print("y.grad = ", y.grad)
    print("y.requires_grad = ", y.requires_grad)
    print("y.is_leaf = ", y.is_leaf)
    print("y.grad_fn = ", y.grad_fn)

    print("y_hat.grad = ", y_hat.grad)
    print("y_hat.requires_grad = ", y_hat.requires_grad)
    print("y_hat.is_leaf = ", y_hat.is_leaf)
    print("y_hat.grad_fn = ", y_hat.grad_fn)    

except RuntimeError as err:
    print("Error = ", err)

