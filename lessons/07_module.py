import torch
import torch.nn as nn

class TinyLinear(nn.Module):
    def __init__(self):
        super().__init__()

        W = torch.tensor([[1.,0.],[0.,1.],[1.,1.]], device = "cpu", dtype = torch.float32)
        b = torch.tensor([0.5, -0.5], device = "cpu", dtype = torch.float32)
        self.W = nn.Parameter(W)
        self.b = nn.Parameter(b)

    def forward(self,X):
        Y_hat = torch.einsum("ij,jk->ik", X, self.W) + self.b

        print(f"self.W.grad = ", self.W.grad)
        print(f"self.W.grad.requires_grad = ", self.W.requires_grad)
        print(f"self.b.grad = ", self.b.grad)
        print(f"self.b.requires_grad = ", self.b.requires_grad)

        return Y_hat

model = TinyLinear()

X = torch.arange(1,7).to(dtype = torch.float32)
X = X.reshape(2,3)
Y = torch.zeros(2,2).to(dtype = torch.float32)
Y_hat = model(X)

loss = torch.mean(torch.square(Y_hat - Y))
loss.backward()

print(f"model.W.grad = ", model.W.grad)
print(f"model.b.grad = ", model.b.grad)

print(f"Y_hat = ", Y_hat)
print(f"Y_hat.shape = ", Y_hat.shape)
print(f"Y_hat.requires_grad = ", Y_hat.requires_grad)
print(f"Y_hat.grad_fn", Y_hat.grad_fn)

        




