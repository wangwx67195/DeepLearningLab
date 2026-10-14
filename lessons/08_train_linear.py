import torch
import torch.nn as nn

class Linear_model(nn.Module):
    def __init__(self):
        super().__init__()

        W = torch.zeros(1).to(device = "cuda:0", dtype = torch.float32)
        b = torch.zeros(1).to(device = "cuda:0", dtype = torch.float32)
        self.W = nn.Parameter(W)
        self.b = nn.Parameter(b)

    def forward(self,x):
        y_hat = self.W * x + self.b

        return y_hat

x = torch.tensor([-2.,-1.,0.,1.,2.], device = "cuda:0", dtype = torch.float32)
y = torch.tensor([-3.,-1.,1.,3.,5.], device = "cuda:0", dtype = torch.float32)

model = Linear_model()
Loss_fn = nn.MSELoss()
learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(),lr =learning_rate)

for i in range(100):
    y_hat = model(x)

    loss = Loss_fn(y_hat, y)
    loss.backward()

    if i == 0 or i == 99:
        print(f"========== LOOP {i} ==========")
        print(f"Loop: {i}, before update, model.W = {model.W}")
        print(f"Loop: {i}, before update, model.b = {model.b}")

        print(f"========== GRAD ==========")
        print(f"Loop: {i}, before update, model.W.grad = {model.W.grad}")
        print(f"Loop: {i}, before update, model.b.grad = {model.b.grad}")

    optimizer.step()
    optimizer.zero_grad()

    if i == 0 or i == 99:
        print(f"========== LOOP {i} ==========")
        print(f"Loop: {i}, Loss = {loss}")
        print(f"Loop: {i}, after update, model.W = {model.W}")
        print(f"Loop: {i}, after update, model.b = {model.b}")

        print(f"========== GRAD ==========")
        print(f"Loop: {i}, after update, model.W.grad = {model.W.grad}")
        print(f"Loop: {i}, after update, model.b.grad = {model.b.grad}")

with torch.no_grad():
    print(f"final prediction y_hat = {model(x)}")

    




