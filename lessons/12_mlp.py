import torch
import torch.nn as nn

gpu = "cuda"

seed = torch.manual_seed(42)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X,Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

class Tiny_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        W1 = torch.zeros(2,4)
        W1 = nn.init.normal_(W1, mean = 0, std = 0.1)
        self.W1 = nn.Parameter(W1)
        b1 = torch.zeros(4)
        self.b1 = nn.Parameter(b1)
        W2 = torch.zeros(4,1)
        W2 = nn.init.normal_(W2, mean = 0, std = 0.1)
        self.W2 = nn.Parameter(W2)
        b2 = torch.zeros(1)
        self.b2 = nn.Parameter(b2)

        self.ReLU = nn.ReLU()

    def forward(self,X):
        X1 = torch.einsum("ij,jk->ik", X, self.W1) + self.b1
        print(f"Hidden.shape = {X1.shape}")
        Y1 = self.ReLU(X1)
        X2 = torch.einsum("ij,jk->ik", Y1, self.W2) + self.b2
        print(f"Output.shape = {X2.shape}")
        Y2 = X2

        return Y2

model = Tiny_MLP()
model.to(device = gpu)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
loss_fn = nn.MSELoss()

X = torch.tensor([[-1, -1],[-1, 1],[1, -1],[1, 1]]).to(device = gpu, dtype = torch.float32)
Y = torch.tensor([[1], [-1], [-1], [1]]).to(device = gpu, dtype = torch.float32)

dataset = Dataset(X, Y)
loader = torch.utils.data.DataLoader(dataset, batch_size = 2, shuffle = True, drop_last = False, num_workers = 0)

params_sum = 0
for params in model.parameters():
    params_sum += params.numel()
    print(f"params = {params}")

print(f"before update, params_sum = {params_sum}")

for epoch in range(5):
    for idx, batch_data in enumerate(loader):
        X_batch, Y_batch = batch_data
        Y_hat = model(X_batch)

        loss = loss_fn(Y_hat, Y_batch)
        loss.backward()

        # print(f"W1.grad = {model.W1.grad}")
        # print(f"b1.grad = {model.b1.grad}")
        # print(f"W2.grad = {model.W2.grad}")
        # print(f"b2.grad = {model.b2.grad}")

        optimizer.step()
        optimizer.zero_grad()

for params in model.parameters():
    print(f"params = {params}")

print(f"after update, params_sum = {params_sum}")

with torch.no_grad():
    print(f"loss = {loss}")








