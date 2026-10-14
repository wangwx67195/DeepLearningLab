import torch 
import torch.nn as nn

X = torch.tensor([[-2.],[-1.],[0.],[1.],[2.]], device = "cpu", dtype = torch.float32)
Y = torch.tensor([[-3.],[-1.],[1.],[3.],[5.]], device = "cpu", dtype = torch.float32)

gpu = "cuda:0"

class Dataset(torch.utils.data.Dataset):
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

dataset = Dataset(X, Y)

print(f"len(dataset) = {len(dataset)}")
print(f"__getitem__(self, 2) = {dataset[2]}")

loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = 2, shuffle = True, drop_last = False, num_workers = 0)
print(f"len(loader) = {len(loader)}")

class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        W = torch.zeros(1)
        self.W = nn.Parameter(W)
        b = torch.zeros(1)
        self.b = nn.Parameter(b)

    def forward(self, X):
        Y_hat = X * self.W + self.b
        return Y_hat

model = Linear()
model.to(device = gpu)
print(next(model.parameters()).device)

optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
Loss_fn = nn.MSELoss()

for epoch in range(100):

    for idx, batch_data in enumerate(loader):
        x_batch, y_batch = batch_data
        x_batch = x_batch.to(device = gpu)
        y_batch = y_batch.to(device = gpu)

        if epoch < 3:
            print(f"Epoch {epoch}, load {idx}, input = {x_batch}, target = {y_batch}")

        y_hat_batch = model(x_batch)
        loss = Loss_fn(y_hat_batch, y_batch)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

with torch.no_grad():
    X_eval = X.to(device = gpu)
    Y_eval = Y.to(device = gpu)
    print(f"final output = {model(X_eval)}")

    final_loss = Loss_fn(model(X_eval), Y_eval)
    print(f"final loss = {final_loss}")


