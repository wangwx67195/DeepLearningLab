import torch
import torch.nn as nn

seed = torch.manual_seed(42)
cpu = "cpu"

X = torch.tensor([[-2.],[-1.],[0.],[1.],[2.]], device=cpu, dtype = torch.float32)
Y = torch.tensor([[-3.],[-1.],[1.],[3.],[5.]], device=cpu, dtype = torch.float32)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x;
        self.y = y;

    def __len__(self):
        return len(self.x)

    def __getitem__(self,idx):
        return (self.x[idx], self.y[idx])

dataset = Dataset(X, Y)
loader = torch.utils.data.DataLoader(dataset, batch_size = 2, shuffle = True, drop_last = False, num_workers = 0)

class Linear_model(nn.Module):
    def __init__(self):
        super().__init__()
        W = torch.zeros(1)
        W = nn.init.normal_(W, mean=0.0, std=0.1, generator = None)
        self.W = nn.Parameter(W)
        b = torch.zeros(1)
        self.b = nn.Parameter(b)

    def forward(self, x):
        y_hat = x * self.W + self.b
        return y_hat

model = Linear_model()
model.to(device=cpu)

optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
Loss_fn = nn.MSELoss()

print(f"initial model.W = {model.W}")

for epoch in range(100):

    for idx, batch_data in enumerate(loader):
        x_batch, y_batch = batch_data
        x_batch = x_batch.to(cpu)
        y_batch = y_batch.to(cpu)
        y_batch_hat = model(x_batch)

        loss = Loss_fn(y_batch_hat, y_batch)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

torch.save(model.state_dict(), "linear_c10.pt")

with torch.no_grad():
    state_dict1 = torch.load("linear_c10.pt", map_location = cpu, weights_only = True)
    model.load_state_dict(state_dict1, strict = True)

    y_hat1 = model(X)

model_copy = Linear_model()
model_no_trained = Linear_model()

with torch.inference_mode():
    state_dict2 = torch.load("linear_c10.pt", map_location = cpu, weights_only = True)
    model_copy.load_state_dict(state_dict2, strict = True)

    y_hat2 = model_copy(X)
    y_hat3 = model_no_trained(X)

allocate_1 = torch.allclose(y_hat1, y_hat2)
allclose_2 = torch.allclose(y_hat1, y_hat3)
print(f"torch.allclose copy = {allocate_1}")
print(f"torch.allclose no_trained = {allclose_2}")



