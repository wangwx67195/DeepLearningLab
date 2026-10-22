import torch
import torch.nn as nn

seed = torch.manual_seed(42)
gpu = "cuda"

def calculate_param_norm(model):
    with torch.no_grad():
        return torch.sqrt(sum(p.square().sum() for p in model.parameters())).item()

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
        Y1 = self.ReLU(X1)
        X2 = torch.einsum("ij,jk->ik", Y1, self.W2) + self.b2
        Y2 = X2

        return Y2

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

X = torch.tensor([[-1, -1],[-1, 1],[1, -1],[1, 1]]).to(device = gpu, dtype = torch.float32)
Y = torch.tensor([[1], [-1], [-1], [1]]).to(device = gpu, dtype = torch.float32)

dataset = Dataset(X, Y)
loader = torch.utils.data.DataLoader(dataset, batch_size = 4, shuffle = False, drop_last = False, num_workers = 0)

loss_fn = nn.MSELoss()

seed = torch.manual_seed(42)
model_A = Tiny_MLP()
seed = torch.manual_seed(42)
model_B = Tiny_MLP()
for idx, (p1, p2) in enumerate(zip(model_A.parameters(), model_B.parameters())):
    print(f"idx = {idx}, equal = {torch.equal(p1, p2)}")
seed = torch.manual_seed(42)
model_C = Tiny_MLP()
for idx, (p1, p2) in enumerate(zip(model_A.parameters(), model_C.parameters())):
    print(f"idx = {idx}, equal = {torch.equal(p1, p2)}")
seed = torch.manual_seed(42)
model_D = Tiny_MLP()
for idx, (p1, p2) in enumerate(zip(model_A.parameters(), model_D.parameters())):
    print(f"idx = {idx}, equal = {torch.equal(p1, p2)}")
model_A.to(device = gpu)
model_B.to(device = gpu)
model_C.to(device = gpu)
model_D.to(device = gpu)

#exp A

optimizer_A = torch.optim.SGD(model_A.parameters(), lr = 0.05)
loss_history_A = []
lr_history_A = []

with torch.no_grad():
    loss_init_A = loss_fn(model_A(X), Y)
    loss_history_A.append(loss_init_A.item())
for epoch in range(200):

    for idx, batch_data in enumerate(loader):
        X_batch, Y_batch = batch_data
        Y_hat = model_A(X_batch)

        loss = loss_fn(Y_hat, Y_batch)
        loss.backward()

        optimizer_A.step()
        optimizer_A.zero_grad()

    with torch.no_grad():
        Y_hat = model_A(X_batch)
        loss = loss_fn(Y_hat, Y_batch)
        loss_history_A.append(loss.item())
        lr_history_A.append(optimizer_A.param_groups[0]['lr'])
        if (epoch + 1) % 50 == 0:
            print(f"Epoch: {epoch}, update_Epoch: {epoch + 1}, loss: {loss}")

norm_A = calculate_param_norm(model_A)
print(f"norm_A = {norm_A}")

# print(f"len(loss_history_A = {len(loss_history_A)})")
# print(f"loss history A = {loss_history_A}")
# print(f"len(lr_history_A = {len(lr_history_A)})")
# print(f"lr history A = {lr_history_A}")
        
#exp B

optimizer_B = torch.optim.AdamW(
    model_B.parameters(),
    lr = 0.05,
    weight_decay = 0,
)
loss_history_B = []
lr_history_B = []

with torch.no_grad():
    loss_init_B = loss_fn(model_B(X), Y)
    loss_history_B.append(loss_init_B.item())
for epoch in range(200):

    for idx, batch_data in enumerate(loader):
        X_batch, Y_batch = batch_data
        Y_hat = model_B(X_batch)

        loss = loss_fn(Y_hat, Y_batch)
        loss.backward()

        optimizer_B.step()
        optimizer_B.zero_grad()

    with torch.no_grad():
        Y_hat = model_B(X_batch)
        loss = loss_fn(Y_hat, Y_batch)
        loss_history_B.append(loss.item())
        lr_history_B.append(optimizer_B.param_groups[0]['lr'])
        if (epoch + 1) % 50 == 0:
            print(f"Epoch: {epoch}, update_Epoch: {epoch + 1}, loss: {loss}")

norm_B = calculate_param_norm(model_B)
print(f"norm_B = {norm_B}")

# print(f"len(loss_history_B = {len(loss_history_B)})")
# print(f"loss history B = {loss_history_B}")
# print(f"len(lr_history_B = {len(lr_history_B)})")
# print(f"lr history B = {lr_history_B}")

#exp C

optimizer_C = torch.optim.AdamW(
    model_C.parameters(),
    lr = 0.05,
    weight_decay = 0.1
)
loss_history_C = []
lr_history_C = []

with torch.no_grad():
    loss_init_C = loss_fn(model_C(X), Y)
    loss_history_C.append(loss_init_C.item())
for epoch in range(200):

    for idx, batch_data in enumerate(loader):
        X_batch, Y_batch = batch_data
        Y_hat = model_C(X_batch)

        loss = loss_fn(Y_hat, Y_batch)
        loss.backward()

        optimizer_C.step()
        optimizer_C.zero_grad()

    with torch.no_grad():
        Y_hat = model_C(X_batch)
        loss = loss_fn(Y_hat, Y_batch)
        loss_history_C.append(loss.item())
        lr_history_C.append(optimizer_C.param_groups[0]['lr'])
        if (epoch + 1) % 50 == 0:
            print(f"Epoch: {epoch}, update_Epoch: {epoch + 1}, loss: {loss}")

norm_C = calculate_param_norm(model_C)
print(f"norm_C = {norm_C}")

# print(f"len(loss_history_C = {len(loss_history_C)})")
# print(f"loss history C = {loss_history_C}")
# print(f"len(lr_history_C = {len(lr_history_C)})")
# print(f"lr history C = {lr_history_C}")

#exp D

optimizer_D = torch.optim.AdamW(
    model_D.parameters(),
    lr = 0.05,
    weight_decay = 0.1
)
scheduler_D = torch.optim.lr_scheduler.StepLR(
    optimizer_D,
    step_size = 50,
    gamma =0.5
)
loss_history_D = []
lr_history_D = []

with torch.no_grad():
    loss_init_D = loss_fn(model_D(X), Y)
    loss_history_D.append(loss_init_D.item())
for epoch in range(200):

    lr_history_D.append(optimizer_D.param_groups[0]['lr'])
    for idx, batch_data in enumerate(loader):
        X_batch, Y_batch = batch_data
        Y_hat = model_D(X_batch)

        loss = loss_fn(Y_hat, Y_batch)
        loss.backward()

        optimizer_D.step()
        optimizer_D.zero_grad()

    scheduler_D.step()

    with torch.no_grad():
        Y_hat = model_D(X_batch)
        loss = loss_fn(Y_hat, Y_batch)
        loss_history_D.append(loss.item())
        if (epoch + 1) % 50 == 0:
            print(f"Epoch: {epoch}, update_Epoch: {epoch + 1}, loss: {loss}, lr: {optimizer_D.param_groups[0]['lr']}")

norm_D = calculate_param_norm(model_D)
print(f"norm_D = {norm_D}")

# print(f"len(loss_history_D = {len(loss_history_D)})")
# print(f"loss history D = {loss_history_D}")
# print(f"len(lr_history_D = {len(lr_history_D)})")
# print(f"lr history D = {lr_history_D}")

    




