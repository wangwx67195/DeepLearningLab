import torch
import torch.nn as nn

seed = torch.manual_seed(42)
gpu = "cuda"

class_A = 0.5 * torch.randn(64, 2) + torch.tensor([-2., -2.])
label_A = torch.full((64,), 0, dtype = torch.long)
class_B = 0.5 * torch.randn(64, 2) + torch.tensor([2., -2.])
label_B = torch.full((64,), 1, dtype = torch.long)
class_C = 0.5 * torch.randn(64, 2) + torch.tensor([0., 2.])
label_C = torch.full((64,), 2, dtype = torch.long)

X = torch.cat([class_A, class_B, class_C], dim = 0)
print(f"X.shape = {X.shape}")
Y = torch.cat([label_A, label_B, label_C], dim = 0)
print(f"Y.shape = {Y.shape}")

class classification_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer_1 = nn.Linear(2, 16, bias = True, device = gpu, dtype = torch.float32)
        self.linear_layer_2 = nn.Linear(16, 3, bias = True, device = gpu, dtype = torch.float32)
        self.relu = nn.ReLU()

    def forward(self, input):
        Y1 = self.linear_layer_1(input)
        Y1 = self.relu(Y1)
        Y2 = self.linear_layer_2(Y1)

        return Y2

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

model = classification_model()
model.to(device = gpu)

X = X.to(next(model.parameters()).device)
Y = Y.to(next(model.parameters()).device)

dataset = Dataset(X, Y)
loader = torch.utils.data.DataLoader(dataset, batch_size = 192, shuffle = True)

optimizer = torch.optim.AdamW(model.parameters(), lr = 0.02, weight_decay = 0)
Loss_fn = nn.CrossEntropyLoss()

model.eval()
with torch.no_grad():
    Y_pred = model(X)
    loss = Loss_fn(Y_pred, Y)
    print(f"Epoch: 0, loss = {loss}")
    print(f"Epoch: 0, max_index = {torch.argmax(Y_pred, dim = 1)}")
    print(f"Epoch: 0, true index = {torch.argmax(Y_pred, dim =1) == Y}")
    print(f"Epoch: 0, if sum_softmax == 1: {torch.softmax(Y_pred, dim = 1)}")

    sum_acc = (torch.argmax(Y_pred, dim = 1) == Y).float().sum()
    acc = sum_acc / len(Y_pred)
    print(f"Epoch: 0, accuracy = {acc}")
model.train()

for epoch in range(200):

    for idx, batch_data in enumerate(loader):
        optimizer.zero_grad()

        X_batch, Y_batch = batch_data
        Y_pred = model(X_batch)

        loss = Loss_fn(Y_pred, Y_batch)
        loss.backward()

        optimizer.step()

    if (epoch + 1) % 50 == 0:
        model.eval()
        with torch.no_grad():
            Y_pred = model(X)
            loss = Loss_fn(Y_pred, Y)
            print(f"Epoch: {epoch + 1}, loss = {loss}")
            print(f"Epoch: {epoch + 1}, max_index = {torch.argmax(Y_pred, dim = 1)}")
            print(f"Epoch: {epoch + 1}, true index = {torch.argmax(Y_pred, dim =1 ) == Y}")
            print(f"Epoch: {epoch + 1}, if sum_softmax == 0: {torch.softmax(Y_pred, dim = 1)}")

            sum_acc = (torch.argmax(Y_pred, dim = 1) == Y).float().sum()
            acc = sum_acc / len(Y_pred)
            print(f"Epoch: {epoch + 1}, accuracy = {acc}")

            probs = torch.softmax(Y_pred, dim = 1)
            print(f"Epoch: {epoch + 1}, small softmax = {probs[:5]}")

        model.train()

model.eval()
with torch.no_grad():
    Y_pred = model(X)
    print(f"final 5 pred class = {torch.argmax(Y_pred, dim = 1)[:5]}")
    print(f"final 5 label class = {Y[:5]}")













