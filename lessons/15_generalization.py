import torch
import torch.nn as nn
import copy

gpu = "cuda"
cpu = "cpu"

seed = torch.manual_seed(42)
class_A = 1.2 * torch.randn(64, 2) + torch.tensor([-2., -2.])
label_A = torch.full((64,), 0, dtype = torch.long)
indice_A = torch.randperm(len(class_A))
indice_A_isin = indice_A

# A_train_indice = indice_A[:12]
# A_eval_indice = indice_A[12:38]
# A_test_indice = indice_A[38:]

class_train_A = class_A[indice_A][:12]
label_train_A = label_A[indice_A][:12]
class_eval_A = class_A[indice_A][12:38]
label_eval_A = label_A[indice_A][12:38]
class_test_A = class_A[indice_A][38:]
label_test_A = label_A[indice_A][38:]

class_B = 1.2 * torch.randn(64, 2) + torch.tensor([2., -2.])
label_B = torch.full((64,), 1, dtype = torch.long)
indice_B = torch.randperm(len(class_B))
indice_B_isin = indice_B + 64

# B_train_indice = indice_B[:12]
# B_eval_indice = indice_B[12:38]
# B_test_indice = indice_B[38:]

class_train_B = class_B[indice_B][:12]
label_train_B = label_B[indice_B][:12]
class_eval_B = class_B[indice_B][12:38]
label_eval_B = label_B[indice_B][12:38]
class_test_B = class_B[indice_B][38:]
label_test_B = label_B[indice_B][38:]

class_C = 1.2 * torch.randn(64, 2) + torch.tensor([0., 2.])
label_C = torch.full((64,), 2, dtype = torch.long)
indice_C = torch.randperm(len(class_C))
indice_C_isin = indice_C + 128

# C_train_indice = indice_C[:12]
# C_eval_indice = indice_C[12:38]
# C_test_indice = indice_C[38:]

class_train_C = class_C[indice_C][:12]
label_train_C = label_C[indice_C][:12]
class_eval_C = class_C[indice_C][12:38]
label_eval_C = label_C[indice_C][12:38]
class_test_C = class_C[indice_C][38:]
label_test_C = label_C[indice_C][38:]

train_indice = torch.cat([indice_A_isin[:12], indice_B_isin[:12], indice_C_isin[:12]])
eval_indice = torch.cat([indice_A_isin[12:38], indice_B_isin[12:38], indice_C_isin[12:38]])
test_indice = torch.cat([indice_A_isin[38:], indice_B_isin[38:], indice_C_isin[38:]])

print(f"torch.isin() = {torch.isin(train_indice, eval_indice).any()}")
print(f"torch.isin() = {torch.isin(train_indice, test_indice).any()}")
print(f"torch.isin() = {torch.isin(eval_indice, test_indice).any()}")

class_train = torch.cat([class_train_A, class_train_B, class_train_C], dim = 0).to(device = gpu)
label_train = torch.cat([label_train_A, label_train_B, label_train_C], dim = 0).to(device = gpu)
class_eval = torch.cat([class_eval_A, class_eval_B, class_eval_C], dim = 0).to(device = gpu)
label_eval = torch.cat([label_eval_A, label_eval_B, label_eval_C], dim = 0).to(device = gpu)
class_test = torch.cat([class_test_A, class_test_B, class_test_C], dim = 0).to(device = gpu)
label_test = torch.cat([label_test_A, label_test_B, label_test_C], dim = 0).to(device = gpu)

# class_train = class_all[indices_all][:36]
# class_eval = class_all[indices_all][36:114]
# class_test = class_all[indices_all][114:192]
# label_train = label_all[indices_all][:36]
# label_eval = label_all[indices_all][36:114]
# label_test = label_all[indices_all][114:192]

class Dataset(torch.utils.data.Dataset):
    def __init__(self, input, target):
        self.input = input
        self.target = target

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return (self.input[idx], self.target[idx])

train_dataset = Dataset(class_train, label_train)
eval_dataset = Dataset(class_eval, label_eval)
test_dataset = Dataset(class_test, label_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 36, shuffle = True)
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size = 78, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 78, shuffle = True)

class Linear_model_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = nn.Linear(2, 8)
        self.L2 = nn.Linear(8, 3)
        self.relu = nn.ReLU()

    def forward(self, input):
        Y1 = self.L1(input)
        Y1 = self.relu(Y1)
        Y2 = self.L2(Y1)

        return Y2

class Linear_model_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = nn.Linear(2, 128)
        self.L2 = nn.Linear(128, 3)
        self.relu = nn.ReLU()

    def forward(self, input):
        Y1 = self.L1(input)
        Y1 = self.relu(Y1)
        Y2 = self.L2(Y1)

        return Y2

seed = torch.manual_seed(123)
model_1 = Linear_model_1()
model_1.to(gpu)

seed = torch.manual_seed(123)
model_2 = Linear_model_2()
model_2.to(gpu)

optimizer_1 = torch.optim.AdamW(model_1.parameters(), lr = 0.01, weight_decay = 0)
optimizer_2 = torch.optim.AdamW(model_2.parameters(), lr = 0.01, weight_decay = 0)

Loss_fn = nn.CrossEntropyLoss()

def evaluate(model, dataset, loss_fn):
    model.eval()
    with torch.no_grad():
        input, target = dataset.input, dataset.target
        logits = model(input)
        loss = loss_fn(logits, target)
        acc = (torch.argmax(logits, dim = 1) == target).float().mean()

        return (loss, acc)

train_1_loss_list = []
train_1_acc_list = []
val_1_loss_list = [] 
val_1_acc_list = []  
best_val_1_loss = float("inf")

with torch.no_grad():

        loss_eval, acc_eval = evaluate(model_1, eval_dataset, Loss_fn)
        loss_eval, acc_eval = loss_eval.item(), acc_eval.item()
        loss_train, acc_train = evaluate(model_1, train_dataset, Loss_fn)
        loss_train, acc_train = loss_train.item(), acc_train.item()

        val_1_loss_list.append(loss_eval)
        val_1_acc_list.append(acc_eval)
        train_1_loss_list.append(loss_train)
        train_1_acc_list.append(acc_train)

        if loss_eval < best_val_1_loss:
            best_val_1_loss = loss_eval
            best_state_1 = copy.deepcopy(model_1.state_dict())
            best_1 = {"model_state_dict": best_state_1, "epoch": 0}
            torch.save(best_1, "model_1.pt")
            best_val_1_loss = loss_eval

model_1.train()

for epoch in range(400):

    for idx, batch_data in enumerate(train_loader):
        input_batch, tgt_batch = batch_data

        optimizer_1.zero_grad()

        pred_batch = model_1(input_batch)
        loss = Loss_fn(pred_batch, tgt_batch)
        loss.backward()

        optimizer_1.step()

    model_1.eval()
    with torch.no_grad():
        train_1_loss, train_1_acc = evaluate(model_1, train_dataset, Loss_fn)
        val_1_loss, val_1_acc = evaluate(model_1, eval_dataset, Loss_fn)
        train_1_loss = train_1_loss.item()
        train_1_acc = train_1_acc.item()
        val_1_loss = val_1_loss.item()
        val_1_acc = val_1_acc.item()

        if (epoch + 1) % 20 == 0:
            if val_1_loss < best_val_1_loss:
                best_val_1_loss = val_1_loss
                best_epoch_1 = epoch + 1
                best_state_1 = copy.deepcopy(model_1.state_dict())
                best_1 = {"model_state_dict": best_state_1, "epoch": best_epoch_1}
                torch.save(best_1, "model_1.pt")

            train_1_loss_list.append(train_1_loss)
            train_1_acc_list.append(train_1_acc)
            val_1_loss_list.append(val_1_loss)
            val_1_acc_list.append(val_1_acc)

    model_1.train()

train_2_loss_list = []
train_2_acc_list = []
val_2_loss_list = [] 
val_2_acc_list = []   
best_val_2_loss = float("inf")

with torch.no_grad():
        loss_eval, acc_eval = evaluate(model_2, eval_dataset, Loss_fn)
        loss_eval, acc_eval = loss_eval.item(), acc_eval.item()
        loss_train, acc_train = evaluate(model_2, train_dataset, Loss_fn)
        loss_train, acc_train = loss_train.item(), acc_train.item()

        val_2_loss_list.append(loss_eval)
        val_2_acc_list.append(acc_eval)
        train_2_loss_list.append(loss_train)
        train_2_acc_list.append(acc_train)

        if loss_eval < best_val_2_loss:
            best_val_2_loss = loss_eval
            best_state_2 = copy.deepcopy(model_2.state_dict())
            best_2 = {"model_state_dict": best_state_2, "epoch": 0}
            torch.save(best_2, "model_2.pt")
            best_val_2_loss = loss_eval

model_2.train()

for epoch in range(400):

    for idx, batch_data in enumerate(train_loader):
        input_batch, tgt_batch = batch_data

        optimizer_2.zero_grad()

        pred_batch = model_2(input_batch)
        loss = Loss_fn(pred_batch, tgt_batch)
        loss.backward()

        optimizer_2.step()

    model_2.eval()
    with torch.no_grad():
        train_2_loss, train_2_acc = evaluate(model_2, train_dataset, Loss_fn)
        val_2_loss, val_2_acc = evaluate(model_2, eval_dataset, Loss_fn)
        train_2_loss = train_2_loss.item()
        train_2_acc = train_2_acc.item()
        val_2_loss = val_2_loss.item()
        val_2_acc = val_2_acc.item()

        if (epoch + 1) % 20 == 0:
            if val_2_loss < best_val_2_loss:
                best_val_2_loss = val_2_loss
                best_epoch_2 = epoch + 1
                best_state_2 = copy.deepcopy(model_2.state_dict())
                best_2 = {"model_state_dict": best_state_2, "epoch": best_epoch_2}
                torch.save(best_2, "model_2.pt")

            train_2_loss_list.append(train_2_loss)
            train_2_acc_list.append(train_2_acc)
            val_2_loss_list.append(val_2_loss)
            val_2_acc_list.append(val_2_acc)

    model_2.train()

print(f"train_1_loss_list: {train_1_loss_list}")
print(f"train_1_acc_list: {train_1_acc_list}")
print(f"val_1_loss_list: {val_1_loss_list}")
print(f"val_1_acc_list: {val_1_acc_list}")

print(f"train_2_loss_list: {train_2_loss_list}")
print(f"train_2_acc_list: {train_2_acc_list}")
print(f"val_2_loss_list: {val_2_loss_list}")
print(f"val_2_acc_list: {val_2_acc_list}")


with torch.no_grad():
    if best_val_1_loss <= best_val_2_loss:
        max_model_index = 1
        state_dict1 = torch.load("model_1.pt", map_location = cpu, weights_only = True)
        model_1.load_state_dict(state_dict1["model_state_dict"], strict = True)
        final_loss, final_acc = evaluate(model_1, test_dataset, Loss_fn)
    else: 
        max_model_index = 2
        state_dict2 = torch.load("model_2.pt", map_location = cpu, weights_only = True)
        model_2.load_state_dict(state_dict2["model_state_dict"], strict = True)
        final_loss, final_acc = evaluate(model_2, test_dataset, Loss_fn)

print(f"best model is {max_model_index}")
print(f"final_loss = {final_loss}")
print(f"final_acc = {final_acc}")


