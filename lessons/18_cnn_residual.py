import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy

gpu = "cuda"
cpu = "cpu"

seed = torch.manual_seed(42)

# create 3 classes images
image_1 = torch.zeros((32, 1, 16, 16), dtype = torch.float32, device = gpu)
image_1[:, :, :, 6:10] = 1
image_1 = image_1 + torch.randn_like(image_1) * 0.2
image_2 = torch.zeros((32, 1, 16, 16), dtype = torch.float32, device = gpu)
image_2[:, :, 6:10, :] = 1
image_2 = image_2 + torch.randn_like(image_2) * 0.2
image_3 = torch.zeros((32, 1, 16, 16), dtype = torch.float32, device = gpu)
image_3[:, :, :, 6:10] = 1
image_3[:, :, 6:10, :] = 1
image_3 = image_3 + torch.randn_like(image_3) * 0.2

label_1 = torch.full((32,), 0, dtype = torch.long, device = gpu)
label_2 = torch.full((32,), 1, dtype = torch.long, device = gpu)
label_3 = torch.full((32,), 2, dtype = torch.long, device = gpu)

indices = torch.randperm(32)
train_indice = indices[:16]
val_indice = indices[16:24]
test_indice = indices[24:]
train_indices = torch.cat([train_indice, train_indice + 32, train_indice + 64], dim = 0)
val_indices = torch.cat([val_indice, val_indice + 32, val_indice + 64], dim = 0)
test_indices = torch.cat([test_indice, test_indice + 32, test_indice + 64], dim = 0)

print(f"train_In_val = {torch.any(torch.isin(train_indices, val_indices))}")
print(f"train_In_test = {torch.any(torch.isin(train_indices, test_indices))}")

train_images = torch.cat([image_1[train_indice], image_2[train_indice], image_3[train_indice]], dim = 0)
val_images = torch.cat([image_1[val_indice], image_2[val_indice], image_3[val_indice]], dim = 0)
test_images = torch.cat([image_1[test_indice], image_2[test_indice], image_3[test_indice]], dim = 0)
print(f"train_images.shape = {train_images.shape}")

train_labels = torch.cat([label_1[train_indice], label_2[train_indice], label_3[train_indice]], dim = 0)
val_labels = torch.cat([label_1[val_indice], label_2[val_indice], label_3[val_indice]], dim = 0)
test_labels = torch.cat([label_1[test_indice], label_2[test_indice], label_3[test_indice]], dim = 0)
print(f"train_labels.shape = {train_labels.shape}")

# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, input, target):
#         self.input = input
#         self.target = target

#     def __len__(self):
#         return len(self.input)

#     def __getitem__(self, idx):
#         return (self.input[idx], self.target[idx])

train_dataset = TensorDataset(train_images, train_labels)
val_dataset = TensorDataset(val_images, val_labels)
test_dataset = TensorDataset(test_images, test_labels)

train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True, num_workers = 0)
val_loader = DataLoader(val_dataset, batch_size = 16, shuffle = True, num_workers = 0)
test_loader = DataLoader(test_dataset, batch_size = 16, shuffle = True, num_workers = 0)

def evaluate(model, images, labels, loss_fn):
    model.eval()
    with torch.no_grad():
        pred_logits = model(images)
        pred_labels = pred_logits.argmax(dim = 1)
        pred_correct = (pred_labels == labels).float().sum()
        pred_acc = (pred_labels == labels).float().mean()
        # .item() 只能用于单元素 Tensor
        pred_CE = loss_fn(pred_logits, labels)

        return (pred_CE, pred_correct, pred_acc)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        if in_channels == out_channels and stride == 1:
            self.Layers_shortcut = nn.Identity()
        else:
            self.Layers_shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels)
            )

        self.Layers_main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, input):
        pred_main = self.Layers_main(input)
        pred_shortcut = self.Layers_shortcut(input)
        assert pred_main.shape == pred_shortcut.shape, \
            "After forward, main_shape and shortcut_shape should be SAME !"

        pred = nn.ReLU()(pred_main + pred_shortcut)
        # pred = nn.ReLU(pred) 是错误调用，返回一个 ReLU 对象
        # 正确方式 
        # 1.self.relu = nn.ReLU() -> pred = self.relu(input)
        # 2.pred = F.relu(input)
        # 3.pred = nn.ReLU()(input)

        return pred

class TinyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            ResidualBlock(8, 8, stride = 1),
            ResidualBlock(8, 16, stride = 2)
        )

        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.Linear = nn.Linear(16, 3)

    def forward(self, input):
        pred = self.Layers(input)
        pred = self.AdaptiveAvgPool(pred)
        pred = torch.flatten(pred, start_dim = 1)
        pred = self.Linear(pred)

        return pred

seed = torch.manual_seed(123)

model = TinyResNet()
model.to(gpu)

Loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.01, weight_decay=0)

train_loss_list = []
train_correct_list = []
train_acc_list = []
val_loss_list = []
val_correct_list = []
val_acc_list = []
max_train_loss = float("inf")
max_val_loss = float("inf")

best_state = {}
weight_before = model.Layers[4].Layers_shortcut[0].weight.detach().clone()
print(f"WEIGHT BEFORE = {weight_before[0][0]}")

model.eval()
with torch.no_grad():
    print("=" * 50, "BEFORE Train", "=" * 50)
    pred_train_CE, pred_train_correct, pred_train_acc = evaluate(model, train_images, train_labels, Loss_fn)
    print(f"pred_train_CE = {pred_train_CE} | pred_train_correct = {pred_train_correct} | pred_train_acc = {pred_train_acc}")
    pred_val_CE, pred_val_correct, pred_val_acc = evaluate(model, val_images, val_labels, Loss_fn)
    print(f"pred_val_CE = {pred_val_CE} | pred_val_correct = {pred_val_correct} | pred_val_acc = {pred_val_acc}")

    for name, parameter in model.named_parameters():
        print(f"{name}, parameter.grad is None: {parameter.grad is None}")

for epoch in range(20):
    model.train()
    for idx, batch_data in enumerate(train_loader):
        optimizer.zero_grad()

        input_batch, target_batch = batch_data
        pred_batch = model(input_batch)

        loss = Loss_fn(pred_batch, target_batch)
        loss.backward()

        if epoch == 0 and idx == 0:
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    print(f"{name}: NO GRADIENT !!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                else:
                    print(f"{name}: finite = {torch.isfinite(parameter.grad).all().item()}")
        weight_before_step = model.Layers[4].Layers_shortcut[0].weight.detach().clone()
        print(f"WEIGHT before_step = {weight_before_step[0][0]}")

        optimizer.step()
        weight_after = model.Layers[4].Layers_shortcut[0].weight.detach().clone()
        print(f"WEIGHT after = {weight_after[0][0]}")

    model.eval()
    with torch.no_grad():
        # train_loss = Loss_fn(model(train_images), train_labels)
        # train_loss_value = train_loss.item()
        pred_train_CE, pred_train_correct, pred_train_acc = evaluate(model, train_images, train_labels, Loss_fn)
        pred_val_CE, pred_val_correct, pred_val_acc = evaluate(model, val_images, val_labels, Loss_fn)

        train_loss_value = pred_train_CE.item()
        train_correct_value = pred_train_correct.item()
        train_acc_value = pred_train_acc.item()
        val_loss_value = pred_val_CE.item()
        val_correct_value = pred_val_correct.item()
        val_acc_value = pred_val_acc.item()

        if val_loss_value <= max_val_loss:
            max_val_loss = val_loss_value
            best_state["state"] = deepcopy(model.state_dict())
            best_state["epoch"] = epoch
            torch.save(best_state, "best_model.pt")

        train_loss_list.append(train_loss_value)
        train_correct_list.append(train_correct_value)
        train_acc_list.append(train_acc_value)
        val_loss_list.append(val_loss_value)
        val_correct_list.append(val_correct_value)
        val_acc_list.append(val_acc_value)

        if epoch % 5 == 0:
            print("~" * 50, epoch, "~" * 50)
            print(f"Train CE = {pred_train_CE}")
            print(f"Train ACC = {pred_train_acc}")
            print(f"Val CE = {pred_val_CE}")
            print(f"Val Acc = {pred_val_acc}")

    if epoch == 0:
        print("=" * 50, "EPOCH 0", "=" * 50)
        for name, parameter in model.named_parameters():
            print(f"{name}, parameter.grad is None: {parameter.grad is None}")

model.eval()
with torch.no_grad():
    print("=" * 50, "Within Test", "=" * 50)
    model.load_state_dict(best_state["state"])
    best_epoch = best_state["epoch"]
    print(f"BEST EPOCH = {best_epoch}")

    pred_test_CE, pred_test_correct, pred_test_acc = evaluate(model, test_images, test_labels, Loss_fn)
    print(f"pred_test_CE = {pred_test_CE} | pred_test_correct = {pred_test_correct} | pred_test_acc = {pred_test_acc}")

    for name, parameter in model.named_parameters():
        print(f"{name}, parameter.grad is None: {parameter.grad is None}")
    






