import torch
import torch.nn as nn

seed = torch.manual_seed(42)

def return_mean_and_var(name: str, x: torch.Tensor, dim: int):
    x_mean = x.mean(dim = dim)
    x_var = x.var(dim = dim, correction = 0)

    print(f"{name}, mean = {x_mean.tolist()}")
    print(f"{name}, var = {x_var.tolist()}")

    return

gpu = "cuda"

X = torch.tensor([[1,2,4],[2,5,6],[4,3,8],[7,8,2]]).to(dtype = torch.float32, device = gpu) #X.shape = (4, 3)
return_mean_and_var("X_bn", X, 0)
return_mean_and_var("X_ln", X, 1)
# X_bn_mean = X.mean(dim = 0)
# X_bn_var = X.var(dim = 0, correction = 1) # 决定方差计算的分母，分母为 N-correction
# X_ln_mean = X.mean(dim = 1)
# X_ln_var = X.var(dim = 1, correction = 1)
# print(f"X_bn_mean = {X_bn_mean.tolist()}, its shape = {X_bn_mean.shape}") # .item() 只能对单元素tensor使用，使用 .tolist() 将多元素tensor转换为列表
# print(f"X_bn_var = {X_bn_var.tolist()}, its shape = {X_bn_var.shape}")
# print(f"X_ln_mean = {X_ln_mean.tolist()}, its shape = {X_ln_mean.shape}")
# print(f"X_ln_var = {X_ln_var.tolist()}, its shape = {X_ln_var.shape}")

bn = nn.BatchNorm1d(3)
bn_test = nn.BatchNorm1d(3, momentum = 1.0).to(gpu)
ln = nn.LayerNorm(3)
bn.to(gpu)
ln.to(gpu)
# bn_shape = bn.shape() bn 是 nn.Module() 模块，没有 shape 属性
print(f"bn.weight.shape = {bn.weight.shape}")

bn_running_mean_before = bn.running_mean.clone()
bn_running_var_before = bn.running_var.clone()
bn_num_batches_tracked_before = bn.num_batches_tracked.clone()
for name, buffer in bn.named_buffers():
    print(name, buffer.shape)
# bn_named_buffers_before = bn.named_buffers().clone() named_buffers()是迭代器，包含若干(名称， tensor)，.clone()只能对于 tensor 类型使用

bn.train()
ln.train()
bn_test.train()

print(f"bn_running_mean_before = {bn_running_mean_before}")
print(f"bn_running_var_before = {bn_running_var_before}")
print(f"bn_num_batches_tracked_before = {bn_num_batches_tracked_before}")
# print(f"bn_named_buffers() = {bn_named_buffers_before}")

with torch.no_grad():
    Y_bn_train = bn(X)
    Y_bn_train_test = bn_test(X)
    Y_ln_train = ln(X)
# Y_bn_train = bn(X)
# Y_ln_train = ln(X)

print(f"Y_bn_train = {Y_bn_train}")
print(f"Y_bn_train.requires_grad = {Y_bn_train.requires_grad}")
# print(f"torch.isfinite = {torch.isfinite(bn.parameters().grad)}") # .parameters()是迭代器，可以通过.weight.grad, .bias.grad查看
print(f"Y_ln_train = {Y_ln_train}")
print(f"Y_ln_train.requires_grad = {Y_ln_train.requires_grad}")
# print(f"torch.isfinite = {torch.isfinite(ln.parameters().grad)}") # 并且 1. with torch.no_grad() 2.没有调用 .backward() 此时 grad = None

bn_running_mean_after_train = bn.running_mean.clone()
bn_running_var_after_train = bn.running_var.clone()
bn_test_running_mean_after_train = bn_test.running_mean.clone()
bn_test_running_var_after_train = bn_test.running_var.clone()
bn_num_batches_tracked_after_train = bn.num_batches_tracked.clone()
# bn_named_buffers_after = bn.named_buffers().clone()

print(f"bn_running_mean() = {bn_running_mean_after_train}") # BN 使用 correction = 0 进行输入数据的归一化，但是更新统计量 running_var 使用 correction = 1
print(f"bn_running_var() = {bn_running_var_after_train}")
print(f"bn_num_batches_tracked() = {bn_num_batches_tracked_after_train}")

allclose_bn_running_mean = torch.allclose(bn_running_mean_before, bn_running_mean_after_train, atol = 1e-5, rtol = 1e-4) # torch.allclose() 比较的参数最好位于同一个设备
allclose_bn_running_var = torch.allclose(bn_running_var_before, bn_running_var_after_train, atol = 1e-5, rtol = 1e-4)
print(f"allclose_bn_running_mean = {allclose_bn_running_mean}")
print(f"allclose_bn_running_var = {allclose_bn_running_var}")

return_mean_and_var("Y_bn_train", Y_bn_train, dim = 0)
return_mean_and_var("Y_ln_train", Y_ln_train, dim = 1)

bn.eval()
ln.eval()
bn_test.eval()

with torch.no_grad():
    Y_bn_eval = bn(X)
    Y_bn_eval_test = bn_test(X)
    Y_ln_eval = ln(X)

bn_running_mean_after_eval = bn.running_mean.clone()
bn_running_var_after_eval = bn.running_var.clone()
bn_test_running_mean_after_eval = bn_test.running_mean.clone()
bn_test_running_var_after_eval = bn_test.running_var.clone()
bn_num_batches_tracked_after_eval = bn.num_batches_tracked.clone()

print(f"bn_running_mean() = {bn_running_mean_after_eval}") # BN 使用 correction = 0 进行输入数据的归一化，但是更新统计量 running_var 使用 correction = 1
print(f"bn_running_var() = {bn_running_var_after_eval}")
print(f"bn_num_batches_tracked() = {bn_num_batches_tracked_after_eval}")

allclose_bn_running_mean_train_and_eval = torch.allclose(bn_running_mean_after_train, bn_running_mean_after_eval, atol = 1e-5, rtol = 1e-4) # torch.allclose() 比较的参数最好位于同一个设备
allclose_bn_running_var_train_and_eval = torch.allclose(bn_running_var_after_train, bn_running_var_after_eval, atol = 1e-5, rtol = 1e-4)
allclose_bn_test_running_mean_train_and_eval = torch.allclose(bn_test_running_mean_after_train, bn_test_running_mean_after_eval, atol = 1e-5, rtol = 1e-4) 
allclose_bn_test_running_var_train_and_eval = torch.allclose(bn_test_running_var_after_train, bn_test_running_var_after_eval, atol = 1e-5, rtol = 1e-4) 
allclose_bn_train_and_eval = torch.allclose(Y_bn_train, Y_bn_eval, atol = 1e-5, rtol = 1e-4)
allclose_ln_train_and_eval = torch.allclose(Y_ln_train, Y_ln_eval, atol = 1e-5, rtol = 1e-4)
allclose_bn_test_train_and_eval = torch.allclose(Y_bn_train_test, Y_bn_eval_test, atol = 1e-5, rtol = 1e-4)
equal_bn_num_batches_tracked_train_and_eval = torch.equal(bn_num_batches_tracked_after_train, bn_num_batches_tracked_after_eval)

print(f"allclose_bn_running_mean_train_and_eval = {allclose_bn_running_mean_train_and_eval}")
print(f"allclose_bn_running_var_train_and_eval = {allclose_bn_running_var_train_and_eval}")
print(f"allclose_bn_test_running_mean_train_and_eval = {allclose_bn_test_running_mean_train_and_eval}")
print(f"allclose_bn_test_running_var_train_and_eval = {allclose_bn_test_running_var_train_and_eval}")
print(f"allclose_bn_train_and_eval = {allclose_bn_train_and_eval}")
print(f"allclose_ln_train_and_eval = {allclose_ln_train_and_eval}")
print(f"allclose_bn_test_train_and_eval = {allclose_bn_test_train_and_eval}")
print(f"equal_bn_num_batches_tracked_train_and_eval = {equal_bn_num_batches_tracked_train_and_eval}")

Y = torch.ones(32, 16).to(device = gpu) # 默认 torch.ones 创建在 cpu 上
dropout = nn.Dropout(p = 0.5)

dropout.train()
with torch.no_grad():
    Y1 = dropout(Y)
    Y2 = dropout(Y)

unique_Y1 = torch.unique(Y1)
mean_0_Y1 = (Y1 == 0).float().mean()
unique_Y2 = torch.unique(Y2)
mean_0_Y2 = (Y2 == 0).float().mean()
allclose_Y1_Y2 = torch.allclose(Y1, Y2)
print(f"unique_Y1 = {unique_Y1}")
print(f"unique_Y2 = {unique_Y2}")
print(f"mean_0_Y1 = {mean_0_Y1}")
print(f"mean_0_Y2 = {mean_0_Y2}")
print(f"allclose_Y1_Y2 = {allclose_Y1_Y2}")

print(f"Y1.shape = {Y1.shape}, Y1.requires_grad = {Y1.requires_grad}")
print(f"Y2.shape = {Y2.shape}, Y2.requires_grad = {Y2.requires_grad}")

dropout.eval()
with torch.no_grad():
    Y3 = dropout(Y)

allclose_Y_Y3 = torch.allclose(Y, Y3)
print(f"allclose_Y_Y3 = {allclose_Y_Y3}")

print(f"Y1.shape = {Y1.shape}, Y1.requires_grad = {Y1.requires_grad}")
print(f"Y2.shape = {Y2.shape}, Y2.requires_grad = {Y2.requires_grad}")
print(f"Y3.shape = {Y3.shape}, Y3.requires_grad = {Y3.requires_grad}")

print("=" * 100)

class Linear_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer_1 = nn.Linear(3, 8)
        self.bn = nn.BatchNorm1d(8)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.linear_layer_2 = nn.Linear(8, 3)

    def forward(self, X):
        input = X
        Y1 = self.linear_layer_1(input)
        Y2 = self.bn(Y1)
        Y3 = self.relu(Y2)
        Y4 = self.dropout(Y3)
        Y5 = self.linear_layer_2(Y4)
        output = Y5

        return output

target = torch.tensor([0, 1, 2, 1], device = gpu, dtype = torch.long)

model = Linear_Model()
model.to(gpu)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
Loss_fn = nn.CrossEntropyLoss()

model_bn_rm_before_train = model.bn.running_mean.clone()
model_bn_rv_before_train = model.bn.running_var.clone()
print(f"model_bn_rm_before_train = {model_bn_rm_before_train}")
print(f"model_bn_rv_before_train = {model_bn_rv_before_train}")

params_before_step_dict = {}
params_after_step_dict = {}

model.train()

for name, parameter in model.named_parameters():
    print(f"before stap, name = {name}, parameter.grad = {parameter.grad}")
    params_before_step_dict[name] = parameter.detach().clone()
    # params_before_step_dict["name"] =  name
    # params_before_step_dict["params"] = parameter.detach().clone()
    # params_before_step_list.append(params_before_step_dict)
    if parameter.grad is not None:
        print(f"isfinite = {torch.isfinite(parameter.grad).all()}")
print(f"params_before_step_dict = {params_before_step_dict}")

for epoch in range(1):
    optimizer.zero_grad()

    pred = model(X)

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            print("!" * 100)
            print(f"{name}'s parameter.grad is None")

    loss = Loss_fn(pred, target)
    loss.backward()

    optimizer.step()

for name, parameter in model.named_parameters():
    print(f"after step, name = {name}, parameter.grad = {parameter.grad}")
    params_after_step_dict[name] = parameter.detach().clone()
    # params_before_step_dict["name"] =  name
    # params_before_step_dict["params"] = parameter.detach().clone()
    # params_after_step_list.append(params_after_step_dict)
    if parameter.grad is not None:
        print(f"isfinite = {torch.isfinite(parameter.grad).all()}")
print(f"params_after_step_dict = {params_after_step_dict}")


model_bn_rm_after_train = model.bn.running_mean.clone()
model_bn_rv_after_train = model.bn.running_var.clone()
print(f"model_bn_rm_after_train = {model_bn_rm_after_train}")
print(f"model_bn_rv_after_train = {model_bn_rv_after_train}")
allclose_model_bn_rm_train = torch.allclose(model_bn_rm_before_train, model_bn_rm_after_train, atol = 1e-5, rtol = 1e-4)
allclose_model_bn_rv_train = torch.allclose(model_bn_rv_before_train, model_bn_rv_after_train, atol = 1e-5, rtol = 1e-4)
print(f"allclose_model_bn_rm_train = {allclose_model_bn_rm_train}")
print(f"allclose_model_bn_rv_train = {allclose_model_bn_rv_train}")

model.eval()
pred_eval = model(X)

model_bn_rm_after_eval = model.bn.running_mean.clone()
model_bn_rv_after_eval = model.bn.running_var.clone()
print(f"model_bn_rm_after_eval = {model_bn_rm_after_eval}")
print(f"model_bn_rv_after_eval = {model_bn_rv_after_eval}")
allclose_model_bn_rm_eval = torch.allclose(model_bn_rm_after_train, model_bn_rm_after_eval, atol = 1e-5, rtol = 1e-4)
allclose_model_bn_rv_eval = torch.allclose(model_bn_rv_after_train, model_bn_rv_after_eval, atol = 1e-5, rtol = 1e-4)
print(f"allclose_model_bn_rm_eval = {allclose_model_bn_rm_eval}")
print(f"allclose_model_bn_rv_eval = {allclose_model_bn_rv_eval}")

with torch.no_grad():
    pred_eval_no_grad = model(X)
    
model_bn_rm_after_eval_no_grad = model.bn.running_mean.clone()
model_bn_rv_after_eval_no_grad = model.bn.running_var.clone()
print(f"model_bn_rm_after_eval_no_grad = {model_bn_rm_after_eval_no_grad}")
print(f"model_bn_rv_after_eval_no_grad = {model_bn_rv_after_eval_no_grad}")
allclose_model_bn_rm_eval_no_grad = torch.allclose(model_bn_rm_after_train, model_bn_rm_after_eval_no_grad, atol = 1e-5, rtol = 1e-4)
allclose_model_bn_rv_eval_no_grad = torch.allclose(model_bn_rv_after_train, model_bn_rv_after_eval_no_grad, atol = 1e-5, rtol = 1e-4)
print(f"allclose_model_bn_rm_eval_no_grad = {allclose_model_bn_rm_eval_no_grad}")
print(f"allclose_model_bn_rv_eval_no_grad = {allclose_model_bn_rv_eval_no_grad}")

print(f"pred.requires_grad = {pred.requires_grad}")
print(f"pred_eval.requires_grad = {pred_eval.requires_grad}")
print(f"pred_eval_no_grad.requires_grad = {pred_eval_no_grad.requires_grad}")

for name, parameter in model.named_parameters():
    before = params_before_step_dict[name]
    after = params_after_step_dict[name]
    equal = torch.equal(before, after)
    print(f"name = {name}, equal = {equal}")

allclose_eval_and_eval_no_grad = torch.allclose(pred_eval, pred_eval_no_grad)
print(f"allclose_eval_and_eval_no_grad = {allclose_eval_and_eval_no_grad}")


