import torch
import torch.nn as nn

X = torch.tensor([-2, -1, 1, 2], dtype = torch.float32)
X = X.reshape(4, 1)
Y = 2 * X + 1

def create_model():
    # torch.manual_seed(42)
    model = nn.Linear(1, 1)

    return model

def create_optimizer(model, lr = 0.1, momentum = 0.0, weight_decay = 0.0):
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay = weight_decay)

    return optimizer

torch.manual_seed(42)
model_a = create_model()
model_b = create_model()
model_c = create_model()
model_b.load_state_dict(model_a.state_dict())
model_c.load_state_dict(model_a.state_dict())
# model_b = create_model()
# model_c = create_model()

optimizer_a = create_optimizer(model_a)
optimizer_b = create_optimizer(model_b)
optimizer_c = create_optimizer(model_c)

loss_fn = nn.MSELoss()

def saveParameters(model):
    weight = model.weight.detach().clone()
    bias = model.bias.detach().clone()

    return weight, bias

def savegrad(model):
    grad_w = model.weight.grad.detach().clone()
    grad_b = model.bias.grad.detach().clone()

    return grad_w, grad_b

def reference(model, optimizer, loss_fn, input, tgt):

    optimizer.zero_grad()

    pred = model(input)
    loss = loss_fn(pred, tgt)
    loss.backward()

    grad_w, grad_b = savegrad(model)

    optimizer.step()

    weight_after, bias_after = saveParameters(model)

    result = {
        "grad_w": grad_w,
        "grad_b": grad_b,
        "weight_after": weight_after,
        "bias_after": bias_after
    }

    return result

def accumulation(model, optimizer, loss_fn, input, tgt):
    X_chunks = torch.chunk(input, chunks = 2, dim = 0)
    Y_chunks = torch.chunk(tgt, chunks = 2, dim = 0)

    optimizer.zero_grad()

    for micro_x, micro_y in zip(X_chunks, Y_chunks):
        micro_pred = model(micro_x)
        loss = loss_fn(micro_pred, micro_y)
        scaled_loss = loss / 2
        scaled_loss.backward()

    grad_w, grad_b = savegrad(model)

    optimizer.step()

    weight_after, bias_after = saveParameters(model)

    result = {
        "grad_w": grad_w,
        "grad_b": grad_b,
        "weight_after": weight_after,
        "bias_after": bias_after
    }

    return result

def grad_norm(model):
    grad_w = model.weight.grad
    grad_b = model.bias.grad

    sq_sum_w = grad_w.square().sum()
    sq_sum_b = grad_b.square().sum()

    norm = torch.sqrt(sq_sum_w + sq_sum_b)

    return norm

def accumulation_clip(model, optimizer, loss_fn, input, tgt):
    X_chunks = torch.chunk(input, chunks = 2, dim = 0)
    Y_chunks = torch.chunk(tgt, chunks = 2, dim = 0)

    optimizer.zero_grad()

    for micro_x, micro_y in zip(X_chunks, Y_chunks):
        micro_pred = model(micro_x)
        loss = loss_fn(micro_pred, micro_y)
        scaled_loss = loss / 2
        scaled_loss.backward()

    grad_w, grad_b = savegrad(model)

    norm_before = nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm = 0.1,
        norm_type = 2.0
    )

    grad_w_clipped, grad_b_clipped = savegrad(model)
    norm_after = grad_norm(model)

    optimizer.step()

    weight_after, bias_after = saveParameters(model)

    result = {
        "grad_w": grad_w,
        "grad_b": grad_b,
        "grad_w_clipped": grad_w_clipped,
        "grad_b_clipped": grad_b_clipped,
        "norm_before": norm_before,
        "norm_after": norm_after,
        "weight_after": weight_after,
        "bias_after": bias_after
    }

    return result

def printParameter(model):
    print(f"model.weight.grad = {model.weight.grad}")
    print(f"model.bias.grad = {model.bias.grad}")

def main():
    result_a = reference(model_a, optimizer_a, loss_fn, X, Y) # Original optimizer update
    result_b = accumulation(model_b, optimizer_b, loss_fn, X, Y) # minibatch optimizer update
    result_c = accumulation_clip(model_c, optimizer_c, loss_fn, X, Y) # minibatch optimizer update

    same_grad_w = torch.allclose(
        result_a["grad_w"],
        result_b["grad_w"],
        rtol=1e-5,
        atol=1e-6
    )
    same_grad_b = torch.allclose(
        result_a["grad_b"],
        result_b["grad_b"],
        rtol=1e-5,
        atol=1e-6
    )
    same_weight = torch.allclose(
        result_a["weight_after"],
        result_b["weight_after"],
        rtol=1e-5,
        atol=1e-6
    )
    same_bias = torch.allclose(
        result_a["bias_after"],
        result_b["bias_after"],
        rtol=1e-5,
        atol=1e-6
    )
    same_grad_w_before_clip = torch.allclose(
        result_b["grad_w"],
        result_c["grad_w"],
        rtol=1e-5,
        atol=1e-6
    )
    same_grad_b_before_clip = torch.allclose(
        result_b["grad_b"],
        result_c["grad_b"],
        rtol=1e-5,
        atol=1e-6
    )

    # result_c = reference(model_c, optimizer_c, loss_fn, X, Y)

    print(f"same_grad_w = {same_grad_w}")
    print(f"same_grad_b = {same_grad_b}")
    print(f"same_weight = {same_weight}")
    print(f"same_bias = {same_bias}")
    print(f"same_grad_w_before_clip = {same_grad_w_before_clip}")
    print(f"same_grad_b_before_clip = {same_grad_b_before_clip}")

    norm_before = result_c["norm_before"]
    print(f"norm_before = {norm_before}")
    print(norm_before.item() > 0.1)
    scale = min(1.0, 0.1 / norm_before.item())
    expected_grad_w = result_c["grad_w"] * scale
    same_scaled_grad_w = torch.allclose(
        expected_grad_w,
        result_c["grad_w_clipped"],
        rtol=1e-5,
        atol=1e-6
    )
    print(f"same_scaled_grad_w = {same_scaled_grad_w}")

    norm_after = result_c["norm_after"]
    assert norm_after.item() <= 0.1 + 1e-6
    print(f"norm_after = {norm_after}")

    print(f"grad_before_clip_direct = {(result_c['grad_w'], result_c['grad_b'])}")
    print(f"grad_after_clip_direct = {(result_c['grad_w_clipped'], result_c['grad_b_clipped'])}")

if __name__ == "__main__":
    main()
