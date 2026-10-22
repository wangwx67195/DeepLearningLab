import torch
import torch.nn as nn
import torch.nn.functional as F

seed = torch.manual_seed(42)
gpu = "cuda"
X = torch.arange(1, 17, dtype=torch.float32, device=gpu)
X = X.reshape((4, 4))
print(f"X = {X}")
kernel = torch.tensor([[1, 0], [-1, 2]], dtype=torch.float32, device=gpu)
kernel_size = int(2)
bias = torch.tensor([0.5], dtype = torch.float32, device = gpu)
stride = int(1)
padding = int(0)
stride1 = int(2)
padding1 = int(0)
stride2 = int(1)
padding2 = int(1)
stride3 = int(2)
padding3 = int(1)

def conv2d_single(x, kernel, bias=0.5, stride=1, padding=padding):
    print(f"h, w before padding = {x.shape}")
    src = F.pad(x, (padding, padding, padding, padding), mode="constant", value=0)
    # print(f"src.device = {src.device}")
    h, w = src.shape
    print(f"h, w after padding = {src.shape}")
    kh, kw = kernel.shape
    h_out = (h - kh) // stride + 1
    w_out = (w - kw) // stride + 1
    output = torch.zeros((h_out, w_out), dtype = torch.float32, device = gpu)
    # print(f"output.device = {output.device}")
    print(f"outputs_single h, w = {output.shape}")

    for hk in range(h_out):
        for wk in range(w_out):
            start_h = hk * stride
            start_w = wk * stride
            window = src[start_h: start_h+kh, start_w: start_w+kw]
            window_output = (window * kernel).sum() + bias
            output[hk, wk] = window_output

    return output

def maxpool2d_single(x, kernel_size=2, stride=1, padding=0):
    src = F.pad(x, (padding, padding, padding, padding), mode="constant", value=float("-inf"))
    h, w = src.shape
    print(f"h, w after padding = {src.shape}")
    kh = kernel_size
    kw = kernel_size
    h_out = (h - kh) // stride + 1
    w_out = (w - kw) // stride + 1
    output = torch.zeros((h_out, w_out), dtype = torch.float32, device = gpu)
    print(f"outputs_single h, w = {output.shape}")

    for hk in range(h_out):
        for wk in range(w_out):
            start_h = hk * stride
            start_w = wk * stride
            window = src[start_h: start_h+kh, start_w: start_w+kw]
            window_output = window.max()
            output[hk, wk] = window_output

    return output

def avgpool2d_single(x, kernel_size=2, stride=1, padding=0):
    src = F.pad(x, (padding, padding, padding, padding), mode="constant", value=0)
    h, w = src.shape
    print(f"h, w after padding = {src.shape}")
    kh = kernel_size
    kw = kernel_size
    h_out = (h - kh) // stride + 1
    w_out = (w - kw) // stride + 1
    output = torch.zeros((h_out, w_out), dtype = torch.float32, device = gpu)
    print(f"outputs_single h, w = {output.shape}")

    for hk in range(h_out):
        for wk in range(w_out):
            start_h = hk * stride
            start_w = wk * stride
            window = src[start_h: start_h+kh, start_w: start_w+kw]
            window_output = window.mean()
            output[hk, wk] = window_output

    return output

output_single = conv2d_single(X, kernel, bias)
print(f"output_single = {output_single}")
output_single_maxpool2d = maxpool2d_single(X, kernel_size)
print(f"output_single_maxpool2d = {output_single_maxpool2d}")
output_single_avgpool2d = avgpool2d_single(X, kernel_size)
print(f"output_single_avgpool2d = {output_single_avgpool2d}")

X1 = X
X1 = X1.unsqueeze(0)
X1 = X1.unsqueeze(0)
kernel1 = kernel
kernel1 = kernel1.unsqueeze(0)
kernel1 = kernel1.unsqueeze(0)

output_F_conv = F.conv2d(X1, kernel1, bias=bias).to(device=gpu)
print(f"output_F_conv = {output_F_conv}")
output_F_conv = output_F_conv[0, 0]
error_max = (output_single - output_F_conv).abs().max()
allclose_manual_and_torch = torch.allclose(output_single, output_F_conv)
print(f"error_max manual and torch = {error_max}")
print(f"allclose_manual_and_torch = {allclose_manual_and_torch}")

target = output_F_conv

conv2d = nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=2,
    stride=1,
    padding=0,
    bias=True
)
print(f"conv2d.weight.shape = {conv2d.weight.shape}")
print(f"conv2d.bias.shape = {conv2d.bias.shape}")



output_single1 = conv2d_single(X, kernel, bias = bias, stride = stride1, padding = padding1)
print(f"output_single1 = {output_single1}")
output_single_maxpool2d1 = maxpool2d_single(X, kernel_size, stride = stride1, padding = padding1)
print(f"output_single_maxpool2d1 = {output_single_maxpool2d1}")
output_single_avgpool2d1 = avgpool2d_single(X, kernel_size, stride = stride1, padding = padding1)
print(f"output_single_avgpool2d1 = {output_single_avgpool2d1}")
output_F_conv1 = F.conv2d(X1, kernel1, bias=bias, stride = stride1, padding = padding1).to(device=gpu)
print(f"output_F_conv = {output_F_conv1}")
output_F_conv1 = output_F_conv1[0, 0]
allclose_manual_and_torch1 = torch.allclose(output_single1, output_F_conv1)
print(f"allclose_manual_and_torch = {allclose_manual_and_torch1}")
error_max1 = (output_single1 - output_F_conv1).abs().max()
print(f"error_max manual and torch = {error_max1}")

output_single2 = conv2d_single(X, kernel, bias = bias, stride = stride2, padding = padding2)
print(f"output_single2 = {output_single2}")
output_single_maxpool2d2 = maxpool2d_single(X, kernel_size, stride = stride2, padding = padding2)
print(f"output_single_maxpool2d2 = {output_single_maxpool2d2}")
output_single_avgpool2d2 = avgpool2d_single(X, kernel_size, stride = stride2, padding = padding2)
print(f"output_single_avgpool2d2 = {output_single_avgpool2d2}")
output_F_conv2 = F.conv2d(X1, kernel1, bias=bias, stride = stride2, padding = padding2).to(device=gpu)
print(f"output_F_conv = {output_F_conv2}")
output_F_conv2 = output_F_conv2[0, 0]
allclose_manual_and_torch2 = torch.allclose(output_single2, output_F_conv2)
print(f"allclose_manual_and_torch = {allclose_manual_and_torch2}")
error_max2 = (output_single2 - output_F_conv2).abs().max()
print(f"error_max manual and torch = {error_max2}")

output_single3 = conv2d_single(X, kernel, bias = bias, stride = stride3, padding = padding3)
print(f"output_single3 = {output_single3}")
output_single_maxpool2d3 = maxpool2d_single(X, kernel_size, stride = stride3, padding = padding3)
print(f"output_single_maxpool2d3 = {output_single_maxpool2d3}")
output_single_avgpool2d3 = avgpool2d_single(X, kernel_size, stride = stride3, padding = padding3)
print(f"output_single_avgpool2d3 = {output_single_avgpool2d3}")
output_F_conv3 = F.conv2d(X1, kernel1, bias=bias, stride = stride3, padding = padding3).to(device=gpu)
print(f"output_F_conv = {output_F_conv3}")
output_F_conv3 = output_F_conv3[0, 0]
allclose_manual_and_torch3 = torch.allclose(output_single3, output_F_conv3)
print(f"allclose_manual_and_torch = {allclose_manual_and_torch3}")
error_max3 = (output_single3 - output_F_conv3).abs().max()
print(f"error_max manual and torch = {error_max3}")

maxpool = nn.MaxPool2d(kernel_size = kernel_size, stride = stride, padding = padding)
avgpool = nn.AvgPool2d(kernel_size = kernel_size, stride = stride, padding = padding)


print("=" * 100)

X_multi_1 = torch.arange(25, dtype = torch.float32, device = gpu)
X_multi_2 = X_multi_1.reshape(5, 5)
X_multi_3 = torch.stack((X_multi_2, X_multi_2), dim = 0)
X_multi_4 = torch.stack((X_multi_3, X_multi_3), dim = 0)
print(X_multi_4.dtype)
# print(X_multi_4.shape)
# print(X_multi_4)

conv2d1 = nn.Conv2d(
    in_channels=2,
    out_channels=3,
    kernel_size=3,
    stride=2,
    padding=1,
    bias=True
)
conv2d1.to(device = gpu)
print(conv2d1.weight.dtype)

Y_multi = conv2d1(X_multi_4)
print(f"Y_multi.shaope = {Y_multi.shape}")
print(f"conv2d1.weight.shape = {conv2d1.weight.shape}")
print(f"conv2d1.bias.shape = {conv2d1.bias.shape}")
conv_params_numel = sum(p.numel() for p in conv2d1.parameters())
print(f"total conv2d1 params numel = {conv_params_numel}")

maxpool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
Y_multi_1 = maxpool1(Y_multi)
print(f"Y_multi_1.shape = {Y_multi_1.shape}")
avgpool1 = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)
Y_multi_2 = avgpool1(Y_multi)
print(f"Y_multi_2.shape = {Y_multi_2.shape}")
