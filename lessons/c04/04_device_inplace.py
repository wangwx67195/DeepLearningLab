import torch

a = torch.arange(1,7).to(dtype = torch.float32)
a = a.reshape(2,3).to(device = "cuda:0")
if torch.cuda.is_available():
    print("a device is", a.device)
    a = a.to(device = "cpu")
    print("a.device is", a.device)
print("a is ", a)
print("a.device is ", a.device)

alias = a[:, 0:1]
print("alias is", alias)

backup = a.clone()
print("backup is ", backup)
print("backup is a =", backup is a)

a.add_(10)
print("a is ", a)
print("alias is", alias)
print("backup is", backup)

a = torch.arange(1,7).to(dtype = torch.float32)
a = a.reshape(2,3)
alias = a
print("alias is a = ", alias is a)
a = a + 10
print("alias is", alias)
print("a is ", a)
print("a is alias =", a is alias)

if torch.cuda.is_available():
    X = torch.arange(1.,7.)
    X = X.reshape(2,3)
    X = X.to(device = "cpu")

    bias = torch.tensor([10., 20., 30.])
    row_offset = torch.tensor([100., 200.])

    W = torch.tensor([[1., 0.],[0., 1.],[1., 1.]])
    W = W.to(device = "cuda:0")

    try:
        Y = torch.matmul(X, W)
    except RuntimeError as err:
        print("error is",err)

    try:
        Y_0 = torch.matmul(X, W.to("cpu"))
        print("Y_0 is", Y_0)
        print("Y_0.shape is", Y_0.shape)
        print("Y_0.dtype is", Y_0.dtype)
    except RuntimeError as err:
        print("error is", err)

c = torch.arange(6)
c = c.reshape(2,3)
print("c is", c)
print("c.detype is", c.dtype)
c = c.to(torch.float32)
print("c.dtype is", c.dtype)

c1 = c.to(device = "cpu")
print("c1 is c", c1 is c)

c2 = c.to(device = "cpu", copy = True)
print("c2 is", c2)
print("c2 is c", c2 is c)

c3 = c.to(device = "cpu", dtype = torch.float64)
print("c3 is", c3)
print("c3 is c", c3 is c)
