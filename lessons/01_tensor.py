import torch

x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

print(x)
print(x.shape)
print(x.dtype)
print(x.device)

print(x[0,:])
print(x[:,1])

print(x.mean(dim = 0))
print(x.mean(dim = 1))

y = x.to("cuda")

print(y.device)

z = torch.tensor([[2., 4.], [6., 8.,],[10., 12.,]])
print(z[0, :]) # [2., 4.],shape = [2,]
print(z[0, :].shape)

print(z[:,1]) # [4., 8., 12.],shape = [3,]
print(z[:,1].shape)

print(z.mean(dim = 0)) # [6., 8.],shape = [2,]
print(z.mean(dim = 0).shape)

print(z.mean(dim = 1)) # [3., 7., 11.], shape = [3,]
print(z.mean(dim = 1).shape)

print(z[0:1, :]) # [[2., 4.]], shape = [1, 2]
print(z[0:1, :].shape)
#索引中的 : 表示取该维度的全部元素；dim = 0 表示沿着行轴求平均，dim = 1表示沿着列轴求平均