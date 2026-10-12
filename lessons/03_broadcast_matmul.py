import torch 

X = torch.arange(1.,7.)
X = X.reshape(2,3)

bias = torch.tensor([10., 20., 30.])
row_offset = torch.tensor([100., 200.])

W = torch.tensor([[1., 0.],[0., 1.],[1., 1.]])

output_bias = torch.tensor([0.5, -0.5])

print("X + bias is", X+bias)
print("X + bias shape is", (X + bias).shape) #  X + bias = [[11., 22., 33],[14., 25., 36.]] , shape = (2,3)

Y = torch.matmul(X, W) + output_bias
print("Y is ", Y)
print("Y.shape is", Y.shape) # Y = [[4.5, 4.5],[10.5, 10.5]] , shape = (2,2)

Z = torch.stack((X, X+10), dim=0) # Z = [[[1.,2.,3.], [4.,5.,6.]],[[11.,12.,13.],[14.,15.,16.]]], Z.shape = (2,2,3)
print("Z is ", Z)
print("Z.shape is ", Z.shape)

R = X * bias # R = [[10.,40.,90.], [40.,100.,180]]
R0 = torch.einsum("ij,jk->ik", R, W)
R1 = torch.matmul(R, W)
close = torch.allclose(R0, R1)
print("allclose result is", close)

row_offset_unsqueeze = row_offset.unsqueeze(1)
X_row_offset = X + row_offset_unsqueeze
print("X + row_offset_unqueeze is", X_row_offset)
print("X_row_offset.shape is ", X_row_offset.shape)

Z_1 = torch.matmul(Z,W) + output_bias # Z_1 = [[[4.5, 4.5],[10.5, 10.5]],[[24.5, 24.5],[30.5, 30.5]]], Z_1.shape = (2,2,2)
print("Z_1 is ", Z_1)
print("Z_1.shape is", Z_1.shape)

Z_2 = torch.einsum("ijk,kt->ijt", Z, W) # k被求和，ijt被保留
print("Z_2.shape is", Z_2.shape)
Z_3 = torch.matmul(Z,W)
print("Z_3.shape is", Z_3.shape)
close_1 = torch.allclose(Z_2, Z_3)
print("allclose_Z is", close_1)


