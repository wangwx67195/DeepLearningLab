import torch

x = torch.tensor([[1., 2., 3.],[4., 5., 6.]])

a = x.unsqueeze(0) # a.shape = [1,2,3]
print(a.shape)

b = x.unsqueeze(1) # b.shape [2,1,3]
print(b.shape)

c = x.unsqueeze(2) # c.shape = [2,3,1]
print(c.shape)

d = b.squeeze(1) # d.shape = [2,3]
print(d.shape)

e = b.squeeze(0) # e.shape = [2,1,3]
print(e.shape)

f = x.reshape(3,2) # f = [[1.,2], [3.,4.],[5.,6.]],f.shape = [3,2]
print(f)
print(f.shape)

g = x.reshape(-1) # g = [1., 2., 3., 4., 5., 6.], g.shape = [6]
print(g)
print(g.shape)

h = x.reshape(2, -1) # h = [[1., 2., 3.],[4., 5., 6.]],h.shape = [2,3]
print(h)
print(h.shape)

# x.reshape(4,2)不合法原因：x总元素个数为6，4 \times 2 = 8 > 6

i = x.transpose(0,1) # i = [[1.,4 .],[2., 5.],[3., 6.]] i.shape = [3,2]
print(i)
print(i.shape)

# i和x.reshape(3,2)不同之处：shape相同，每一行的元素不同

j = i.transpose(0,1) # j = [[1.,2.,3.], [4., 5., 6.]] j.shape = [2,3]
print(j)
print(j.shape)

print("x.shape is ",x.shape)
print("x.stride() is", x.stride())
print("x.is_contiguous() is", x.is_contiguous())

print("i.shape is", i.shape)
print("i.stride() is",i.stride())
print("i.is_contiguous() is", i.is_contiguous())

# i.is_contiguous() = False，i的元素排列不同于原本的元素排列，因此.view(-1)会出错

t = torch.arange(24)
t = t.reshape(2,3,4)
# t = [[[0.,1.,2.,3.],[4.,5.,6.,7.],[8.,9.,10.,11.]],[[12.,13.,14.,15.],[16.,17.,18.,19.],[20.,21.,22.,23.,]]]
p = t.permute(2,0,1) # p.shape = (4,2,3)
print(p.shape)
# t[1,2,3]对应p[3,1,2]
print(t[1,2,3])
print(p[3,1,2])

k = i.contiguous()
print("k.contiguous is", k.contiguous)
print("k is", k)
print("k.shape is", k.shape)
print("k.stride() is", k.stride())
print("k.is_contiguous() is", k.is_contiguous())

i_1 = i.reshape(-1)
print("i.reshape(-1) is", i_1)
k_1 = k.view(-1)
print("k.view(-1) is", k_1)
