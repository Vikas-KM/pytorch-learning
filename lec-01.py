# Pytorch basics - Tensors and Gradients

import torch

t1 = torch.tensor(4.0)
print(t1)
print(t1.dtype)  # prints float32
print(t1.shape)  # prints single empty braces

t2 = torch.tensor([1, 2, 3, 4])
print(t2)
print(t2.dtype)  # prints int64
print(t2.shape)  # prints 4

# tensor needs all values same, so all are converted to float
t3 = torch.tensor([1.0, 2, 3, 4, 5])
print(t3)
print(t3.dtype)  # prints float32
print(t3.shape)

t4 = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
print(t4)
print(t4.dtype)
print(t4.shape)

# Tensors without appropriate size
# t5 = torch.tensor([
#     [1,2,3],
#     [4,5]
# ])
# print(t5)

x = torch.tensor(5.)
w = torch.tensor(3., requires_grad=True)
b = torch.tensor(7., requires_grad=True)

y = w*x+b
print(y)
y.backward()
print(x.grad)
print(w.grad)
print(b.grad)
