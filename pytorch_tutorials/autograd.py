# coding:utf-8
"""
AutoGrad: automatic differentiation
https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py
"""

#%% import library
import torch

#%% constract tensor, and setting grad-calc using requires_grad=True
x = torch.ones(2,2, requires_grad=True)
print(x)

y = x + 2
print(y)
print(y.grad_fn)

#%% add operation
z = y * y * 3
out = z.mean()
print(z, out)

a = torch.randn(2,2)
a = ( (a*3) / (a-1) )
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b= (a*a).sum()
print(b.grad_fn)

#%% let's backpropagation
out.backward()
print(x.grad)

#%% look at an example of vector-jacobian product:
x = torch.randn(3, requires_grad=True)
y = x *2 
while y.data.norm() < 1000:
    y = y * 2

print(y)

#%%
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)

#%% .requires_grad=Trueのコードブロックをラップして、テンソルの履歴追跡からautogradを停止できる
print(x.requires_grad)
print((x**2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)