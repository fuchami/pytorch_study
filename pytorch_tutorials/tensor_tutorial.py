# coding:utf-8
'''
What is PyTorch?
https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py
'''
#%% import library
from __future__ import print_function
import torch

#%% uninitialized 5x3matrix
x = torch.empty(5,3)
print(x)

#%% randomly initialied 5x3matrix
x = torch.rand(5,3)
print(x)

#%% zeros and dtype long initialized matrix
x = torch.zeros(5,3, dtype=torch.long)
print(x)

#%% directly tensor 
x = torch.tensor([5,5, 3])
print(x)

'''
既存のテンソルに基づいてテンソルを作成
これらのメソッドは、ユーザによって新しい値が提供されない限り、入力テンソルのプロパティ、例えばdtypeを再利用します。
'''
#%%
x = x.new_ones(5,3, dtype=torch.double)
print(x)
x = torch.rand_like(x, dtype=torch.float)
print(x)

#%% get size
print(x.size())

#%% addition
y = torch.rand(5,3)
print(x+y)
print(torch.add(x,y))

result = torch.empty(5,3)
torch.add(x,y, out=result)
print(result)
# adds x to y
y.add_(x)
print(y)

#%% we can user standart numpy-like indexing
print(x[:,1])
#%% resize/reshape
x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1, 8) # -1は他の次元数から自動で推測する
print(x.size(), y.size(), z.size())

#%% .item()でpythonの数値として値を取得できる
x = torch.randn(1)
print(x)
print(x.item())

#%% conterting a Torch tensor to a numpy array
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(2)
print(a)
print(b)

#%% converting a numpy array to Torch Tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
