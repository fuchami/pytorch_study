# coding:utf-8

#%% import
import numpy as np
import torch
import torch.nn as nn

#%% テンソルの作成
"""
requires_grad=Falseだと微分の対象にならず勾配はNoneが返る
Fie-tuningで層のパラメータを固定したいときに便利
計算グラフを構築してbackward()を実行するとグラフを構築する各変数のgradに勾配が入る
"""
x = torch.tensor(1.0, requires_grad=True)
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

# 計算グラフを構築
y = w * x + b
# 勾配を計算
y.backward()

# 勾配の表示
print(x.grad)
print(w.grad)
print(b.grad)

#%% example1
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)

#%% example2
x = torch.tensor(2.0, requires_grad=True)
y = torch.exp(x)
y.backward()
print(x.grad)

#%% example3
x = torch.tensor(np.pi, requires_grad=True)
y = torch.sin(x)
y.backward()
print(x.grad)

#%% example4
x = torch.tensor(0.0, requires_grad=True)
y = (x - 4) * ( x ** 2 + 6)
y.backward()
print(x.grad)

#%% example5
x = torch.tensor(2.0, requires_grad=True)
y = (torch.sqrt(x) + 1) ** 3
y.backward()
print(x.grad)

#%% example6
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = (x + 2*y)**2
z.backward()
print(x.grad) # dz/dx
print(y.grad) # dz/dy

# lossを微分する
# バッチサンプル数=5, 入力の特徴量の次元数=3
x = torch.randn(5, 3)
# バッチサンプル数=5, 出力の特徴量の次元数=2
y = torch.randn(5, 2)

# Liner層を作成
linear = nn.Linear(3, 2)
# Linear層のパラメータ
print('w:', linear.weight)
print('b:', linear.bias)

# lossとoptimier
criterion = nn.MSELoss()
optimzier = torch.optim.SGD(linear.parameters(), lr=0.01)

# forward
pred = linear(x)

# loss = L
loss = criterion(pred, y)
print('loss:', loss)

# backpropagation
loss.backward()

# 勾配を表示
print('dL/dw:', linear.weight.grad)
print('dL/db:', linear.bias.grad)

# 勾配を用いてパラメータを更新
print('*** by hand')
print(linear.weight.sub(0.01 * linear.weight.grad))
print(linear.bias.sub(0.01 * linear.bias.grad))

# 勾配降下法
optimzier.step()

# 1ステップ更新後のパラメータを表示
# 上の式と結果が一致することがわかる
print('*** by optimizer.step(')
print(linear.weight)
print(linear.bias)