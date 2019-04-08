# coding:utf-8
"""
線形回帰(Linear Regression)
"""

#%% import
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# hyper parameters
input_size = 1
output_size = 1
num_epochs = 600
learning_rate = 0.001

#%% toy dataset, 15samples, 1features
x_train = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                    7.042, 10.791, 5.313, 7.997, 3.1], dtype=np.float32)

y_train = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                    2.827, 3.465, 1.65, 2.904, 1.3], dtype=np.float32)

# nn.Linearへの入力は(N, *, in_features)のためreshape
x_train = x_train.reshape(15, 1)
y_train = y_train.reshape(15, 1)

#%% linear regression model
# nn.Moduleを継承したクラスを作成
class LinearRegression(nn.Module):

    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        out = self.linear(x)
        return out

model = LinearRegression(input_size, output_size)

# loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr=learning_rate)

#%% train the model
for epoch in range(num_epochs):
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    # 各エポックで勾配をクリアすること！
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    # パラメータ更新
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, loss.item()))

# save the model
# torch.save(model.state_dict(), 'model.pkl')

#%% plot the graph
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()