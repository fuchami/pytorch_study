# coding:utf-8

#%% import libraries
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# Hyperparameters
num_epochs = 10
batch_size = 100
learning_rate = 0.001

#%% GPUモードにするためには明示的にコーディングが必要
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#%% MINST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root='../data/', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = dsets.MNIST(root='../data/', train=False, transform=transforms.ToTensor())

# Dataset Loading (Input Pipline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#%% build CNN model
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # CNNはブロック(Conv+BN+ReLU+Pool)にまとめてSequentialを使うと楽
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1) # Flatten?
        out = self.fc(out)
        return out

#%% テスト
model = CNN().to(device)
images, labels = iter(train_loader).next()
print(images.size())
images = images.to(device)
outputs = model(images)

#%% モデルオブジェクト作成 
model = CNN().to(device) #  モデルをto(devide)でGPUがに転送
criterion =  nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(model)

#%% define train valid flow
def train(train_loader):
    model.train()
    running_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device) # テンソルデータをGPUに転送
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
    
    # バッチ単位でのロスを算出
    train_loss = running_loss / len(train_loader)
    return train_loss

def valid(test_loader):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            predicted = outputs.max(1, keepdim=True)[1]
            correct += predicted.eq(labels.view_as(predicted)).sum().item()
            total += labels.size(0)
    
    val_loss = running_loss / len(test_loader)
    val_acc = correct / total

    return val_loss, val_acc

#%% let's train!
loss_list = []
val_loss_list = []
val_acc_list = []
for epoch in range(num_epochs):
    loss = train(train_loader)
    val_loss, val_acc =  valid(test_loader)

    print('epoch %d, loss: %.4f val_loss: %.4f val_loss: %.4f' % (epoch, loss, val_loss, val_acc))

    # logging
    loss_list.append(loss)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

"""save the trained model
np.save('loss_list.npy', np.array(loss_list))
np.save('val_loss_list.npy', np.array(val_loss_list))
np.save('acc_loss_list.npy', np.array(acc_loss_list))
"""

#%% plot learning curve
import matplotlib.pyplot as plt

plt.figure()
plt.plot(range(num_epochs), loss_list, 'r-', label='train_loss')
plt.plot(range(num_epochs), val_loss_list, 'b-', label='val_loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()

plt.figure()
plt.plot(range(num_epochs), val_acc_list, 'g-', label='val_acc')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.grid()