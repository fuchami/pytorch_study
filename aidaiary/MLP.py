# coding:utf-8
'''
Multilayer Perceptron with MNIST
'''

#%% import and define hyperparameters
import torch
import torch.nn as nn

import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# Hyerparameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 50
batch_size = 100
learning_rate = 0.001

#%% MNIST Datasets (Images and Labels)
train_dataset = dsets.MNIST(root='../data/', train=True, transform=transforms.ToTensor(), download=True)

test_datasest = dsets.MNIST(root='../data/', train=False, transform=transforms.ToTensor())

# Dataset Loader (Input Pipline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_datasest, batch_size=batch_size, shuffle=False)

#%% Datasetのlenはサンプル数、DataLoaderはミニバッチ数を返す
print(len(train_dataset))
print(len(test_datasest))
print(len(train_loader))
print(len(test_loader))

#%% Dataloaderから1バッチ分のデータを取り出すにたite()で囲んでnext()
image, label = iter(train_loader).next()
print(type(image))
print(type(label))
print(image.size())
print(label.size())

#%% 可視化
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    npimg = img.numpy()
    # [c, h, w] => [h, w, c]
    plt.imshow(np.transpose(npimg, (1,2,0)))

images, labels = iter(train_loader).next()
images, labels = images[:25], labels[:25]
imshow(torchvision.utils.make_grid(images, nrow=5, padding=1))
plt.axis('off')

#%% build MLP
class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # print( 'MLP input_size:' ,x.size())
        out = self.fc1(x)
        # print('MLP fc1 output size: ', out.size())
        out = self.relu(out)
        # print('MLP relu output size: ', out.size())
        out = self.fc2(out)
        # print('MLP fc2 output size: ', out.size())
        return out

model = MultiLayerPerceptron(input_size, hidden_size, num_classes)

#%% テスト
image, label = iter(train_loader).next()
print("before view: ", image.size())
image = image.view(-1, 28*28) #reshape的な関数
print("after view: ", image.size())
output = model(image)
print(output.size())

#%% define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#%% define trian valid function
def train(train_loader):
    model.train()
    running_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28*28)
        
        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        running_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    train_loss = running_loss / len(train_loader)

    return train_loss

def valid(test_loader):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    #  評価時に勾配は不要なのでno_grad()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.view(-1, 28*28)

            outputs = model(images)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
        val_loss = running_loss / len(test_loader)
        val_acc = float(correct) / total
        return val_loss, val_acc

loss_list = []
val_loss_list = []
val_acc_list = []
for epoch in range(num_epochs):
    loss = train(train_loader)
    val_loss, val_acc = valid(test_loader)

    print('epoch %d, loss: %.4f val_loss: %.4f val_acc: %.4f' % (epoch, loss, val_loss, val_acc))

    # logging
    loss_list.append(loss)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

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