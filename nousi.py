# coding:utf-8
"""
とりあえず脳死で書いてみるpytorch
https://qiita.com/wataoka/items/c72261bae7d7ef300e29

"""
#%% import 
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#%% define model
"""
__init__とforwardを実装すればok
"""
class MNISTConvNet(nn.Module):

    # 使用する層を書く(関数は書かないスタンス)
    def __init__(self):
        super(MNISTConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
    
    # モデルの設計
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

#%% ハイパラ
epochs = 20
batch_size = 128

# model
model = MNISTConvNet()
print(model) #モデルのsummaryを出力

#%% load datasets
# trainsformオブジェクトの生成、正則化などを指定
transform  = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5), (0.5, ))]
)

trainset = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform)

# 学習時のfor文のイテレータとしてぶちこまれるもの。(inputs, labels)タプルをイテレートしてくれる
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

#%% define loss function & optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#%% let's train!
for epoch in range(epochs):
    running_loss = 0.0

    for i, data in enumerate(trainloader):
        inputs, labels = data
        # optimizerを初期化
        optimizer.zero_grad()
        # 入力画像をモデルに入力し、出力値をoutputsに
        outputs = model(inputs)
        # 損失値を計算
        loss = criterion(outputs, labels)
        # 逆誤差伝播
        loss.backward()
        # 最適化
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print('Epoch:{}/{} loss: {:.3f}'.format(epoch+1, epochs, running_loss/100))
            running_loss = 0.0

print('Finished Training')


