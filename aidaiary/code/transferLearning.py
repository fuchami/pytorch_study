# coding:utf-8

#%% import library
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

import os,time,copy
import numpy as np
import matplotlib.pyplot as plt

#%% download datasets
# !wget https://download.pytorch.org/tutorial/hymenoptera_data.zip -P data/

#%% load image from directory
data_dir = os.path.join('./data', 'hymenoptera_data')
image_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'))
print(len(image_dataset)) #224枚の訓練データ
image, label = image_dataset[0] # 0番目の画像とラベル

''' Augmented Data '''

#%% Random Crop
plt.figure()
plt.imshow(image)

t = transforms.RandomResizedCrop(224)
trans_image = t(image)

plt.figure()
plt.imshow(trans_image)


#%% Random Crop
plt.figure()
plt.imshow(image)

t = transforms.RandomHorizontalFlip()
trans_image = t(image)

plt.figure()
plt.imshow(trans_image)

#%% データ変換関数の作成
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256,256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ]),
}

#%% DataSetとDataLoader
data_dir = os.path.join('./data', 'hymenoptera_data')
image_datasets = { x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
            for x in ['train', 'val'] }
dataloaders = { x: torch.utils.data.DataLoader(image_datasets[x],
                                batch_size=4,
                                shuffle=True,
                                num_workers=4) for x in ['train', 'val']}
dataset_sizes = { x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(class_names)

#%% 訓練画像の可視化
def imshow(images, title=None):
    # データ変換ToTensor()したのでnumpy()でndarrayに戻す
    images = images.numpy().transpose((1,2,0)) #(h, w, c)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # 元ピクセル値に戻す(標準偏差をかけて平均を足す)
    images = std * images + mean
    images = np.clip(images, 0, 1)
    plt.imshow(images)
    if title is not None:
        plt.title(title)

images, classes = next(iter(dataloaders['train']))
print(images.size(), classes.size())
images = torchvision.utils.make_grid(images)
imshow(images, title=[class_names[x] for x in classes])

#%% 訓練用関数の定義
use_gpu = torch.cuda.is_available()

def train_mode(model, criterion, optimizer, scheduler,num_epochs=25):
    since = time.time()

    best_mode_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs -1))
        print('-' * 10)

        # 各エポックで訓練+バリデーションを実行
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True) # training mode
            else:
                # model.eval()じゃなくてこっちでも良い
                model.train(False)
            
            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                inputs, labels = data

                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                # statiscs
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # サンプル数で割って平均を求める
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deepcopy the model
            # 精度が改善したらモデルを保存する
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # best_model_wts = copy.deepcopy(model.state_dic())
        
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    print('Best val acc: {:.4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_mode_wts)
    return model

#%% 学習済みモデルをFine-tuning
model_ft = models.resnet18(pretrained=True)
# print(model_ft)

num_features = model_ft.fc.in_features
print(num_features)

# fc層を置き換える
model_ft.fc = nn.Linear(num_features, 2)
print(model_ft)

if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# 7エポックごとに学習率を0.1倍する
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_mode(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=35)
# torch.save(model_ft.state_dict(), 'model_ft.pkl')


"""
学習済みの重みを固定(ConvNet as fixed feature extractor)
レイヤの重みはすべて固定でFine-tuning
"""

# 訓練済みResNet18をロード


