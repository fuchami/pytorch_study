# coding:utf-8
"""
VGG-16 finetuning
"""

#%% import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

import json
import numpy as np
from PIL import Image

#%% load VGG-16 model
vgg16 = models.vgg16(pretrained=True)
print(vgg16) # モデル構造を表示
# 学習時と推論時でモードが違う
vgg16.eval()

#%% データの前処理(Imagenet学習時と同じデータ標準化)
normalize = transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225])

# 256x256にリサイズし画像中心の224x224をクロップ
# テンソルに変換
# ImageNetの訓練データのmeanを引いてstdで割る(標準化)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

#%% test plot image
img = Image.open('./data/dog.jpg')
img_tensor = preprocess(img)
print(img_tensor.shape) # 画像が3次元テンソルに変換

preprocess2 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224)
])
trans_img = preprocess2(img)
print(type(trans_img))
trans_img

#%% バッチサイズの次元を追加した4次元テンソルにする
img_tensor.unsqueeze_(0)
print(img_tensor.size())

out = vgg16(Variable(img_tensor))
print(out.size())

#%% 
'''
outはsoftmaxをとる前なので確率になっていない(足して1.0にならない)
が、分類するときは出力が最大のクラスに分類すればよい
'''
result = np.argmax(out.data.numpy())
out.topk(5)

#%% ImageNetの1000クラスのラベル情報を読み込む
class_index = json.load(open('./aidaiary/imagenet_class_index.json', 'r'))
print(class_index)

#%% 
labels = {int(key):value for(key, value) in class_index.items()}
print(labels[0])
print(labels[1])

print(labels[result])

#%% 関数化して評価してみる
def predict(image_file):
    img = Image.open(image_file)
    img_tensor = preprocess(img)
    img_tensor.unsqueeze_(0)
    
    out = vgg16(Variable(img_tensor))

    # 出力を確立にする
    out = nn.functional.softmax(out, dim=1)
    out = out.data.numpy()

    maxid = np.argmax(out)
    maxprob = np.max(out)
    label =labels[maxid]
    return img, label, maxprob

img, label, prob = predict('./data/dog.jpg')
print(label, prob)
img


