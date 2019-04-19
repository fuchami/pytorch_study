# coding:utf-8
'''
https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html#sphx-glr-beginner-blitz-data-parallel-tutorial-py

・pytorch on GPU
devide = torch.device("cuda:0)
model.to(device)

・tensor copy to GPU
mytensor = my_tensor.to(device)

モデルの並列実行
model = nn.DataParallel(model)

'''

#%% import
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# parameters and DataLoaders
input_size = 5
outpus_size = 2

batch_size = 30
data_size = 100

#%% define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('run on ' , device)

#%% dammy datasets
class RandomDataset(Dataset):
    
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)
    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                        batch_size=batch_size, shuffle=True)

#%% define model
class Model(nn.Module):

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print('\tIn Model: input size', input.size(), "output size", output.size())

        return output

#%% data parallel 
model =  Model(input_size, outpus_size)
if torch.cuda.device_count() > 1:
    print("let's use ", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

model.to(device)

#%% モデルの実行
for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outsize: input size", input.size(),
            "outpus_size", output.size())
