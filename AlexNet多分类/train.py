import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import torch.optim as optim
from model import AlexNet
import os
import json
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

data_transform = {
    "train": transforms.Compose([
             # 实现数据增强
             transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
    "val": transforms.Compose([
           transforms.Resize((224, 224)),   # 必须写为(224, 224)而不能是224(它将会保持原始的宽高比，而不是直接调整为正方形)
           transforms.ToTensor(),
           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
}