'''
Dropout 层通常用于防止过拟合，通过随机“丢弃”一些神经元来减少模型对特定特征的依赖
通常在全连接层之后添加 Dropout 层，在激活函数之后
在卷积神经网络（CNN）中，这通常是在卷积层和池化层之后的全连接层之间
避免在输出层之前使用 Dropout

在 AlexNet 中，Dropout 层被用于全连接层，以减少过拟合
由于 AlexNet 包含大量的参数，如果不采取措施，模型很容易在训练数据上过拟合
Dropout 通过在训练过程中随机“丢弃”一些神经元，降低了模型对特定神经元的依赖，从而提升了模型的泛化能力
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    # num_classes根据实际数据集的分类数传入
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        # 数据集小 + 加快训练速度 => 参数设置为原论文中的一半
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=2),   # [3, 224, 224] to [48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),   # [48, 27, 27]
            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, padding=2, stride=1),   # [128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),   # [128, 13, 13]
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1),   # [192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),   # [192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1),   # [128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0)   # [128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128 * 6 * 6, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=2048, out_features=num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
    
    # 实际上Pytorch会自动实现此操作
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)