import torch
import torch.nn as nn
import torch.nn.functional as F

# 将卷积+ReLU合成一个模块
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
    
# 每个branch里kernel_size和padding的不同值是为了确保沿深度方向可以拼接(包括pool的branch)
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3reduce, ch3x3, ch5x5reduce, ch5x5, pool_proj):
        super(Inception, self).__init__()

        # 1x1 conv branch
        self.branch1x1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        # 1x1 -> 3x3 conv branch
        self.branch3x3 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3reduce, kernel_size=1),
            BasicConv2d(ch3x3reduce, ch3x3, kernel_size=3, padding=1)
        )

        # 1x1 -> 5x5 conv branch
        self.branch5x5 = nn.Sequential(
            # 为了适配Pytorch给出的预训练权重，把5x5卷积改成两个3x3卷积
            BasicConv2d(in_channels, ch5x5reduce, kernel_size=1),
            BasicConv2d(ch5x5reduce, ch5x5, kernel_size=3, padding=1),
            BasicConv2d(ch5x5, ch5x5, kernel_size=3, padding=1)
        )

        # 3x3 -> 1x1 conv branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),   # MaxPool不改变channels
            BasicConv2d(in_channels, pool_proj, kernel_size=1)   # 1x1卷积不改变H和W，但可以改变channels
        )
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        # 沿深度方向拼接
        return torch.cat(outputs, dim=1)
    
# 辅助分类器模块Auxiliary Classifiers
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, out_channels=128, kernel_size=1)   # output[batch_size, 128, 4, 4]

        # 128 * 4 * 4 = 2048
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, start_dim=1)
        x = F.dropout(x, p=0.5, training=self.training)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, p=0.5, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        return x
    
# 为了使思路更清晰，最后定义GooGleNet模块
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        # 布尔型变量，决定了是否加上辅助分类器
        self.aux_logits = aux_logits

        # padding是推算出来的，(224 - 7 + 2*3) / 2 + 1 = 112.5，向下取整
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # ceil_mode=True则使用向上取整来计算输出尺寸
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            # aux1接在inception4a的输出
            self.aux1 = InceptionAux(512, num_classes)
            # aux2接在inception4d的输出
            self.aux2 = InceptionAux(528, num_classes)

        # 自适应平均池化会根据输入特征图的尺寸自动调整池化窗口的大小和步幅，确保输出特征图的大小为目标尺寸
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        if self.train and self.aux_logits:   # model.train()时才考虑辅助分类器
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:   # model.eval()时不考虑辅助分类器
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, start_dim=1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000(num_classes)

        if self.training and self.aux_logits:
            return x, aux2, aux1
        # 非model.training()模式只需要返回主分类器x即可
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
