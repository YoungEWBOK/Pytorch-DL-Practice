import torch
import torch.nn as nn

'''
由于庞大网络难以通过小数量的数据集进行训练，
应当使用迁移学习，例如使用在ImageNet上预训练的权重
'''
# 预训练权重从以下链接下载
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N × 3 × 224 × 224
        x = self.features(x)
        # N × 512 × 7 × 7
        x = torch.flatten(x, start_dim=1)
        # N × (512 * 7 * 7)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

def make_features(cfg: list):
    layers = []
    in_channels = 3
    # 下一层的in_channels = 上一层的out_channel
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

# 使用 **kwargs 将多余的关键字参数传递给 VGG 类的构造函数
'''
当调用 vgg 函数时，可以传递 VGG 构造函数中定义的参数，
如 num_classes 或 init_weights，
例如：
    model = vgg(model_name="vgg16", num_classes=10, init_weights=True)
在这种情况下，num_classes=10 和 init_weights=True
会被传递给 VGG 类的构造函数 __init__ 方法，通过 **kwargs。
'''
def vgg(model_name='vgg16', **kwargs):
    assert model_name in cfgs, f"Warning: model {model_name} not in cfgs dict."
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)
    return model
