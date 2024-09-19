'''
如果最优权重是由预训练权重迁移学习训练出来的
则注意创建模型时需要注释原代码，使用另一条代码
fc.weight 和 fc.bias 是关键的权重参数，对模型的最终输出有重要影响
(其他4个辅助分类器参数在预测阶段不会被用到)
'''
import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from model import GoogLeNet

import os
import json

def main():
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

    data_transform = transforms.Compose([
        transforms.CenterCrop(224),     # 裁剪到 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载一张图片
    img_path = './tulip.jpg'
    assert os.path.exists(img_path), f"file: '{img_path}' does not exist."
    img = Image.open(img_path)

    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # 增加batch_size维度，第0维，[3, 224, 224] → [1, 3, 224, 224]
    '''
    对于单张图像，我们通常会将它的形状调整为 [1, channels, height, width]，
    以匹配模型的输入要求，其中 1 是 batch_size，表示只有一张图像。
    '''
    img = torch.unsqueeze(img, dim=0)

    # 读入类别Json文件
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."

    with open(json_path, 'r') as f:
        class_indict = json.load(f)

    # 此处需要进行model的选择
    # model = GoogLeNet(num_classes=5, aux_logits=False).to(device)
    model = models.googlenet(num_classes=5).to(device)

    weight_path = './googleNet.pth'
    assert os.path.exists(weight_path), f"file: '{weight_path}' does not exist."
    model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)

    model.eval()
    with torch.no_grad():
        # 预测类别
        output = torch.squeeze(model(img.to(device))).to(device)
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = f"class: {class_indict[str(predict_cla)]} prob: {predict[predict_cla].numpy():.3}"

    plt.title(print_res)
    for i in range(len(predict)):
        print(f"class: {class_indict[str(i)]:10} prob: {predict[i].numpy():.3}")

    plt.show()

if __name__ == '__main__':
    main()