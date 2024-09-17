import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import AlexNet

import os
import json

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 保持比例
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
    # 增加batch_size维度，第0维
    img = torch.unsqueeze(img, dim=0)

    # 读入类别Json文件
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."

    with open(json_path, 'r') as f:
        class_indict = json.load(f)

    model = AlexNet(num_classes=5).to(device)

    weight_path = './AlexNet.pth'
    assert os.path.exists(weight_path), f"file: '{weight_path}' does not exist."
    model.load_state_dict(torch.load(weight_path, map_location=device))

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