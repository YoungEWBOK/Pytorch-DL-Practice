import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from model import AlexNet

import os
import json
import sys

import numpy as np
import matplotlib.pyplot as plt

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}.")

    # 数据增强
    data_transform = {
        "train": transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
        "val": transforms.Compose([
            transforms.Resize((256, 256)),   # 保持比例，再裁剪
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    image_path = './flower_data'
    assert os.path.exists(image_path), f"{image_path} path does not exist."
    
    # 加载数据集
    def create_dataloader(root, transform, batch_size, shuffle, num_workers):
        dataset = datasets.ImageFolder(root=root, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return dataset, loader
    
    train_dataset, train_loader = create_dataloader(
        os.path.join(image_path, 'train'), data_transform['train'], batch_size=32, shuffle=True, num_workers=2)
    val_dataset, val_loader = create_dataloader(
        os.path.join(image_path, 'val'), data_transform['val'], batch_size=4, shuffle=False, num_workers=2)

    train_num = len(train_dataset)
    val_num = len(val_dataset)

    # 保存类别索引
    flower_list = train_dataset.class_to_idx
    # 反转字典键值对：将模型预测的类别标签（整数）转换回类别名称（字符串），以便于理解和展示结果。
    '''
    {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflower': 3, 'tulips': 4}
                                    ↓↓↓
    {0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflower', 4: 'tulips'}
    '''
    cla_dict = {val: key for key, val in flower_list.items()}
    with open('class_indices.json', 'w') as json_file:
        json.dump(cla_dict, json_file, indent=4)

    print(f"Using {train_num} images for training, {val_num} images for validating.")

    # 数据可视化
    test_data_iter = iter(val_loader)
    test_image, test_label = next(test_data_iter)

    def imshow(img):
        img = img / 2 + 0.5   # 反标准化
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    imshow(utils.make_grid(test_image))

    # 初始化模型
    model = AlexNet(num_classes=5, init_weights=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    # 训练和验证
    epochs = 10
    save_path = './AlexNet.pth'
    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # 直接转移到设备
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss:.3f}"

        # val
        model.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                outputs = model(val_images)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels).sum().item()

        val_acc = acc / val_num
        print(f'[epoch {epoch + 1}] train_loss: {running_loss / train_steps:.3f} val_acc: {val_acc:.3f}')

        if val_acc > best_acc:
            best_acc = val_acc
            try:
                torch.save(model.state_dict(), save_path)
            except Exception as e:
                print(f"Error saving model: {e}")

    print("Finished Training.")

if __name__ == '__main__':
    main()