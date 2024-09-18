'''
通过 load_state_dict 加载预训练的权重
冻结卷积层参数，以避免它们在训练过程中更新(可选)
对最后的全连接层进行微调，使其适应特定的分类任务
'''
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from model import vgg

import os
import sys
import json

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device.")

    data_transform = {
        'train': transforms.Compose([
                 transforms.RandomResizedCrop(224),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    image_path = './flower_data'
    assert os.path.exists(image_path), f"{image_path} path does not exist."

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'train'), transform=data_transform['train'])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # 将字典写入Json文件
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    num_workers = 2
    print(f"Using {num_workers} dataloader workers every process.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'val'), transform=data_transform['val'])
    val_num = len(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"Using {train_num} images for training, {val_num} images for validation.")

    model_name = 'vgg16'

    # 创建模型实例，但不初始化权重
    model = vgg(model_name=model_name, num_classes=5, init_weights=False).to(device)

    # 加载预训练权重
    # current_dir = os.getcwd()
    # pretrain_weights_path = os.path.join(current_dir, './vgg16_pretrained.pth')
    pretrain_weights_path = './vgg16_pretrained.pth'
    assert os.path.exists(pretrain_weights_path), f"File: '{pretrain_weights_path}' does not exist."
    pretrained_dict = torch.load(pretrain_weights_path, map_location=device)

    # 获取模型中与预训练模型匹配的参数
    model_dict = model.state_dict()
    # 过滤掉分类层（'classifier.6'）的参数，因为它与我们当前的分类任务不匹配
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'classifier.6' not in k}

    # 更新现有模型的权重
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # 替换分类层（全连接层）为适应新的类别数
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 5)  # 替换最后一层为5分类

    # 将模型移动到设备上
    model.to(device)

    # 冻结卷积层参数(可选)
    for param in model.features.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    epochs = 30
    best_acc = 0.0
    save_path = './{}Net.pth'.format(model_name)
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = model(images.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss:.3f}"

        # 验证阶段
        model.eval()
        acc = 0.0   # 累计准确率
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_acc = acc / val_num
        print(f"[epoch {epoch + 1}] train_loss: {running_loss / train_steps:.3f} val_accuracy: {val_acc:.3f}")

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print("Finished Training.")

if __name__ == '__main__':
    main()