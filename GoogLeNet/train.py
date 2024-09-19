'''
不使用预训练权重，从头开始训练
'''
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from model import GoogLeNet
from tqdm import tqdm

import os
import sys
import json

def main():
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
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

    # 未使用预训练的GoogLeNet，从头开始训练
    model = GoogLeNet(num_classes=5, aux_logits=True, init_weights=True)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    epochs = 30
    best_acc = 0.0
    save_path = './googleNet.pth'
    train_steps = len(train_loader)

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits, aux_logits2, aux_logits1 = model(images.to(device))   # 只使用最后的输出
            loss0 = criterion(logits, labels.to(device))
            loss1 = criterion(aux_logits1, labels.to(device))
            loss2 = criterion(aux_logits2, labels.to(device))
            loss = loss0 + loss1 * 0.3 + loss2 * 0.3
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss:.3f}"

        # 验证阶段
        model.eval()
        acc = 0.0
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