import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets
from src.model import AlexNet  # 导入定义好的AlexNet模型
from src.utils import get_transforms  # 导入定义好的数据预处理函数
import config
import os

def validate(model, val_loader, criterion, device):
    """
    在验证集上评估模型性能
    
    - model: 待评估的模型
    - val_loader: 验证集的 DataLoader
    - criterion: 损失函数
    - device: 计算设备（CPU 或 GPU）
    
    Returns:
    - 验证集上的准确率和平均损失
    """
    model.eval()  # 设置模型为评估模式，关闭 dropout 等训练特有的操作
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 禁用梯度计算，加快推理速度
        for inputs, targets in val_loader:  # 逐批处理验证数据
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据转移到计算设备
            outputs = model(inputs)  # 模型前向传播
            loss = criterion(outputs, targets)  # 计算损失

            val_loss += loss.item()  # 累加损失
            _, predicted = outputs.max(1)  # 获取预测值
            total += targets.size(0)  # 累加总样本数
            correct += predicted.eq(targets).sum().item()  # 累加正确预测的样本数

    val_loss /= len(val_loader)  # 计算平均损失
    val_acc = 100. * correct / total  # 计算准确率
    return val_acc, val_loss

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, writer):
    """
    训练模型并在验证集上评估最佳模型
    
    - model: 待训练的模型
    - train_loader: 训练集的 DataLoader
    - val_loader: 验证集的 DataLoader
    - criterion: 损失函数
    - optimizer: 优化器
    - device: 计算设备（CPU 或 GPU）
    - num_epochs: 训练的轮数
    - writer: TensorBoard 记录器
    
    Returns:
    - 最佳模型的保存路径
    """
    best_acc = 0.0  # 初始化最佳准确率为0
    best_model_path = config.BEST_MODEL_PATH  # 设置最佳模型的保存路径

    for epoch in range(num_epochs):  # 迭代每个训练周期
        model.train()  # 设置模型为训练模式，启用 dropout 等训练特有的操作
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):  # 逐批处理训练数据
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据转移到计算设备
            optimizer.zero_grad()  # 清空之前的梯度
            outputs = model(inputs)  # 模型前向传播
            loss = criterion(outputs, targets)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数

            running_loss += loss.item()  # 累加损失
            _, predicted = outputs.max(1)  # 获取预测值
            total += targets.size(0)  # 累加总样本数
            correct += predicted.eq(targets).sum().item()  # 累加正确预测的样本数

            if batch_idx % 10 == 0:  # 每10个批次记录一次损失
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, Running Loss: {running_loss / (batch_idx + 1):.4f}')
                writer.add_scalar('training loss', running_loss / 10, epoch * len(train_loader) + batch_idx)
                running_loss = 0.0

        train_acc = 100. * correct / total  # 计算训练集的准确率
        print(f'Epoch [{epoch + 1}/{num_epochs}] - Training Accuracy: {train_acc:.2f}%')
        writer.add_scalar('training accuracy', train_acc, epoch)  # 记录训练准确率

        # 在验证集上评估模型
        val_acc, val_loss = validate(model, val_loader, criterion, device)
        print(f'Epoch [{epoch + 1}/{num_epochs}] - Validation Accuracy: {val_acc:.2f}%, Validation Loss: {val_loss:.4f}')
        writer.add_scalar('validation accuracy', val_acc, epoch)  # 记录验证集准确率
        writer.add_scalar('validation loss', val_loss, epoch)  # 记录验证集损失

        # 如果当前模型在验证集上的准确率超过之前的最佳准确率，保存模型
        if val_acc > best_acc:
            best_acc = val_acc  # 更新最佳准确率
            torch.save(model.state_dict(), best_model_path)  # 保存最佳模型

    return best_model_path  # 返回最佳模型的路径

if __name__ == "__main__":
    transform = get_transforms()  # 获取图像预处理步骤

    # 创建训练集和验证集的 DataLoader
    train_dataset = datasets.ImageFolder(root=config.TRAIN_DATASET_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_dataset = datasets.ImageFolder(root=config.VAL_DATASET_DIR, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

    device = torch.device(config.DEVICE)  # 选择设备（GPU 或 CPU）
    model = AlexNet(num_classes=2).to(device)  # 实例化模型并转移到设备

    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM)  # 使用SGD优化器

    writer = SummaryWriter(config.TENSORBOARD_LOG_DIR)  # 初始化 TensorBoard 记录器
    best_model_path = train(model, train_loader, val_loader, criterion, optimizer, device, config.NUM_EPOCHS, writer)
    writer.close()  # 训练结束后关闭 TensorBoard 记录器

    print(f'Best model saved at: {best_model_path}')  # 输出最佳模型的保存路径