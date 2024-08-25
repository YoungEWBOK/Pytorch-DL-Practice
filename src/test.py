import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from src.model import AlexNet  # 导入定义好的AlexNet模型
from src.utils import get_transforms  # 导入定义好的数据预处理函数
import config

def test(model, test_loader, criterion, device):
    """
    在测试集上评估模型性能
    
    - model: 待评估的模型
    - test_loader: 测试集的 DataLoader
    - criterion: 损失函数
    - device: 计算设备（CPU 或 GPU）
    
    Returns:
    - 测试集上的准确率和平均损失
    """
    model.eval()  # 设置模型为评估模式，关闭 dropout 等训练特有的操作
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 禁用梯度计算，加快推理速度
        for inputs, targets in test_loader:  # 逐批处理测试数据
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据转移到计算设备
            outputs = model(inputs)  # 模型前向传播
            loss = criterion(outputs, targets)  # 计算损失
            test_loss += loss.item()  # 累加损失
            _, predicted = outputs.max(1)  # 获取预测值
            total += targets.size(0)  # 累加总样本数
            correct += predicted.eq(targets).sum().item()  # 累加正确预测的样本数

    accuracy = 100. * correct / total  # 计算准确率
    print(f'Test Accuracy: {accuracy:.2f}%')  # 打印测试集的准确率
    return accuracy, test_loss  # 返回测试集的准确率和损失

if __name__ == "__main__":
    device = torch.device(config.DEVICE)  # 选择设备（GPU 或 CPU）
    model = AlexNet(num_classes=2).to(device)  # 实例化模型并转移到设备
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH))  # 加载最佳模型的权重

    transform = get_transforms()  # 获取图像预处理步骤
    test_dataset = datasets.ImageFolder(root=config.TEST_DATASET_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    
    criterion = torch.nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    test_acc, test_loss = test(model, test_loader, criterion, device)  # 评估测试集
    print(f"Test Loss: {test_loss:.4f}")  # 打印测试集的损失