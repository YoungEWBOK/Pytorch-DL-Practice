import torch
import config  # 导入配置文件
from src.model import AlexNet  # 从src文件夹导入模型
from src.utils import get_transforms  # 从src文件夹导入数据预处理函数
from src.train import train  # 从src文件夹导入训练函数
from src.test import test  # 从src文件夹导入测试函数
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter

def main():
    """
    主函数，用于训练和测试模型，结合训练集、验证集和测试集的功能
    """
    device = torch.device(config.DEVICE)  # 选择设备（GPU 或 CPU）
    
    # 定义数据预处理
    transform = get_transforms()  # 获取图像预处理步骤

    # 创建训练集和验证集的 DataLoader
    train_dataset = datasets.ImageFolder(root=config.TRAIN_DATASET_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_dataset = datasets.ImageFolder(root=config.VAL_DATASET_DIR, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

    # 实例化模型并转移到设备
    model = AlexNet(num_classes=2).to(device)

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM)  # 使用SGD优化器

    # 初始化 TensorBoard 记录器
    writer = SummaryWriter(config.TENSORBOARD_LOG_DIR)

    # 训练模型并保存最佳权重
    best_model_path = train(model, train_loader, val_loader, criterion, optimizer, device, config.NUM_EPOCHS, writer)
    
    # 训练结束后关闭 TensorBoard 记录器
    writer.close()

    # 加载最佳模型的权重
    model.load_state_dict(torch.load(best_model_path))

    # 测试集数据加载
    test_dataset = datasets.ImageFolder(root=config.TEST_DATASET_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

    # 在测试集上评估模型
    test_acc, test_loss = test(model, test_loader, criterion, device)

    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()