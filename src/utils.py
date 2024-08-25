import torchvision.transforms as transforms

def get_transforms():
    """
    获取图像数据的预处理操作，适用于训练、验证和测试集。
    
    主要操作包括：
    - 调整图像大小为224x224（与AlexNet输入层匹配）
    - 转换为Tensor
    - 对图像进行归一化，使用的均值和标准差与ImageNet数据集一致

    :return: 一个组合了多种变换操作的Compose对象
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小为224x224像素
        transforms.ToTensor(),  # 将图像转换为Tensor类型
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 使用ImageNet数据集的均值进行归一化
                             std=[0.229, 0.224, 0.225])   # 使用ImageNet数据集的标准差进行归一化
    ])
    return transform