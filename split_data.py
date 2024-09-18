'''
按9:1的比例划分数据集
'''
import os
import random
from shutil import copy, rmtree

def make_dirs(dir_path):
    """
    创建目录。如果目录已存在，则不执行任何操作。

    参数:
    dir_path (str): 要创建的目录路径
    """
    # 如果目录不存在则创建
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def split_dataset(origin_path, train_path, val_path, split_rate=0.1):
    """
    按照给定的比例将数据集划分为训练集和验证集，并将文件复制到相应的目录中。

    参数:
    origin_path (str): 原始数据集路径，包含类别文件夹
    train_path (str): 训练集文件夹路径
    val_path (str): 验证集文件夹路径
    split_rate (float): 验证集占整个数据集的比例，默认为0.1
    """
    # 设置随机种子，确保结果的可复现性
    random.seed(0)
    
    # 创建训练集和验证集的根目录
    make_dirs(train_path)
    make_dirs(val_path)
    
    # 获取原始数据集中的所有类别文件夹
    classes = [cla for cla in os.listdir(origin_path) if os.path.isdir(os.path.join(origin_path, cla))]
    
    for cla in classes:
        # 对每个类别创建对应的训练和验证目录
        make_dirs(os.path.join(train_path, cla))
        make_dirs(os.path.join(val_path, cla))
        
        cla_path = os.path.join(origin_path, cla)
        images = os.listdir(cla_path)  # 获取类别文件夹中的所有图片
        num = len(images)
        
        # 随机选择验证集的索引
        eval_index = random.sample(images, k=int(num * split_rate))
        
        for index, image in enumerate(images):
            # 构建原始图片的完整路径
            src_path = os.path.join(cla_path, image)
            
            # 根据图片索引将其分配到验证集或训练集
            if image in eval_index:
                # 验证集路径
                dest_path = os.path.join(val_path, cla)
            else:
                # 训练集路径
                dest_path = os.path.join(train_path, cla)
            
            try:
                # 复制图片到相应的目录
                copy(src_path, dest_path)
                # 显示进度
                print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")
            except Exception as e:
                # 处理复制过程中可能出现的错误
                print(f"\nFailed to copy {src_path} to {dest_path}: {e}")
        
        print()  # 换行，方便下一类别显示

def main():
    """
    主函数，设置数据集路径并调用数据集划分函数。
    """
    # 获取当前工作目录
    cwd = os.getcwd()   # 例如我的是'C:\Users\25705\Desktop\Pytorch-DL-Practice\AlexNet多分类'
    
    # 设置数据集的根目录
    data_root = os.path.join(cwd, 'flower_data')
    
    # 设置原始花卉数据集路径
    origin_flower_path = os.path.join(data_root, 'flower_photos')
    
    # 检查原始数据集路径是否存在
    assert os.path.exists(origin_flower_path), f"Path '{origin_flower_path}' does not exist."

    # 设置训练集和验证集的路径
    train_root = os.path.join(data_root, 'train')
    val_root = os.path.join(data_root, 'val')

    # 调用函数划分数据集
    split_dataset(origin_flower_path, train_root, val_root, split_rate=0.1)

    print("Data-Processing Done.")

if __name__ == '__main__':
    # 程序入口，调用主函数
    main()