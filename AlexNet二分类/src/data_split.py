'''使用os和shutil库划分数据
   shutil.copy 是用来复制文件的函数
   它将文件从源路径复制到目标路径'''
import os
import shutil
import random

# 定义数据集路径，'./' 表示当前工作目录，直接写成 'origin_data'也可以
data_dir = "./origin_data"
categories = ['cat', 'dog']

# 定义划分后的输出文件夹
output_dir = "./data"

# 递归创建所有必要的目录，exist_ok设为True保证目录已存在时不报错
os.makedirs(output_dir, exist_ok=True)

# 定义数据集划分的比例
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# 创建train、val、test三个文件夹
for split in ['train', 'val', 'test']:
    for category in categories:
        # os.path.join() 函数会自动根据操作系统的路径分隔符(/ 或 \)将路径组件连接起来
        os.makedirs(os.path.join(output_dir, split, category), exist_ok=True)

# 按类别划分数据集
for category in categories:
    category_dir = os.path.join(data_dir, category)
    # os.listdir(category_dir) 返回一个包含目录中所有文件和子目录名称的列表，如['1.jpg', '2.jpg', '3.jpg']
    images = os.listdir(category_dir)
    random.shuffle(images)

    # 设置images列表的划分点
    train_split = int(train_ratio * len(images))
    val_split = int((train_ratio + val_ratio) * len(images))

    train_images = images[:train_split]
    val_images = images[train_split:val_split]
    test_images = images[val_split:]

    # 移动文件到训练集，img变量依次取train_images列表中的每一个图像文件名，如 '1.jpg'、'2.jpg' 等
    for img in train_images:
        shutil.copy(os.path.join(category_dir, img), os.path.join(output_dir, 'train', category, img))

    # 移动文件到验证集，img变量依次取val_images列表中的每一个图像文件名
    for img in val_images:
        shutil.copy(os.path.join(category_dir, img), os.path.join(output_dir, 'val', category, img))

    # 移动文件到测试集，img变量依次取test_images列表中的每一个图像文件名
    for img in test_images:
        shutil.copy(os.path.join(category_dir, img), os.path.join(output_dir, 'test', category, img))

print("数据集划分完成")

