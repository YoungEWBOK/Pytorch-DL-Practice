from datetime import datetime
import torch
import os

# 数据集路径配置
TRAIN_DATASET_DIR = './data/train'  # 训练集路径
VAL_DATASET_DIR = './data/val'      # 验证集路径
TEST_DATASET_DIR = './data/test'    # 测试集路径

# 模型保存路径配置
MODEL_DIR = './weights'  # 模型权重保存的文件夹
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)  # 如果文件夹不存在，则创建该文件夹

# 设置模型保存路径，包含日期和时间信息，以便区分不同的训练
BEST_MODEL_PATH = os.path.join(MODEL_DIR, f'alexnet_best_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')

# 训练参数配置
NUM_EPOCHS = 10            # 训练的轮数
BATCH_SIZE = 32            # 每个batch的大小
LEARNING_RATE = 0.001      # 学习率
MOMENTUM = 0.9             # 优化器的动量参数

# TensorBoard 日志路径
TENSORBOARD_LOG_DIR = 'runs'

# 设备配置，优先使用GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'