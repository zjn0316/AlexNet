import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from models.AlexNet import AlexNet
from utils.data_loader import get_data_loaders
from utils.time import timer_decorator, Timer, get_current_time
from utils.config import load_config, get_device
from utils.train import train
from utils.test import test


def main():
    # 加载配置
    config = load_config()
    print(f"配置加载完成")

    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 提取训练参数
    batch_size = config['train']['batch_size']
    epochs = config['train']['epochs']
    learning_rate = config['train']['learning_rate']
    optimizer_name = config['train']['optimizer']
    loss_function = config['train']['loss_function']

    # 打印训练配置
    print(f"\n训练配置:")
    print(f"  批次大小: {batch_size}")
    print(f"  训练轮数: {epochs}")
    print(f"  学习率: {learning_rate}")
    print(f"  优化器: {optimizer_name}")
    print(f"  损失函数: {loss_function}")


    # 数据加载计时
    with Timer("数据加载"):
        train_loader, test_loader = get_data_loaders(batch_size=64)

    # 初始化模型
    model = AlexNet().to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练计时
    with Timer("模型训练"):
        train(model, train_loader, criterion, optimizer, device, epochs,batch_size)


    # 测试计时
    with Timer("模型测试"):
        test(model, test_loader, criterion, device,batch_size)

    # 确保所有输出完成后再保存模型
    sys.stdout.flush()
    time.sleep(0.1)

    # 保存模型
    torch.save(model.state_dict(), 'checkpoints/alexnet_fashion_mnist.pth')
    print("Model saved to 'alexnet_fashion_mnist.pth'")


if __name__ == "__main__":
    main()
