import torch
import sys
from tqdm import tqdm


def test(model, test_loader, criterion, device,batch_size=64):
    """
    在测试集上评估模型性能

    参数:
        model: 训练好的AlexNet模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 计算设备（'cuda'或'cpu'）

    返回:
        test_loss: 测试集平均损失
        accuracy: 测试集准确率（百分比）
    """
    model.eval()  # 设置模型为评估模式
    test_loss = 0
    correct = 0
    total = 0

    # 初始化测试进度条
    loop = tqdm(
        test_loader,
        desc="Testing",
        dynamic_ncols=True,
        leave=True,
        file=sys.stdout,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
        colour='blue'
    )

    with torch.no_grad():  # 禁用梯度计算以节省内存
        for inputs, labels in loop:
            # 数据移至指定设备
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 累计测试指标
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条，使用当前批次大小计算损失
            current_batch_size = inputs.size(0)
            loop.set_postfix(
                loss=f"{test_loss / (total / current_batch_size):.3f}",
                acc=f"{100. * correct / total:.1f}%"
            )
            # 强制刷新进度条
            loop.refresh()
            sys.stdout.flush()

    # 计算平均损失和准确率
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total

    # 打印测试总结
    print(
        f"Test Loss: {test_loss:.4f}, "
        f"Accuracy: {accuracy:.2f}%"
    )

