import sys
from tqdm import tqdm


def train(model, train_loader, criterion, optimizer, device, epochs=5,batch_size=64):
    """
    训练AlexNet模型

    参数:
        model: 初始化的AlexNet模型
        train_loader: 训练数据加载器
        criterion: 损失函数（如CrossEntropyLoss）
        optimizer: 优化器（如Adam）
        device: 计算设备（'cuda'或'cpu'）
        epochs: 训练轮数（默认5轮）

    返回:
        train_losses: 每轮训练的平均损失列表
    """
    model.train()  # 设置模型为训练模式
    train_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # 初始化进度条（动态调整宽度，显示训练轮次）
        loop = tqdm(
            train_loader,
            desc=f'Epoch {epoch + 1}/{epochs}',
            dynamic_ncols=True,
            leave=True,
            colour='green'
        )

        for inputs, labels in loop:
            # 数据移至指定设备
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播与参数更新
            loss.backward()
            optimizer.step()

            # 统计训练指标
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 计算当前批次大小
            batch_size = inputs.size(0)

            # 更新进度条（显示损失和准确率，强制刷新）
            loop.set_postfix(
                loss=f"{running_loss / (total / batch_size):.3f}",
                acc=f"{100. * correct / total:.1f}%"
            )
            loop.refresh()  # 强制刷新显示
            sys.stdout.flush()  # 额外刷新输出缓冲区

        # 计算本轮平均损失
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # 打印轮次总结（刷新输出缓冲区）
        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {epoch_loss:.4f}, "
            f"Accuracy: {100. * correct / total:.2f}%",
            flush=True
        )

