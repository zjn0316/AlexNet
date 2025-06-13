
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_data_loaders(batch_size=64):
    """
    获取Fashion-MNIST数据集的数据加载器

    参数:
        batch_size: 批次大小
    返回:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # AlexNet需要224x224的输入
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载训练数据集
    train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # 加载测试数据集
    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, test_loader
