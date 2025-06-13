import os
import yaml
import torch

def load_config(config_path='config/config.yaml'):
    """
    加载YAML配置文件
    参数:
        config_path: 配置文件路径
    返回:
        解析后的配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def get_device(config):
    """
    根据配置获取设备
    参数:
        config: 配置字典
    返回:
        设备对象
    """
    use_cuda = config['device']['use_cuda'] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    return device
