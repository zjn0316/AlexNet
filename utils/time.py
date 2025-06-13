import time
import functools
from datetime import datetime


def timer_decorator(func):
    """
    装饰器：测量函数执行时间
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"函数 {func.__name__} 执行时间: {elapsed_time:.4f} 秒")
        return result

    return wrapper


def get_current_time():
    """
    获取当前时间的字符串表示
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_time(seconds):
    """
    将秒数格式化为更易读的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f} 秒"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:.0f} 分 {seconds:.2f} 秒"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{hours:.0f} 时 {minutes:.0f} 分 {seconds:.2f} 秒"


class Timer:
    """
    计时器类：用于测量代码块执行时间
    """

    def __init__(self, name="代码块"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def start(self):
        """开始计时"""
        self.start_time = time.time()
        print(f"{self.name} 开始时间: {get_current_time()}")
        return self

    def stop(self):
        """停止计时并返回经过的时间"""
        if self.start_time is None:
            raise ValueError("计时器尚未开始")

        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        print(f"{self.name} 结束时间: {get_current_time()}")
        print(f"{self.name} 执行时间: {format_time(elapsed_time)}")
        return elapsed_time

    def __enter__(self):
        """上下文管理器：进入时开始计时"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器：退出时停止计时"""
        self.stop()
