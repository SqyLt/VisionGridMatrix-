# image_processor_project/commands/base.py
from abc import ABC, abstractmethod
# 假设项目根目录在 PYTHONPATH 中，或者运行时从项目根目录的父目录执行
# 这样可以直接从顶层模块导入
from config import OperationConfig
from data_models import ImageDataCarrier

class Command(ABC):
    def __init__(self, config: OperationConfig):
        self.config = config

    @abstractmethod
    def execute(self, data_carrier: ImageDataCarrier) -> None: pass