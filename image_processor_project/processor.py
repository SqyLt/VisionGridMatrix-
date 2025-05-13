# image_processor_project/processor.py
import traceback
from typing import List

# 绝对路径导入
from config import OperationConfig
from data_models import ImageDataCarrier, CommandExecutionError
from commands.base import Command
from commands.implementations import (
    LoadImageCommand, PreprocessMaskCommand, FindLocatorsCommand,
    SortLocatorsCommand, WarpAndCropCommand, ExtractMatrixCommand,
    SaveArtifactsCommand
)

class CommandProcessor:
    def __init__(self, config: OperationConfig):
        self.config = config
        self._commands: List[Command] = []
        self._build_default_command_sequence()

    def _build_default_command_sequence(self):
        self.add_command(LoadImageCommand(self.config))
        self.add_command(PreprocessMaskCommand(self.config))
        self.add_command(FindLocatorsCommand(self.config))
        self.add_command(SortLocatorsCommand(self.config))
        self.add_command(WarpAndCropCommand(self.config))
        self.add_command(ExtractMatrixCommand(self.config))
        self.add_command(SaveArtifactsCommand(self.config))

    def add_command(self, command: Command):
        self._commands.append(command)

    def process(self, data_carrier: ImageDataCarrier) -> str:
        status = "success"
        try:
            for command in self._commands:
                command.execute(data_carrier)
        except CommandExecutionError as cee:
            status = f"命令执行失败: {str(cee)}"
        except Exception as e:
            status = f"意外的处理错误: {str(e)}"
            print(f"处理 {data_carrier.image_path} 时发生意外错误: {traceback.format_exc()}")

        if status == "success" and data_carrier.warnings:
            return "success_with_warnings"
        return status