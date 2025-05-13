# image_processor_project/data_models.py
import os
import numpy as np
from typing import Dict, List, Tuple, Optional

# --- 自定义异常 ---
class CommandExecutionError(Exception): pass
class DataValidationError(CommandExecutionError): pass

# --- 数据载体 ---
class ImageDataCarrier:
    def __init__(self, image_path: str, output_dir: str):
        self.image_path: str = image_path
        self.output_dir: str = output_dir
        self.filename_base: str = os.path.basename(image_path)
        self.is_noisy: bool = 'noise' in self.filename_base.lower()

        self.raw_image: Optional[np.ndarray] = None
        self.processed_mask: Optional[np.ndarray] = None
        self.detected_locators: List[Tuple[int, int, int]] = []
        self.ordered_locator_points: Optional[np.ndarray] = None
        self.final_warped_image: Optional[np.ndarray] = None
        self.extracted_matrix: List[List[str]] = []

        self.warnings: List[str] = []
        self.saved_artifacts: Dict[str, str] = {}

    def add_warning(self, message: str): self.warnings.append(message)

    def add_artifact_path(self, key: str, path: str): self.saved_artifacts[key] = path