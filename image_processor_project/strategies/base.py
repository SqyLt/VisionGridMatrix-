# image_processor_project/strategies/base.py
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Any

class PreprocessingStrategy(ABC):
    def __init__(self, params: Dict[str, Any]):
        self.params = params

    @abstractmethod
    def generate_mask(self, image: np.ndarray) -> np.ndarray: pass

class LocatorFindingStrategy(ABC):
    def __init__(self, params: Dict[str, Any]):
        self.params = params

    @abstractmethod
    def find(self, mask: np.ndarray, image_area: float) -> List[Tuple[int, int, int]]: pass