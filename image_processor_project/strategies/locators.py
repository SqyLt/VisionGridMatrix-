# image_processor_project/strategies/locators.py
import cv2
import numpy as np
from typing import List, Tuple
from .base import LocatorFindingStrategy # 从同包的 base.py 导入

class ClearImageLocatorFindingStrategy(LocatorFindingStrategy):
    def find(self, mask: np.ndarray, image_area: float) -> List[Tuple[int, int, int]]:
        p = self.params
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        locators = []
        min_a, max_a = p["area_min"], image_area * p["area_max_factor"]
        for c in cnts:
            perim = cv2.arcLength(c, True)
            if perim == 0: continue
            area = cv2.contourArea(c)
            if area == 0: continue
            circ = 4 * np.pi * area / (perim ** 2)
            if min_a < area < max_a and circ > p["circularity_min"]:
                (x, y), r = cv2.minEnclosingCircle(c)
                locators.append((int(x), int(y), int(r)))
        return locators

class NoisyImageLocatorFindingStrategy(LocatorFindingStrategy):
    def find(self, mask: np.ndarray, image_area: float) -> List[Tuple[int, int, int]]:
        p = self.params
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        locators = []
        min_a, max_a = image_area * p["area_min_factor"], image_area * p["area_max_factor"]
        for c_ in cnts:
            perim = cv2.arcLength(c_, True)
            if perim == 0: continue
            area = cv2.contourArea(c_)
            if area == 0: continue
            circ = 4 * np.pi * area / (perim ** 2)
            approx = cv2.approxPolyDP(c_, p["approx_eps_factor"] * perim, True)
            if (min_a < area < max_a and circ > p["circularity_min"] and len(approx) >= p["approx_min_verts"]):
                (x, y), r = cv2.minEnclosingCircle(approx)
                actual_circle_a = np.pi * r ** 2
                if actual_circle_a > 0 and p["area_ratio_min"] < area / actual_circle_a < p["area_ratio_max"]:
                    locators.append((int(x), int(y), int(r)))
        return locators