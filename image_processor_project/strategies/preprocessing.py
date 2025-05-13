# image_processor_project/strategies/preprocessing.py
import cv2
import numpy as np
from .base import PreprocessingStrategy # 从同包的 base.py 导入

class ClearImagePreprocessingStrategy(PreprocessingStrategy):
    def generate_mask(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.params["hsv_black_lower"], self.params["hsv_black_upper"])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.params["morph_kernel_size"])
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.params["morph_open_iterations"])

class NoisyImagePreprocessingStrategy(PreprocessingStrategy):
    def generate_mask(self, image: np.ndarray) -> np.ndarray:
        p = self.params
        img = image.copy()
        img = cv2.medianBlur(img, p["median_blur_ksize"])
        img = cv2.bilateralFilter(img, p["bilateral_d"], p["bilateral_sigma_color"], p["bilateral_sigma_space"])
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=p["clahe_clip"], tileGridSize=p["clahe_tile"])
        cl = clahe.apply(l)
        img_clahe = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        hsv_e = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2HSV)
        _, _, v_e = cv2.split(hsv_e)
        v_s = cv2.convertScaleAbs(v_e, alpha=p["v_alpha"], beta=p["v_beta"])
        adapt_th = cv2.adaptiveThreshold(v_s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                         p["adaptive_block"], p["adaptive_c"])
        black_m = cv2.inRange(hsv_e, p["hsv_black_lower_noise"], p["hsv_black_upper_noise"])
        mask = cv2.bitwise_and(adapt_th, black_m)
        k_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, p["morph_kernel_noise"])
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_noise, iterations=p["morph_close_iter"])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_noise, iterations=p["morph_open_iter_noise"])
        mask = cv2.GaussianBlur(mask, p["gauss_ksize_mask"], 0)
        _, mask = cv2.threshold(mask, p["mask_final_thresh"], 255, cv2.THRESH_BINARY)
        return mask