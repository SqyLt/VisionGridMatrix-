# image_processor_project/config.py
import cv2
import numpy as np
from typing import Dict, List, Any

class OperationConfig:
    def __init__(self):
        self.color_ranges_bgr: Dict[str, Dict[str, List[int]]] = {
            'r': {'lower': [0, 0, 150], 'upper': [175, 175, 255]},
            'y': {'lower': [0, 200, 185], 'upper': [185, 255, 255]},
            'g': {'lower': [0, 100, 0], 'upper': [200, 255, 195]},
            'b': {'lower': [120, 0, 0], 'upper': [255, 185, 179]},
            'w': {'lower': [200, 200, 200], 'upper': [255, 255, 255]}
        }
        self.color_standardization: Dict[str, str] = {
            'r': 'r', 'y': 'y', 'g': 'g', 'b': 'b', 'w': 'w', '?': '?'
        }
        self.grid_size: int = 4

        self.preprocess_strategies: Dict[str, Dict[str, Any]] = {
            "clear": {
                "hsv_black_lower": np.array([0, 0, 0]), "hsv_black_upper": np.array([180, 255, 80]),
                "morph_kernel_size": (5, 5), "morph_open_iterations": 1,
            },
            "noisy": {
                "median_blur_ksize": 5, "bilateral_d": 9, "bilateral_sigma_color": 75, "bilateral_sigma_space": 75,
                "clahe_clip": 3.0, "clahe_tile": (16, 16), "v_alpha": 1.5, "v_beta": 0,
                "adaptive_block": 21, "adaptive_c": 5,
                "hsv_black_lower_noise": np.array([0, 0, 0]), "hsv_black_upper_noise": np.array([180, 255, 80]),
                "morph_kernel_noise": (9, 9), "morph_close_iter": 3, "morph_open_iter_noise": 2,
                "gauss_ksize_mask": (5, 5), "mask_final_thresh": 127,
            }
        }
        self.locator_finding_strategies: Dict[str, Dict[str, Any]] = {
            "clear": {
                "area_min": 45.0, "area_max_factor": 1 / 20.0, "circularity_min": 0.15,
            },
            "noisy": {
                "area_min_factor": 1 / 1000.0, "area_max_factor": 1 / 10.0, "circularity_min": 0.45,
                "approx_eps_factor": 0.01, "approx_min_verts": 8,
                "area_ratio_min": 0.6, "area_ratio_max": 1.1,
            }
        }
        self.warp_params: Dict[str, Any] = {"adaptive_crop_block": 21, "adaptive_crop_c": 5}

        self.text_drawing_params: Dict[str, Any] = {
            "font_face": cv2.FONT_HERSHEY_SIMPLEX,
            "font_scale": 0.6,
            "font_color": (255, 255, 255),
            "outline_color": (0, 0, 0),
            "thickness": 1,
            "outline_thickness_diff": 1,
            "line_type": cv2.LINE_AA,
        }

DEFAULT_OPERATION_CONFIG = OperationConfig()