# image_processor_project/commands/implementations.py
import cv2
import numpy as np
import os
# from typing import List # 如果方法内部需要复杂的类型提示

from .base import Command # 从同包的 base.py 导入
# 绝对路径导入，假设项目根目录在 PYTHONPATH 中
from config import OperationConfig
from data_models import ImageDataCarrier, CommandExecutionError, DataValidationError
from strategies.preprocessing import ClearImagePreprocessingStrategy, NoisyImagePreprocessingStrategy
from strategies.locators import ClearImageLocatorFindingStrategy, NoisyImageLocatorFindingStrategy


class LoadImageCommand(Command):
    def execute(self, data_carrier: ImageDataCarrier) -> None:
        if not os.path.exists(data_carrier.image_path) or not os.access(data_carrier.image_path, os.R_OK):
            raise CommandExecutionError(f"文件访问错误: {data_carrier.image_path}")
        img = cv2.imread(data_carrier.image_path)
        if img is None: raise CommandExecutionError(f"图片解码失败: {data_carrier.image_path}")
        data_carrier.raw_image = img

class PreprocessMaskCommand(Command):
    def execute(self, data_carrier: ImageDataCarrier) -> None:
        if data_carrier.raw_image is None: raise DataValidationError("原始图片未加载。")

        strategy_params_key = "noisy" if data_carrier.is_noisy else "clear"
        strategy_params = self.config.preprocess_strategies[strategy_params_key]

        if data_carrier.is_noisy:
            strategy = NoisyImagePreprocessingStrategy(strategy_params)
        else:
            strategy = ClearImagePreprocessingStrategy(strategy_params)

        data_carrier.processed_mask = strategy.generate_mask(data_carrier.raw_image)

class FindLocatorsCommand(Command):
    def execute(self, data_carrier: ImageDataCarrier) -> None:
        if data_carrier.processed_mask is None or data_carrier.raw_image is None:
            raise DataValidationError("预处理后的掩码或原始图片缺失。")

        image_area = float(data_carrier.raw_image.shape[0] * data_carrier.raw_image.shape[1])
        strategy_params_key = "noisy" if data_carrier.is_noisy else "clear"
        strategy_params = self.config.locator_finding_strategies[strategy_params_key]

        if data_carrier.is_noisy:
            strategy = NoisyImageLocatorFindingStrategy(strategy_params)
        else:
            strategy = ClearImageLocatorFindingStrategy(strategy_params)

        locators = strategy.find(data_carrier.processed_mask, image_area)

        if len(locators) != 4:
            raise CommandExecutionError(f"定位器查找失败: 检测到 {len(locators)} 个定位器 (需要 4 个)。")
        data_carrier.detected_locators = locators

class SortLocatorsCommand(Command):
    def execute(self, data_carrier: ImageDataCarrier) -> None:
        if not data_carrier.detected_locators or len(data_carrier.detected_locators) != 4:
            raise DataValidationError("定位器数量不正确或未找到定位器，无法排序。")
        pts = np.array([(c[0], c[1]) for c in data_carrier.detected_locators], dtype=np.float32)
        sy = pts[np.argsort(pts[:, 1])]
        t2, b2 = sy[:2], sy[2:]
        tl, tr = t2[np.argmin(t2[:, 0])], t2[np.argmax(t2[:, 0])]
        bl, br = b2[np.argmin(b2[:, 0])], b2[np.argmax(b2[:, 0])]
        data_carrier.ordered_locator_points = np.array([tl, tr, br, bl], dtype=np.float32)

class WarpAndCropCommand(Command):
    def execute(self, data_carrier: ImageDataCarrier) -> None:
        if data_carrier.raw_image is None or data_carrier.ordered_locator_points is None:
            raise DataValidationError("原始图片或有序定位点缺失。")

        o_img, s_pts = data_carrier.raw_image, data_carrier.ordered_locator_points
        p_warp = self.config.warp_params

        sides = [np.linalg.norm(s_pts[i] - s_pts[(i + 1) % 4]) for i in range(4)]
        tsl = int(round(np.mean(sides)))
        if tsl <= 0: raise CommandExecutionError("无效的目标变换边长。")

        dst = np.array([[0, 0], [tsl - 1, 0], [tsl - 1, tsl - 1], [0, tsl - 1]], dtype=np.float32)
        tfm = cv2.getPerspectiveTransform(s_pts, dst)
        warped = cv2.warpPerspective(o_img, tfm, (tsl, tsl))

        try:
            gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            th_w = cv2.adaptiveThreshold(gray_w, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                         p_warp["adaptive_crop_block"], p_warp["adaptive_crop_c"])
            cnts_w, _ = cv2.findContours(th_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts_w:
                mc = max(cnts_w, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(mc)
                cw = warped[y:y + h, x:x + w]
                if cw.size > 0:
                    warped = cw
                else:
                    data_carrier.add_warning("智能裁剪结果为空。")
            else:
                data_carrier.add_warning("智能裁剪未找到轮廓。")
        except Exception as e:
            data_carrier.add_warning(f"智能裁剪错误: {str(e)}")
        data_carrier.final_warped_image = warped

class ExtractMatrixCommand(Command):
    def execute(self, data_carrier: ImageDataCarrier) -> None:
        if data_carrier.final_warped_image is None: raise DataValidationError("变换后的图片缺失。")
        w_img, cfg = data_carrier.final_warped_image, self.config
        gs = cfg.grid_size
        ch, cw = w_img.shape[0] // gs, w_img.shape[1] // gs
        if ch == 0 or cw == 0: raise CommandExecutionError("变换后的图片太小，无法创建网格单元。")

        matrix = []
        for r_ in range(gs):
            row = []
            for c_ in range(gs):
                roi_center_fraction = 0.5
                padding_h = int(ch * (1 - roi_center_fraction) / 2)
                padding_w = int(cw * (1 - roi_center_fraction) / 2)
                y1, y2 = r_ * ch + padding_h, (r_ + 1) * ch - padding_h
                x1, x2 = c_ * cw + padding_w, (c_ + 1) * cw - padding_w
                y1, y2 = max(0, y1), min(w_img.shape[0], y2)
                x1, x2 = max(0, x1), min(w_img.shape[1], x2)

                if not (y1 < y2 and x1 < x2):
                    roi = w_img[r_ * ch:(r_ + 1) * ch, c_ * cw:(c_ + 1) * cw]
                else:
                    roi = w_img[y1:y2, x1:x2]

                if roi.size == 0:
                    row.append(cfg.color_standardization['?'])
                    continue

                avg_bgr = np.mean(roi, axis=(0, 1)).astype(int)
                d_char = cfg.color_standardization['?']
                if all(avg_bgr[k] >= cfg.color_ranges_bgr['w']['lower'][k] and \
                       avg_bgr[k] <= cfg.color_ranges_bgr['w']['upper'][k] for k in range(3)):
                    d_char = cfg.color_standardization['w']
                else:
                    color_check_order = ['g', 'y', 'r', 'b']
                    for color_key in color_check_order:
                        if color_key == 'w': continue
                        if color_key in cfg.color_ranges_bgr:
                            color_range = cfg.color_ranges_bgr[color_key]
                            if all(color_range['lower'][k] <= avg_bgr[k] <= color_range['upper'][k] for k in range(3)):
                                d_char = cfg.color_standardization[color_key]
                                break
                row.append(d_char)
            matrix.append(row)
        data_carrier.extracted_matrix = matrix

class SaveArtifactsCommand(Command):
    def execute(self, data_carrier: ImageDataCarrier) -> None:
        image_to_save = None
        if data_carrier.final_warped_image is not None:
            image_to_save = data_carrier.final_warped_image.copy()
            if data_carrier.extracted_matrix and self.config.grid_size > 0:
                txt_p = self.config.text_drawing_params
                gs = self.config.grid_size
                img_h, img_w = image_to_save.shape[:2]
                cell_h = img_h // gs
                cell_w = img_w // gs

                if cell_h > 0 and cell_w > 0:
                    if not (len(data_carrier.extracted_matrix) == gs and \
                            all(len(row) == gs for row in data_carrier.extracted_matrix)):
                        data_carrier.add_warning(
                            f"矩阵维度不匹配 (期望 {gs}x{gs}, "
                            f"得到 {len(data_carrier.extracted_matrix)}x"
                            f"{len(data_carrier.extracted_matrix[0]) if data_carrier.extracted_matrix and data_carrier.extracted_matrix[0] else 0}). "
                            "无法准确在单元格上绘制矩阵。"
                        )
                    else:
                        for r_idx in range(gs):
                            for c_idx in range(gs):
                                char_to_draw = data_carrier.extracted_matrix[r_idx][c_idx]
                                if not char_to_draw: continue
                                cell_center_x = c_idx * cell_w + cell_w // 2
                                cell_center_y = r_idx * cell_h + cell_h // 2
                                (text_w, text_h), baseline = cv2.getTextSize(
                                    char_to_draw, txt_p["font_face"], txt_p["font_scale"], txt_p["thickness"])
                                org_x = cell_center_x - text_w // 2
                                org_y = cell_center_y + text_h // 2
                                cv2.putText(image_to_save, char_to_draw, (org_x, org_y),
                                            txt_p["font_face"], txt_p["font_scale"], txt_p["outline_color"],
                                            txt_p["thickness"] + txt_p["outline_thickness_diff"], txt_p["line_type"])
                                cv2.putText(image_to_save, char_to_draw, (org_x, org_y),
                                            txt_p["font_face"], txt_p["font_scale"], txt_p["font_color"],
                                            txt_p["thickness"], txt_p["line_type"])
                else:
                    data_carrier.add_warning(
                        f"变换后的图片单元格太小 (高:{cell_h}, 宽:{cell_w}) "
                        "无法绘制矩阵。变换后的图片可能太小或网格尺寸太大。"
                    )

            base_name_no_ext = os.path.splitext(data_carrier.filename_base)[0]
            f_name = f"{base_name_no_ext}_result.png"
            f_path = os.path.join(data_carrier.output_dir, f_name)
            try:
                if cv2.imwrite(f_path, image_to_save):
                    data_carrier.add_artifact_path('processed_image_with_matrix', f_path)
                else:
                    data_carrier.add_warning(f"cv2.imwrite 未能保存 {f_name}")
            except Exception as e:
                data_carrier.add_warning(f"保存 {f_name} 失败: {str(e)}")
        else:
            data_carrier.add_warning("没有最终变换后的图片可供保存。")