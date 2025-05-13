# image_processor_project/main.py
import os
from typing import Dict, Any

# Assuming image_processor_project is in your PYTHONPATH, or you run from its parent directory
from config import OperationConfig, DEFAULT_OPERATION_CONFIG
from data_models import ImageDataCarrier
from processor import CommandProcessor

def batch_process_with_commands(
        input_folder: str,
        output_base_dir: str,
        config: OperationConfig = DEFAULT_OPERATION_CONFIG
) -> Dict[str, Any]:
    processor = CommandProcessor(config)
    summary = {
        'total_images': 0, 'processed_successfully': 0,
        'processed_with_warnings': 0, 'failed_images': [],
        'all_data_carriers_summary': []
    }

    if not os.path.isdir(input_folder):
        print(f"Warning: Input folder {input_folder} does not exist. No images to process.")
        return summary

    os.makedirs(output_base_dir, exist_ok=True) # Ensure output directory exists

    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')): continue
        if fname == "sample_clear_image_for_grid_text.png": # Adjust as per original logic
            print(f"Skipping specified sample file: {fname}")
            continue

        summary['total_images'] += 1
        image_file_path = os.path.join(input_folder, fname)

        carrier = ImageDataCarrier(image_file_path, output_base_dir)
        final_status = processor.process(carrier)

        carrier_summary = {
            'status': final_status, 'source_image': carrier.image_path,
            'circle_count': len(carrier.detected_locators),
            'warped_image_path': carrier.saved_artifacts.get('processed_image_with_matrix'),
            'matrix': carrier.extracted_matrix, 'warnings': carrier.warnings,
        }
        summary['all_data_carriers_summary'].append(carrier_summary)

        if final_status == 'success':
            summary['processed_successfully'] += 1
        elif final_status == 'success_with_warnings':
            summary['processed_with_warnings'] += 1
        else:
            summary['failed_images'].append({'filename': fname, 'error': final_status})

    return summary


if __name__ == "__main__":
    # Define paths relative to the project root
    # current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # PROJECT_ROOT = current_script_dir # Assuming main.py is at the root of image_processor_project

    # More general way, use relative paths directly (assuming running from image_processor_project directory or its parent)
    INPUT_DIR_CMD = "./images"
    OUTPUT_DIR_CMD = "./results"


    os.makedirs(INPUT_DIR_CMD, exist_ok=True)
    # os.makedirs(OUTPUT_DIR_CMD, exist_ok=True) # Output dir creation is handled in batch_process_with_commands

    # Example: if images folder is empty, a dummy image can be created
    # import cv2
    # import numpy as np
    # if not os.listdir(INPUT_DIR_CMD):
    #     dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    #     cv2.imwrite(os.path.join(INPUT_DIR_CMD, "dummy.png"), dummy_image)
    #     print(f"Created a dummy image in {INPUT_DIR_CMD} as it was empty.")


    cmd_summary = batch_process_with_commands(INPUT_DIR_CMD, OUTPUT_DIR_CMD, config=DEFAULT_OPERATION_CONFIG)
    print(f"Total images: {cmd_summary['total_images']}")
    print(f"Succeeded: {cmd_summary['processed_successfully']}")
    print(f"Succeeded (with warnings): {cmd_summary['processed_with_warnings']}")
    num_failed_cmd = len(cmd_summary['failed_images'])
    print(f"Failed: {num_failed_cmd}")
    print(f"Results are stored at:'{os.path.abspath(OUTPUT_DIR_CMD)}'") # Printing absolute path is clearer

    if num_failed_cmd > 0:
        print("\nFailure Details:")
        for item in cmd_summary['failed_images']:
            print(f"- File: {item['filename']}, Reason: {item['error']}")

    if cmd_summary['total_images'] == 0:
        print(f"\nNo images were found or processed in '{INPUT_DIR_CMD}'.")

    print("--- Command Pattern Processing Complete (Final) ---")