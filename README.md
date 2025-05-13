# VisionGridMatrix / 视觉网格矩阵  
  
## 1、Description(描述):
  
### EN:  
VisionGridMatrix is a Python-based image processing tool designed to automatically identify four fiducial markers (locators) in an image, perform perspective warping to rectify the region of interest, and then extract a grid-based color matrix. It processes each cell within the defined grid to determine its dominant color, representing the entire grid as a character matrix (e.g., 'r' for red, 'g' for green). The system includes strategies for handling both clear and noisy images and provides visual feedback by saving the warped image with the identified color characters overlaid on each grid cell.  
  
### ZH:  
VisionGridMatrix (视觉网格矩阵) 是一个基于 Python 的图像处理工具，旨在自动识别图像中的四个基准标记（定位器），执行透视变换以校正目标区域，然后提取基于网格的颜色矩阵。它处理已定义网格中的每个单元格以确定其主导颜色，并将整个网格表示为字符矩阵（例如，'r' 代表红色，'g' 代表绿色）。该系统包含处理清晰图像和含噪声图像的策略，并通过保存在每个网格单元上叠加已识别颜色字符的变换后图像来提供视觉反馈。  
  
## 2、Key Features(主要特性):  
  
### EN:  
* Fiducial Marker Detection: Identifies four circular locators to define the region of interest.  
* Perspective Warping & Cropping: Rectifies the image based on the detected locators to get a top-down view of the grid.  
* Grid-Based Color Analysis: Divides the warped region into a configurable grid (e.g., 4x4) and identifies the dominant BGR color in each cell.  
* Color-to-Character Mapping: Converts identified colors into a standardized character representation (e.g., 'r', 'g', 'b', 'y', 'w', '?').  
* Noise Handling: Implements distinct processing strategies for clear and noisy images.  
* Configurable Parameters: Color ranges, grid size, and processing strategy parameters can be easily adjusted.  
* Output: Generates a character matrix representing the grid colors and saves a processed image with the matrix overlaid for visual verification.  
* Debug Mode: Allows detailed inspection of color detection for specific cells in designated images.  
* Built with OpenCV and Python. 
  
### ZH:  
* 基准标记检测 (Fiducial Marker Detection): 识别四个圆形定位器以定义目标区域。  
* 透视变换与裁剪 (Perspective Warping & Cropping): 基于检测到的定位器校正图像，以获得网格的俯视图。  
* 基于网格的颜色分析 (Grid-Based Color Analysis): 将变换后的区域划分为可配置的网格（例如 4x4），并识别每个单元格中的主导 BGR 颜色。  
* 颜色到字符映射 (Color-to-Character Mapping): 将识别出的颜色转换为标准化的字符表示（例如 'r', 'g', 'b', 'y', 'w', '?'）。  
* 噪声处理 (Noise Handling): 为清晰图像和含噪声图像实现不同的处理策略。  
* 可配置参数 (Configurable Parameters): 颜色范围、网格大小和处理策略参数可以轻松调整。  
* 输出 (Output): 生成表示网格颜色的字符矩阵，并保存带有叠加矩阵的处理后图像以供视觉验证。  
* 调试模式 (Debug Mode): 允许对指定图像中特定单元格的颜色检测进行详细检查。  
* 基于 OpenCV 和 Python 构建。  

## 3、Prerequisites(运行条件):  
  
### EN：  
Python (It is best to use version 3.0 or above.)   
OpenCV (opencv-python)  
NumPy (numpy)  
     
// You can install the required libraries using pip:  
    
    pip install opencv-python numpy
    
   // If it says 'zsh: command not found: python', try the following method:  
    
    pip3 install opencv-python numpy  
      
  （3 is the version number of Python.）  
    
### ZH:  
Python (最好是3.0版本)     
OpenCV (opencv-python)    
NumPy (numpy)  
  
// 您可以使用 pip 安装所需的库：  

    pip install opencv-python numpy
    
// 如果显示 'zsh： command not found： python'，请尝试以下方法：  

    pip3 install opencv-python numpy

（3 是 Python 的版本号。）  
  
## 4.How to Run(如何运行):  
  
### EN：  
1.Prepare Images:  
Create an input folder (by default, ./images in the same directory as the script).  
2.Run the Script:  
Open your terminal or command prompt.  
Navigate to the directory where you saved the script.  
	  
// Execute the script:  
    
    python main.py
  
### ZH:  
1.准备图像：  
创建一个输入文件夹（默认情况下，./images 与脚本位于同一目录中）。  
2.运行脚本：  
打开终端或命令提示符。  
导航到保存脚本的目录。 
  
// 执行脚本：
    
    python main.py
  
## 5.View Results(查看结果):  

### EN:  
After processing, the script will print a summary to the console.  
Processed images will be saved in an output folder (by default, ./results in the same directory as the script).  
The script will automatically create the output folder if it doesn't exist.  

### ZH:  
处理后，脚本会将摘要打印到控制台。  
处理后的图像将保存在输出文件夹中（默认情况下，./results 与脚本位于同一目录中）。  
如果输出文件夹不存在，该脚本将自动创建该文件夹。  
