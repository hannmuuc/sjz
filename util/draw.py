import numpy as np
import cv2

def drawMatrix(matrix):
    # 图像参数
    pixel_size = 50  # 每个矩阵元素放大后的像素大小
    cols = len(matrix[0])  # 矩阵列数
    rows = len(matrix)     # 矩阵行数
    
    width = cols * pixel_size  # 图像宽度
    height = rows * pixel_size # 图像高度

    # 创建黑色背景图像 (OpenCV默认是BGR格式)
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # 绘制白色像素（对应矩阵中的1）
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 1:
                # 计算当前方块的左上角和右下角坐标
                top_left = (j * pixel_size, i * pixel_size)
                bottom_right = ((j + 1) * pixel_size, (i + 1) * pixel_size)
                
                # 使用cv2.rectangle绘制方块，颜色为白色(255,255,255)，填充整个矩形
                cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), -1)
    
    # 绘制绿色网格线 (BGR格式中绿色是(0,255,0))
    # 绘制垂直线
    for j in range(cols + 1):
        x = j * pixel_size
        # 从顶部到底部绘制垂直线
        cv2.line(image, (x, 0), (x, height), (0, 255, 0), 1)
    
    # 绘制水平线
    for i in range(rows + 1):
        y = i * pixel_size
        # 从左侧到右侧绘制水平线
        cv2.line(image, (0, y), (width, y), (0, 255, 0), 1)
    
    return image