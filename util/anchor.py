from util.model import RapidOcr,AnchorModel
import numpy as np
import cv2


def detect_large_regions(clustered_img, min_size_ratio=50):
    # 确保输入图像是单通道（取第一个通道用于检测）
    if clustered_img.ndim == 3:
        # 提取第一个通道，并二值化（非0值视为前景）
        channel = clustered_img[..., 0]
        binary = (channel != 0).astype(np.uint8) * 255  # 转为0-255的二值图
    else:
        # 单通道图像直接二值化
        binary = (clustered_img != 0).astype(np.uint8) * 255
    
    # 调用OpenCV连通区域检测函数
    # 参数说明：
    # - 输入二值图（0为背景，非0为前景）
    # - connectivity：8表示8邻域检测，4表示4邻域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=4)
    
    # 计算图像总面积和最小区域阈值
    h, w = binary.shape[:2]
    total_area = h * w
    min_area = total_area // min_size_ratio  # 相当于 size*50 < tolSize 的反向判断
    
    # 筛选出符合条件的区域（跳过标签0，因为0是背景）
    large_regions = []
    for label in range(1, num_labels):
        # stats格式：[x, y, width, height, area]
        x, y, width, height, area = stats[label]

        
        # 过滤小区域
        if height*width < min_area:
            continue
        
        # 转换为 [h1, w1, h2, w2] 格式（y1, x1, y2, x2）
        h1, w1 = y, x
        h2, w2 = y + height - 1, x + width - 1  # 减1是因为包含边界
        
        large_regions.append([h1, w1, h2, w2])
    
    return large_regions


def checkContours(counters):
    succ = False
    if len(counters) == 0 or len(counters) > 10:
        return succ,-1,-1,-1

    counters = remove_contained_rectangles(counters)
    if len(counters) != 2:
        return succ,-1,-1,-1

    firstCounter = counters[0]
    seconCounter = counters[1]  

    if firstCounter[0] >seconCounter[0]:
        firstCounter,seconCounter = seconCounter,firstCounter

    aroundFiveDistance = (firstCounter[3] - firstCounter[1])*1.0/5


    aroundMul =round( (seconCounter[3] - seconCounter[1])*1.0/aroundFiveDistance)

    
    distance = (seconCounter[3] - seconCounter[1])*1.0/aroundMul

    succ = True

    return succ,seconCounter[0],seconCounter[1],distance

def is_contained(rect1, rect2):
    """判断rect1是否被rect2包含"""
    x1, y1, x2, y2 = rect1
    a1, b1, a2, b2 = rect2
    # rect1的所有顶点都在rect2内部
    return (a1 <= x1 and x2 <= a2 and 
            b1 <= y1 and y2 <= b2)

def remove_contained_rectangles(rectangles):
    """移除被其他矩形包含的矩形"""
    # 复制矩形列表，避免修改原列表
    filtered = []
    # 遍历每个矩形
    for i, rect in enumerate(rectangles):
        # 假设当前矩形不被任何其他矩形包含
        contained = False
        # 与其他所有矩形比较
        for j, other_rect in enumerate(rectangles):
            if i == j:
                continue  # 跳过与自身比较
            # 检查是否被其他矩形包含
            if is_contained(rect, other_rect):
                contained = True
                break
        # 如果不被包含，则保留
        if not contained:
            filtered.append(rect)
    return filtered

def createContours(x,y,distance,x_num,y_num):
    contours = []
    for i in range(x_num):
        for j in range(y_num):
            x1 = x + i*distance
            y1 = y + j*distance
            contours.append([x1,y1,x1+distance,y1+distance])
    
    return contours


def draw_rectangles_on_image(img, rectangles):
    
    # 复制原图，避免直接修改原图
    img_with_rects = img.copy()
    
    # 定义绘制长方形的颜色和线宽
    color = (0, 255, 0)  # 绿色 (B, G, R)
    thickness = 2        # 线宽
    
    # 遍历所有长方形坐标并绘制
    for idx, rect in enumerate(rectangles):
        # 解析坐标：h1, w1, h2, w2 (y1, x1, y2, x2)
        h1, w1, h2, w2 = rect
        
        # 确保坐标是整数
        h1, w1, h2, w2 = int(h1), int(w1), int(h2), int(w2)
        
        # 绘制长方形
        # cv2.rectangle(图像, 左上角坐标, 右下角坐标, 颜色, 线宽)
        cv2.rectangle(img_with_rects, (w1, h1), (w2, h2), color, thickness)
        
        # 可选：在长方形上方添加索引标签
        cv2.putText(
            img_with_rects, 
            f"Rect {idx+1}", 
            (w1, h1 - 10),  # 文本位置（左上角上方10像素）
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5,            # 字体大小
            (0, 0, 255),    # 红色文本
            1               # 文本线宽
        )
    
    # 显示结果
    cv2.imshow('Image with Rectangles', img_with_rects)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
依赖
ocr识别的结果正常
Canny边缘检测 检测出最大和第二大的线
区域连通性正常

'''

def getSquareParamC(img,display=False):
    ocrModel = RapidOcr()
    res = ocrModel.doOcr(img)

    if len(res.elapse_list) == 0:
        print("未检测到目标")
        return False,-1,-1,-1

    boxes=res.boxes
    txts=res.txts
    index = -1
    for i in range(len(txts)):
        if txts[i] == '正在搜索物资':
            index = i
            break
    if index == -1:
        print("未检测到目标")
        return False,-1,-1,-1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 垂直算子
    sharp_kernel = np.array([
        [-1, 1],
        [-1, 1],
    ],np.float32)

    convolved_image = cv2.filter2D(gray, -1, sharp_kernel)
    scaled_image = cv2.normalize(convolved_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Canny边缘检测
    vertical_edges = cv2.Canny(scaled_image, 50, 150, apertureSize=3)


    # 水平求和并找到最大的
    horizontal_sum = np.sum(vertical_edges, axis=0)
    max_index = np.argmax(horizontal_sum)
    length = len(horizontal_sum)
    # 找到不在最大附近的长度10%sum置为0
    horizontal_sum[max_index-min(length//10,max_index):max_index+min(length//10,length-max_index)] = 0
    second_max_index = np.argmax(horizontal_sum)

    if max_index > second_max_index:
        max_index,second_max_index = second_max_index,max_index

    anchorPoint =boxes[index][2]
    x,y = int(anchorPoint[0]),int(anchorPoint[1])

    for i in range(x+1,vertical_edges.shape[1]):
        if vertical_edges[y,i] == 255:
            break
        vertical_edges[y,i] = 255
    for i in range(x,max_index,-1):
        vertical_edges[y,i] = 255

    sumNums = 0
    centerY = 0
    centerX = max_index
    for i in range(vertical_edges.shape[0]):
        sumNums += int(vertical_edges[i,centerX])
        if sumNums*2>horizontal_sum[centerX]:
            centerY = i
            break
    
    for i in range(max_index,second_max_index):
        vertical_edges[centerY,i] = 255

    kernel = np.ones((2, 2), np.uint8)  # 3x3正方形核
    dilate_image = cv2.dilate(vertical_edges, kernel, iterations=3)

    # 检测区域
    counters = detect_large_regions(dilate_image)

        # 分块
    succ, resX,resY,resDistance = checkContours(counters)

    if succ ==False:
        return succ,-1,-1,-1

    if display:
        contours = createContours(resX,resY,resDistance,6,6)
        draw_rectangles_on_image(img,contours)
    return succ,resX,resY,resDistance


'''
将图像中的斜线区域检测出来
'''
def preImageSolve(img):
    # 如果不是灰度图
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = img.astype(np.float32)

    sharp_kernel = np.array([[-2, 1],
                             [1, 0]], np.float32)

    convolved_image = cv2.filter2D(gray, -1, sharp_kernel)
    scaled_image = cv2.normalize(convolved_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    adaptive_mean = cv2.adaptiveThreshold(
        scaled_image, 255, 
        cv2.ADAPTIVE_THRESH_MEAN_C,  # 基于邻域均值
        cv2.THRESH_BINARY, 
        7,  # 块大小（11x11）
        2    # 常数
    )
    kernel = np.ones((2, 2), np.uint8)  # 3x3正方形核
    dilate_image = cv2.dilate(adaptive_mean, kernel, iterations=1)
    erode_image = 255- cv2.erode(dilate_image, kernel, iterations=6)
    return erode_image


'''
方格是否占用
'''
def squareDetect(img,resX,resY,resDistance,display=False,hNum=6,wNum=6):
    zero_matrix = np.zeros((hNum, wNum), dtype=int)
    for i in range(hNum):
        for j in range(wNum):
            h1 = int(resX + i*resDistance)
            w1 = int(resY + j*resDistance)
            h2 = int(resX + (i+1)*resDistance)
            w2 = int(resY + (j+1)*resDistance)
            if h1 < 0 or w1 < 0 or h2 > img.shape[0] or w2 > img.shape[1]:
                continue
            zero_matrix[i,j] = np.mean(img[h1:h2,w1:w2])>198
            if display and zero_matrix[i,j]:
                cv2.rectangle(img, (w1, h1), (w2, h2), (0, 255, 0), 2)
                cv2.imshow("img",img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    return zero_matrix

'''
根据zero_matrix 获取图片
'''

def getImgByZeroMatrix(img,zero_matrix,resX,resY,resDistance):
    hNum,wNum = zero_matrix.shape
    x1,y1,x2,y2 = 0,0,0,0
    suc = False
    for i in range(hNum):
        for j in range(wNum):
            if zero_matrix[i,j]==1:
                x1 = min(x1,i)
                y1 = min(y1,j)
                x2 = max(x2,i+1)
                y2 = max(y2,j+1)
                suc = True
    if suc == False:
        return suc,img
    boundImg = img[int(resX + x1*resDistance):int(resX + x2*resDistance),int(resY + y1*resDistance):int(resY + y2*resDistance)]
    return suc,boundImg
