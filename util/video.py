import cv2
from util.ocr import RapidOcr
from util.anchor_box import AnchorBound
import cv2
import time
from datetime import datetime, timedelta
import logging
import sys
import numpy as np
import os
import glob
import shutil
import json

def saveImg(img,t,isSearch=False):
    hitResult = "./dataset/source/hit/result.txt"
    noHitResult = "./dataset/source/nohit/result.txt"

    if isSearch:
        cv2.imwrite(f"./dataset/hit/image/{t}.jpg", img)
        # 追加写入
        with open(hitResult, "a") as file:
            file.write(f"{t}\n")
    else:
        cv2.imwrite(f"./dataset/nohit/image/{t}.jpg", img)
        # 追加写入
        with open(noHitResult, "a") as file:
            file.write(f"{t}\n")

def readResult():
    hitResult = "./dataset/hit/result.txt"
    noHitResult = "./dataset/nohit/result.txt"
    # 存储是每行一个数字
    max_index = 0
    # 处理hitResult文件
    try:
        with open(hitResult, "r") as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                if line:  # 跳过空行
                    max_index = max(max_index, int(line))
    except FileNotFoundError:
        # 如果文件不存在，默认从0开始
        pass
        
    # 处理noHitResult文件
    try:
        with open(noHitResult, "r") as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                if line:  # 跳过空行
                    max_index = max(max_index, int(line))
    except FileNotFoundError:
        # 如果文件不存在，默认从0开始
        pass
        
    return max_index

def doOcr(img,ocrModel,anchorModel):
    anchorModel.change_rate(0.69, 0.75, 0.10, 0.20)
    img_anchor, (rateW1,rateW2, rateH1,rateH2) = anchorModel.get(img)
    res = ocrModel.get(img_anchor)
    if len(res.elapse_list) == 0:
        return 0

    txts = res.txts
    for txt in txts:
        if txt == "正在搜索物资":
            return 1
    return 0

def videoCheck(video_file):
      # 配置日志：实时输出到控制台，禁用缓冲
    # 移除之前的日志禁用，确保INFO级别日志能输出
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)  # 输出到控制台
        ]
    )
    # 确保stdout无缓冲，实时输出
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

    from util.picodet import PicoDet

    video_file = video_file
    ocrModel = RapidOcr()
    anchorModel = AnchorBound()
    net = PicoDet(
        "./onnx_file/picodet_xs_320_lcnet_postprocessed.onnx",
        "./onnx_file/coco_label.txt",
        prob_threshold=0.5,
        iou_threshold=0.6)

    begin = 1
    
    cap = cv2.VideoCapture(video_file)
    
    if not cap.isOpened():
        logging.error(f"无法打开视频文件: {video_file}")
        return

    # 获取视频总帧数用于计算进度
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    remaining_frames = total_frames - begin
    
    if remaining_frames <= 0:
        logging.info("没有需要处理的帧")
        cap.release()
        return
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, begin)
    
    frame_count = begin
    success = True
    hitCount = 0
    start_time = time.time()
    frame_times = []  # 存储每帧处理时间，用于更准确的估算
    
    logging.info(f"开始从第 {begin} 帧处理视频，总剩余帧数: {remaining_frames}")
    logging.disable(logging.WARNING)
    
    while success:
        frame_start = time.time()
        success, frame = cap.read()
        
        if not success:
            break
        
        # 处理当前帧
        np_boxes = net.doDetect(frame)
        hitCount += 1
        
        # 记录当前帧处理时间
        frame_time = time.time() - frame_start
        frame_times.append(frame_time)
        
        # 每处理100帧更新一次进度和预计时间
        if frame_count % 100 == 0:
            processed_frames = frame_count - begin
            elapsed_time = time.time() - start_time
            avg_frame_time = sum(frame_times) / len(frame_times)
            remaining = remaining_frames - processed_frames
            est_remaining = remaining * avg_frame_time
            
            # 格式化时间显示
            est_finish_time = datetime.now() + timedelta(seconds=est_remaining)
            
            # 使用print确保关键进度信息实时输出（作为备选方案）
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 已处理: {processed_frames}/{remaining_frames} 帧 "
                  f"命中: {hitCount} 次 "
                  f"耗时: {elapsed_time:.1f}s "
                  f"预计剩余: {est_remaining:.1f}s "
                  f"预计完成时间: {est_finish_time.strftime('%H:%M:%S')}", flush=True)
        
        frame_count += 1
    
    # 处理完成
    total_time = time.time() - start_time
    total_processed = frame_count - begin
    logging.info(f"\n视频处理完毕，共处理 {total_processed} 帧，命中 {hitCount} 次")
    logging.info(f"总耗时: {total_time:.1f} 秒，平均每帧耗时: {total_time/total_processed:.3f} 秒")
    
    cap.release()

def videoSolve():
    # 配置日志：实时输出到控制台，禁用缓冲
    # 移除之前的日志禁用，确保INFO级别日志能输出
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)  # 输出到控制台
        ]
    )
    # 确保stdout无缓冲，实时输出
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

    video_file = "./video/train.mp4"
    ocrModel = RapidOcr()
    anchorModel = AnchorBound()
    begin = readResult() + 1
    
    cap = cv2.VideoCapture(video_file)
    
    if not cap.isOpened():
        logging.error(f"无法打开视频文件: {video_file}")
        return

    # 获取视频总帧数用于计算进度
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    remaining_frames = total_frames - begin
    
    if remaining_frames <= 0:
        logging.info("没有需要处理的帧")
        cap.release()
        return
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, begin)
    
    frame_count = begin
    success = True
    hitCount = 0
    start_time = time.time()
    frame_times = []  # 存储每帧处理时间，用于更准确的估算
    
    logging.info(f"开始从第 {begin} 帧处理视频，总剩余帧数: {remaining_frames}")
    logging.disable(logging.WARNING)
    
    while success:
        frame_start = time.time()
        success, frame = cap.read()
        
        if not success:
            break
        
        # 处理当前帧
        result = doOcr(frame, ocrModel, anchorModel)
        hitCount += result
        
        if result == 1:
            saveImg(frame, frame_count, isSearch=True)
        elif frame_count % 10 == 0:
            saveImg(frame, frame_count, isSearch=False)
        
        # 记录当前帧处理时间
        frame_time = time.time() - frame_start
        frame_times.append(frame_time)
        
        # 每处理100帧更新一次进度和预计时间
        if frame_count % 100 == 0:
            processed_frames = frame_count - begin
            elapsed_time = time.time() - start_time
            avg_frame_time = sum(frame_times) / len(frame_times)
            remaining = remaining_frames - processed_frames
            est_remaining = remaining * avg_frame_time
            
            # 格式化时间显示
            est_finish_time = datetime.now() + timedelta(seconds=est_remaining)
            
            # 使用print确保关键进度信息实时输出（作为备选方案）
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 已处理: {processed_frames}/{remaining_frames} 帧 "
                  f"命中: {hitCount} 次 "
                  f"耗时: {elapsed_time:.1f}s "
                  f"预计剩余: {est_remaining:.1f}s "
                  f"预计完成时间: {est_finish_time.strftime('%H:%M:%S')}", flush=True)
        
        frame_count += 1
    
    # 处理完成
    total_time = time.time() - start_time
    total_processed = frame_count - begin
    logging.info(f"\n视频处理完毕，共处理 {total_processed} 帧，命中 {hitCount} 次")
    logging.info(f"总耗时: {total_time:.1f} 秒，平均每帧耗时: {total_time/total_processed:.3f} 秒")
    
    cap.release()

def doFilter(img):
    if img is None:
        return False,[-1,-1,-1,-1]
    from util.anchor_box import AnchorBound
    anchorModel = AnchorBound()
    anchorModel.change_rate(0.65, 0.95, 0.05, 0.6)
    img_anchor,(rateW1,rateW2, rateH1,rateH2) = anchorModel.get(img)

    gray = cv2.cvtColor(img_anchor, cv2.COLOR_BGR2GRAY).astype(np.float32)

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
    max_sum = horizontal_sum[max_index]
    length = len(horizontal_sum)
    # 找到不在最大附近的长度10%sum置为0
    horizontal_sum[max_index-min(length//10,max_index):max_index+min(length//10,length-max_index)] = 0
    second_max_index = np.argmax(horizontal_sum)
    second_max_sum = horizontal_sum[second_max_index]

    if max_index > second_max_index:
        max_index,second_max_index = second_max_index,max_index
        max_sum,second_max_sum = second_max_sum,max_sum

    anchorPoint = [180.,  69.]
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
        if sumNums*2>max_sum:
            centerY = i
            break
  
    
    for i in range(max_index,second_max_index):
        vertical_edges[centerY,i] = 255

    kernel = np.ones((2, 2), np.uint8)  # 3x3正方形核
    dilate_image = cv2.dilate(vertical_edges, kernel, iterations=3)

    from util.anchor_box import detect_large_regions,getAreaLocation
    # 检测区域
    counters = detect_large_regions(dilate_image)

    succ,x1,y1,x2,y2 = getAreaLocation(counters)
    if succ:
        # 先找回原始坐标
        movex = img.shape[0]*rateH1
        movey = img.shape[1]*rateW1
        x1,y1,x2,y2 = int(x1+movex),int(y1+movey),int(x2+movex),int(y2+movey)
        # 在img上绘图
        return True,[y1,x1,y2,x2]
    
    return False,[-1,-1,-1,-1]


def imageFilter(root_path):
     # 读取原始图像（适配目录结构：source/hit/image和source/nohit/image）
    hit_source = glob.glob(os.path.join(root_path, "source", "hit", "image", "*.jpg")) + \
                 glob.glob(os.path.join(root_path, "source", "hit", "image", "*.png"))

    # 如果没有目标目录，创建它 如果有就删除再创建
    if os.path.exists(os.path.join(root_path, "target")):
        shutil.rmtree(os.path.join(root_path, "target"))
    os.makedirs(os.path.join(root_path, "target"), exist_ok=True)
    os.makedirs(os.path.join(root_path, "target", "hit", "image"), exist_ok=True)
    os.makedirs(os.path.join(root_path, "target", "hit", "vis"), exist_ok=True)
    # 创建result.json文件
    result_file = os.path.join(root_path, "target", "hit", "result.json")
    with open(result_file, "w") as f:
        f.write("")

    # 记录一下进度
    total = len(hit_source)
    # 存储所有图像的标注结果（用于生成标准JSON）
    result_data: Dict[str, List[Dict]] = {"images": []}
    processed = 0

    # 遍历hit_source中的每个图像
    for img_path in hit_source:
        # 每处理100张图片打印一次进度
        processed += 1
        if processed % 100 == 0:
            print(f"处理进度: {processed}/{total}")

        # 提取文件名（无扩展名）
        filename = os.path.basename(img_path).split('.')[0]
        
        # 构建目标路径（target/hit/image/filename.jpg）
        target_path = os.path.join(root_path, "target", "hit", "image", f"{filename}.jpg")
        target_vis_path = os.path.join(root_path, "target", "hit", "vis", f"{filename}.jpg")
        
        '''
        json格式
        {
            "images": [
                {
                    "filename": "1692038523345",
                    "box": [x1,y1,x2,y2]
                },
                {
                    "filename": "1692038523345",
                    "box": [x1,y1,x2,y2]
                },
            ]
        }
        '''

        # 复制图像到目标路径 res以json的形式写到result.json json格式 filename res
        img = cv2.imread(img_path)
        suc,box = doFilter(img)
        if not suc:
            continue
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imwrite(target_vis_path, img)
        shutil.copy2(img_path, target_path)
            # 8. 记录标注结果（标准JSON格式）
        result_data["images"].append({
            "filename": filename,  # 与保存的文件名对应
            "box": box,
            "original_path": img_path  # 可选：记录原始路径便于追溯
        })
    
    # 9. 保存最终JSON结果（标准格式）
    try:
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"所有结果已保存至：{result_file}")
    except Exception as e:
        print(f"错误：JSON文件保存失败 - {str(e)}")

