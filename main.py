from ast import mod
import tkinter as tk
import pyautogui
import time

from win32.lib.win32con import PSD_DISABLEORIENTATION
import cv2
import random
import mss
from PIL import Image, ImageTk
from util.model import RapidOcr,AnchorModel
from util.anchor import getSquareParamC,preImageSolve,squareDetect
from util.draw import drawMatrix
import numpy as np
from util.video import videoSolve


class DynamicRectangleDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("动态长方形绘制")
        
        # 设置窗口为全屏且置顶
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        # 设置窗口透明度（0.0-1.0）
        self.root.attributes("-alpha", 0.7)
        
        # 创建画布
        self.canvas = tk.Canvas(root, cursor="none")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 长方形属性
        self.rectangle = None
        self.x1, self.y1 = 100, 100
        self.x2, self.y2 = 300, 300
        self.dx1, self.dy1 = 2, 2
        self.dx2, self.dy2 = 3, 3
        
        # 屏幕捕捉相关
        self.screen_shot = None
        self.screen_image = None
        self.update_screenshot()
        
        # 开始动画
        self.animate()
        
        # 按ESC键退出
        self.root.bind("<Escape>", lambda e: self.root.destroy())

    def update_screenshot(self):
        """更新屏幕截图"""
        try:
            # 捕捉屏幕
            screenshot = pyautogui.screenshot()
            # 转换为Tkinter可用的格式
            self.screen_shot = ImageTk.PhotoImage(image=screenshot)
            # 更新画布上的图像
            if self.screen_image:
                self.canvas.delete(self.screen_image)
            self.screen_image = self.canvas.create_image(0, 0, image=self.screen_shot, anchor=tk.NW)
            # 将截图放在最底层
            self.canvas.tag_lower(self.screen_image)
        except Exception as e:
            print(f"截图错误: {e}")
        
        # 每2秒更新一次截图
        self.root.after(1, self.update_screenshot)

    def animate(self):
        """动画循环，更新长方形位置和大小"""
        # 获取屏幕尺寸
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # 更新长方形坐标
        self.x1 += self.dx1
        self.y1 += self.dy1
        self.x2 += self.dx2
        self.y2 += self.dy2
        
        # 边界检测，碰到边界反弹
        if self.x1 <= 0 or self.x1 >= screen_width:
            self.dx1 = -self.dx1
            # 随机改变速度，使动画更有趣
            self.dx1 *= random.uniform(0.8, 1.2)
            
        if self.y1 <= 0 or self.y1 >= screen_height:
            self.dy1 = -self.dy1
            self.dy1 *= random.uniform(0.8, 1.2)
            
        if self.x2 <= 0 or self.x2 >= screen_width:
            self.dx2 = -self.dx2
            self.dx2 *= random.uniform(0.8, 1.2)
            
        if self.y2 <= 0 or self.y2 >= screen_height:
            self.dy2 = -self.dy2
            self.dy2 *= random.uniform(0.8, 1.2)
        
        # 确保长方形有一定大小
        if self.x2 <= self.x1 + 50:
            self.dx2 = abs(self.dx2)
        if self.y2 <= self.y1 + 50:
            self.dy2 = abs(self.dy2)
        
        # 删除旧的长方形
        if self.rectangle:
            self.canvas.delete(self.rectangle)
        
        # 绘制新的长方形，使用Tkinter支持的颜色格式
        self.rectangle = self.canvas.create_rectangle(
            self.x1, self.y1, self.x2, self.y2,
            outline="red", width=3,
            fill="#ffff00"  # 纯黄色，不带透明度
        )
        
        # 继续动画循环
        self.root.after(30, self.animate)

def get_max_frame_rate_without_io():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        start_time = time.time()
        screenshot = sct.grab(monitor)
        cost_time = time.time() - start_time
    print(f"max frame rate without io: {1 / cost_time}")

def doSquare():
    
    ocrModel = RapidOcr()
    anchorModel = AnchorModel()

    while(True):
        img = anchorModel.getScreenShotWithOutVis()
        succ, resX, resY, resDistance = getSquareParamC(img, display=False)

        zero_matrix = np.zeros((6, 6), dtype=int)
        if not succ:
            print("未找到目标")
        else:
            preImage = preImageSolve(img)
            zero_matrix = squareDetect(preImage, resX, resY, resDistance, display=False)

        # 生成显示图像
        displayImage = drawMatrix(zero_matrix)
        cv2.imshow("displayImage",displayImage)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def Test():
    url = "./pic/4.png"
    ocrModel = RapidOcr()
    anchorModel = AnchorModel()

    img = cv2.imread(url)
    img,(lowH, lowW, highH, highW) = anchorModel.getAnchor(img)


    succ, resX, resY, resDistance = getSquareParamC(img, display=True)

    zero_matrix = np.zeros((6, 6), dtype=int)
    if not succ:
        print("未找到目标")
    else:
        preImage = preImageSolve(img)
        zero_matrix = squareDetect(preImage, resX, resY, resDistance, display=True)

    # 生成显示图像
    displayImage = drawMatrix(zero_matrix)

def test_gpu():

    import paddle
    from util.model import RapidOcrGPU
    print(paddle.utils.run_check())
    print(paddle.device.get_device())

    url_list = ["train_time_210s.jpg"]

    ocrModel = RapidOcrGPU()
    anchorModel = AnchorModel()    

    img_list = []
    for url in url_list:
        img = cv2.imread(url)
        img_list.append(img)


    img_tmp,(lowH, lowW, highH, highW) = anchorModel.getAnchor(img_list[0])
    res = ocrModel.doOcr(img_tmp)



    start_time = time.time()
    count = 0
    while(True):
        index = count%len(img_list)
        img_tmp,(lowH, lowW, highH, highW) = anchorModel.getAnchor(img_list[index])
        res = ocrModel.doOcr(img_tmp)

        count += 1
        if count % 10 == 0:
            end_time = time.time()
            print("{:.2f}ms".format((end_time - start_time)*1000/10))
            start_time = end_time


def test_cpu():
    ocrModel = RapidOcr()
    anchorModel = AnchorModel()
    anchorModel.changeRate(0.69, 0.75, 0.08, 0.20)

    img =cv2.imread("train_time_210s.jpg")

    img_anchor,(lowH, lowW, highH, highW) = anchorModel.getAnchor(img)

    cv2.imshow("img",img_anchor)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 帮我计算100次平均时间
    start_time = time.time()
    for i in range(10):
        res = ocrModel.doOcr(img_anchor)
    end_time = time.time()
    print("10次平均时间: {:.2f}ms".format((end_time - start_time)*1000/10))

 
def doOcrContinue():
    ocrModel = RapidOcr()
    anchorModel = AnchorModel()
    # 获取截图
    while(True):
        start_time = time.time()
        img = anchorModel.getScreenShot()
        ocrRes = ocrModel.displayImage(img)
        ela_time = time.time() - start_time
        ela_time = int(ela_time*1000)

        str = "{}ms".format(ela_time)

        ocrRes = ocrModel.addChineseText(ocrRes,str,(40,35))

        anchorModel.disPlayImage(ocrRes)
        time.sleep(0.1)




if __name__ == "__main__":
    from util.video import videoSolve,imageFilter,doFilter

    from util.multi_video import multiVideoTask

    # videoSolve(use_cuda=True)
    # multiVideoTask()
    imageFilter("./dataset")

    # img = cv2.imread("10562.jpg")
    # suc,box =doFilter(img)
    # if suc:
    #     cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)




