import numpy as np
import pyautogui
import cv2
from PIL import Image, ImageDraw, ImageFont
from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR
import mss
import win32gui
import win32con
import win32ui
from ctypes import windll
from PIL import Image
import numpy as np

class RapidOcr:
    def __init__(self) -> None:
        self.engine = RapidOCR(params={
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.model_type": ModelType.MOBILE,
        })
        self.boxColor = (0,255,0)
        self.txtColor = (0,0,255)
        self.txtSize = 20
        self.fontStyle = ImageFont.truetype("./font/微软黑体.ttf",self.txtSize,encoding="utf-8")

    def doOcr(self,img):
        return self.engine(img)

    def addChineseText(self,img,text,position):
        if(isinstance(img,np.ndarray)):
            img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        draw.text(position,text,self.txtColor,font=self.fontStyle)
        return cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)

    def displayImage(self,img):
        res = self.doOcr(img)
        displayRes = np.zeros(img.shape,dtype=np.uint8)
        if len(res.elapse_list) == 0:
            return displayRes
        boxes = res.boxes
        txts = res.txts
        for box,txt in zip(boxes,txts):
            x1 = int(box[0][0])
            y1 = int(box[0][1])
            x2 = int(box[2][0])
            y2 = int(box[2][1])
            cv2.rectangle(displayRes,(x1,y1),(x2,y2),self.boxColor,2)
            displayRes = self.addChineseText(displayRes,txt,(x1,y1-35))
        return displayRes


class AnchorModel:
    def __init__(self) -> None:
        self.screenWidth, self.screenHeight = pyautogui.size()
        self.lowH = 0.05
        self.lowW = 0.65
        self.highH = 0.55
        self.highW = 0.9
        self.sct = mss.mss()
        
        # 计算monitor区域
        monitor_width = int(self.screenWidth * (self.highW - self.lowW))
        monitor_height = int(self.screenHeight * (self.highH - self.lowH))
        self.monitor = {
            "top": int(self.screenHeight * self.lowH),
            "left": int(self.screenWidth * self.lowW),
            "width": monitor_width,
            "height": monitor_height
        }
        self.window_name = "anchor"
        # 窗口相关变量
        self.hwnd = None  # 显示窗口句柄
        self.hdc = None
        self.hdc_obj = None  # 新增：PyCDC对象（之前可能遗漏）
        self.hdc_mem = None
        self.bmp = None

    def getAnchor(self, image):
        h, w, c = image.shape
        lowH = int(h * self.lowH)
        lowW = int(w * self.lowW)
        highH = int(h * self.highH)
        highW = int(w * self.highW)
        return image[lowH:highH, lowW:highW], (lowH, lowW, highH, highW)

    def getScreenShot(self, use_anchor=True):
        # 截图前隐藏自身显示窗口（如果存在）
        was_visible = False
        if self.hwnd:
            # 检查窗口是否可见
            if win32gui.IsWindowVisible(self.hwnd):
                was_visible = True
                # 隐藏窗口
                win32gui.ShowWindow(self.hwnd, win32con.SW_HIDE)
        
        try:
            # 执行截图
            monitor = self.sct.monitors[1] if not use_anchor else self.monitor
            screenshot = self.sct.grab(monitor)
            img = np.array(screenshot)
            if img.shape[2] == 4:
                img = img[:, :, :3]
            return img
        finally:
            # 截图后恢复窗口显示（如果之前是可见的）
            if self.hwnd and was_visible:
                win32gui.ShowWindow(self.hwnd, win32con.SW_SHOW)
                # 确保窗口置顶
                win32gui.SetWindowPos(self.hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                     win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

    def getScreenShotWithOutVis(self, use_anchor = True):
        # 执行截图
        monitor = self.sct.monitors[1] if not use_anchor else self.monitor
        screenshot = self.sct.grab(monitor)
        img = np.array(screenshot)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        return img


    def disPlayImageByLocation(self, image, locX, locY):
        """高效显示图像，支持随机位置/指定位置展示，黑色像素区域透明"""
        # 从图像获取目标尺寸（不再依赖self.monitor，更灵活）
        height, width = image.shape[:2]
        target_width, target_height = width, height  # 窗口尺寸与图像一致
        
        # 确保输入是numpy数组
        if not isinstance(image, np.ndarray):
            try:
                image = np.asarray(image)
            except:
                raise ValueError("输入类型不支持，请提供numpy数组或可转换为数组的图像格式")
        
        # 确保图像是BGR格式（OpenCV默认）
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                image = image[:, :, :3]  # 提取BGR通道
            elif image.shape[2] != 3:
                raise ValueError("输入图像通道数必须为3（BGR）或4（BGRA）")
        else:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # 灰度转BGR


        # 首次创建窗口
        if not self.hwnd:
            hwin = win32gui.GetDesktopWindow()
            
            # 创建顶层窗口（支持透明和移动）
            self.hwnd = win32gui.CreateWindow(
                "static", "Transparent Display",
                win32con.WS_POPUP | win32con.WS_VISIBLE,  # 弹出窗口+可见
                locX, locY, target_width, target_height,  # 初始位置和尺寸
                hwin, None, None, None
            )
            
            # 设置窗口样式：顶层+分层（支持透明）
            ex_style = win32gui.GetWindowLong(self.hwnd, win32con.GWL_EXSTYLE)
            win32gui.SetWindowLong(self.hwnd, win32con.GWL_EXSTYLE,
                                ex_style | win32con.WS_EX_TOPMOST | win32con.WS_EX_LAYERED)
            
            # 设置黑色为透明色（LWA_COLORKEY：指定颜色透明）
            win32gui.SetLayeredWindowAttributes(self.hwnd, 0x000000, 0, win32con.LWA_COLORKEY)
            
            # 初始化设备上下文（DC）和位图
            self.hdc = win32gui.GetDC(self.hwnd)
            self.hdc_obj = win32ui.CreateDCFromHandle(self.hdc)
            self.hdc_mem = self.hdc_obj.CreateCompatibleDC()  # 内存DC（双缓冲）
            
            # 创建与图像尺寸匹配的位图
            self.bmp = win32ui.CreateBitmap()
            self.bmp.CreateCompatibleBitmap(self.hdc_obj, target_width, target_height)
            self.old_bmp = self.hdc_mem.SelectObject(self.bmp)  # 保存旧位图，避免资源泄漏

        # 窗口已存在：移动到新位置
        else:
            win32gui.SetWindowPos(
                self.hwnd,
                win32con.HWND_TOPMOST,  # 保持顶层
                locX, locY,  # 新位置
                target_width, target_height,  # 尺寸（与图像一致）
                win32con.SWP_SHOWWINDOW  # 显示窗口
            )

        # 处理图像数据（确保格式兼容Windows位图）
        # 1. 计算位图行对齐字节数（Windows要求4字节对齐）
        row_size = (target_width * 3 + 3) & ~3  # 3通道（BGR），每行字节数向上取4的倍数
        total_bytes = row_size * target_height  # 位图总字节数

        # 2. 转换图像数据为位图兼容格式（补充对齐字节）
        img_bgr = image.copy()
        img_data = img_bgr.tobytes()
        # 补充空白字节以满足行对齐要求
        if len(img_data) < total_bytes:
            img_data += b'\x00' * (total_bytes - len(img_data))

        # 3. 安全更新位图数据
        success = windll.gdi32.SetBitmapBits(
            self.bmp.GetHandle(),  # 位图句柄
            total_bytes,  # 位图所需总字节数（含对齐）
            img_data  # 处理后的图像数据
        )
        if not success:
            raise RuntimeError("更新位图数据失败，可能是格式不兼容")

        # 4. 双缓冲绘制（减少闪烁）
        self.hdc_obj.BitBlt(
            (0, 0),  # 窗口绘制起点
            (target_width, target_height),  # 绘制尺寸
            self.hdc_mem,  # 源内存DC
            (0, 0),  # 内存DC起点
            win32con.SRCCOPY  # 复制模式
        )

        # 5. 刷新窗口显示
        win32gui.UpdateWindow(self.hwnd)
        return self.hwnd

    
    def disPlayImage(self, image):
        """高效显示图像，黑色像素区域透明显示屏幕内容"""
        target_width = self.monitor["width"]
        target_height = self.monitor["height"]
        
        # 确保输入是numpy数组
        if not isinstance(image, np.ndarray):
            try:
                image = np.asarray(image)
            except:
                raise ValueError("输入类型不支持，请提供numpy数组或可转换为数组的图像格式")
        
        # 检查图像尺寸
        height, width = image.shape[:2]
        if width != target_width or height != target_height:
            raise ValueError(f"输入图像尺寸({width}x{height})与目标区域({target_width}x{target_height})不匹配")
        
        # 确保图像是BGR格式
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                image = image[:, :, :3]
            elif image.shape[2] != 3:
                raise ValueError("输入图像通道数必须为3（BGR）或4（BGRA）")
        else:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # 首次创建窗口
        if not self.hwnd:
            hwin = win32gui.GetDesktopWindow()
            left = self.monitor["left"]
            top = self.monitor["top"]
            
            # 创建顶层窗口
            self.hwnd = win32gui.CreateWindow(
                "static", "",
                win32con.WS_POPUP | win32con.WS_VISIBLE | win32con.WS_EX_TOPMOST,
                left, top, target_width, target_height,
                hwin, None, None, None
            )
            
            # 设置窗口透明度和透明色
            win32gui.SetWindowLong(self.hwnd, win32con.GWL_EXSTYLE,
                                  win32gui.GetWindowLong(self.hwnd, win32con.GWL_EXSTYLE) | win32con.WS_EX_LAYERED)
            win32gui.SetLayeredWindowAttributes(self.hwnd, 0, 255, win32con.LWA_COLORKEY)  # 黑色透明
            
            # 设备上下文处理
            self.hdc = win32gui.GetDC(self.hwnd)
            self.hdc_obj = win32ui.CreateDCFromHandle(self.hdc)
            self.hdc_mem = self.hdc_obj.CreateCompatibleDC()
            
            # 创建位图
            self.bmp = win32ui.CreateBitmap()
            self.bmp.CreateCompatibleBitmap(self.hdc_obj, target_width, target_height)
            self.hdc_mem.SelectObject(self.bmp)

        # 处理图像为RGBA格式
        mask = np.all(image == [0, 0, 0], axis=-1)
        img_rgba = np.zeros((target_height, target_width, 4), dtype=np.uint8)
        img_rgba[:, :, :3] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_rgba[:, :, 3] = np.where(mask, 0, 255)
        img_bits = img_rgba.tobytes()

        # 更新位图并显示
        windll.gdi32.SetBitmapBits(self.bmp.GetHandle(), len(img_bits), img_bits)
        self.hdc_obj.BitBlt((0, 0), (target_width, target_height), self.hdc_mem, (0, 0), win32con.SRCCOPY)
        win32gui.UpdateWindow(self.hwnd)

        return self.hwnd

    def cleanup(self):
        """释放资源"""
        if self.bmp:
            win32gui.DeleteObject(self.bmp.GetHandle())
        if self.hdc_mem:
            self.hdc_mem.DeleteDC()
        if self.hdc_obj:
            self.hdc_obj.DeleteDC()
        if self.hdc and self.hwnd:
            win32gui.ReleaseDC(self.hwnd, self.hdc)
        if self.hwnd:
            win32gui.DestroyWindow(self.hwnd)
        # 重置变量
        self.hwnd = None
        self.hdc = None
        self.hdc_obj = None
        self.hdc_mem = None
        self.bmp = None


