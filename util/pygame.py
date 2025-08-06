import pygame
import numpy as np
import cv2
import os
import random
from pygame import gfxdraw

class PygameImageViewer:
    def __init__(self, window_title="Dynamic Image Viewer", initial_pos=None):
        """初始化图片查看器（适配Pygame 2.6.1）"""
        # 初始化Pygame
        pygame.init()
        pygame.font.init()
        
        self.window_title = window_title
        self.running = True  # 运行状态
        self.current_image = None  # 当前显示图像
        self.screen = None  # Pygame窗口对象
        self.clock = pygame.time.Clock()  # 帧率控制
        self.initial_pos = initial_pos  # 初始位置
        
        # 获取屏幕尺寸（用于位置计算）
        info = pygame.display.Info()
        self.screen_width, self.screen_height = info.current_w, info.current_h

    def convert_numpy_to_pygame(self, numpy_img):
        """将OpenCV格式（BGR）转换为Pygame格式（RGB）"""
        if numpy_img is None or not isinstance(numpy_img, np.ndarray):
            raise ValueError("输入必须是有效的NumPy数组图像")
        
        # 处理通道数
        if len(numpy_img.shape) == 2:
            # 灰度图转BGR
            numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_GRAY2BGR)
        elif numpy_img.shape[2] == 4:
            # RGBA转BGR
            numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGBA2BGR)
        
        # BGR转RGB（Pygame需要RGB格式）
        rgb_img = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2RGB)
        return pygame.surfarray.make_surface(rgb_img)

    def set_position(self, x=None, y=None):
        """设置窗口位置（适配Pygame 2.6.1）"""
        if not self.screen:
            return
            
        window_w, window_h = self.screen.get_size()
        
        # 计算随机位置（确保窗口完全显示在屏幕内）
        if x is None or y is None:
            x = random.randint(0, max(0, self.screen_width - window_w))
            y = random.randint(0, max(0, self.screen_height - window_h))
        
        # Pygame 2.6.1 正确设置窗口位置的方法
        try:
            # 关键修复：确保窗口已创建且使用正确的参数格式
            pygame.display.set_window_position(x, y)
            # 强制刷新窗口状态
            pygame.display.update()
        except Exception as e:
            print(f"设置窗口位置失败: {e}")
            # 备选方案：通过环境变量重新创建窗口（仅首次有效）
            if not hasattr(self, "_position_fallback"):
                self._position_fallback = True
                print("尝试备选方案设置位置...")
                os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"
                # 重新创建窗口以应用位置
                self.screen = pygame.display.set_mode((window_w, window_h))
                pygame.display.set_caption(self.window_title)

    def update_image(self, numpy_img, x=None, y=None):
        """更新显示的图片，支持动态调整位置和尺寸"""
        # 转换图像格式
        try:
            self.current_image = self.convert_numpy_to_pygame(numpy_img)
        except Exception as e:
            print(f"图像转换失败: {e}")
            return
        
        img_w, img_h = self.current_image.get_size()
        # 首次创建窗口或调整窗口尺寸
        if not self.screen or self.screen.get_size() != (img_w, img_h):
            # 首次创建时应用初始位置
            if self.initial_pos and not self.screen:
                init_x, init_y = self.initial_pos
                # 初始位置通过环境变量设置（Pygame推荐方式）
                os.environ['SDL_VIDEO_WINDOW_POS'] = f"{init_x},{init_y}"
                self.screen = pygame.display.set_mode((img_w, img_h))
                self.initial_pos = None  # 仅生效一次
            else:
                # 非首次创建直接调整尺寸
                self.screen = pygame.display.set_mode((img_w, img_h))
            pygame.display.set_caption(self.window_title)
        
        # 更新窗口位置
        self.set_position(x, y)

    def run(self):
        """主循环：处理事件并刷新显示"""
        while self.running:
            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False  # ESC键退出
                    elif event.key == pygame.K_SPACE:
                        self.set_position()  # 空格键随机切换位置
            
            # 绘制图像
            if self.screen and self.current_image:
                self.screen.blit(self.current_image, (0, 0))  # 将图像绘制到窗口
                pygame.display.flip()  # 刷新显示
            
            self.clock.tick(30)  # 限制帧率为30FPS

    def close(self):
        """关闭窗口并释放资源"""
        self.running = False
        pygame.quit()