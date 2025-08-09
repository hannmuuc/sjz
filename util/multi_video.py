

from itertools import count
import multiprocessing
import cv2
import numpy as np
from queue import Empty
import time
from util.model import RapidOcr,AnchorModel,RapidOcrGPU
from util.video import doOcr,saveImg,readResult
from ast import If
import cv2
from util.model import RapidOcr,AnchorModel,RapidOcrGPU
import cv2
import time
from datetime import datetime, timedelta
import logging
import sys
import numpy as np


def process_worker(input_queue, output_queue, stop_event,process_id,use_cuda):

    """子进程处理函数：从输入队列获取图片，处理后放入输出队列"""

    ocrModel = RapidOcr()
    anchorModel = AnchorModel()
    if use_cuda:
        ocrModel = RapidOcrGPU()

    while not stop_event.is_set():
        try:
            # 从输入队列获取图片（超时1秒，避免阻塞）
            img_id,start_time, img = input_queue.get(timeout=1)
        except Empty:
            continue  # 继续等待，不终止进程

        result = doOcr(img,ocrModel,anchorModel,use_cuda)

        if result == 1:
            saveImg(img, img_id, isSearch=True)
        elif img_id % 10 == 0:
            saveImg(img, img_id, isSearch=False)

        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算耗时（秒）
        
        # 将结果放入输出队列
        output_queue.put((process_id,elapsed_time))
        
        # 标记任务完成
        input_queue.task_done()
    
    print("Worker process exiting...")


def result_collector(output_queue, stop_event,num_workers,afterTime):
    """结果收集进程：从输出队列获取结果并保存"""
    ocrTimeList = []
    count = 0
    for i in range(num_workers):
        ocrTimeList.append(1.0)

    while not stop_event.is_set() or not output_queue.empty():
        try:
            img_id, elapsed_time = output_queue.get(timeout=1)
        except Empty:
            continue
        # 更新时间
        index = img_id%num_workers
        prior_time = ocrTimeList[index]
        ocrTimeList[index] = elapsed_time*0.1+prior_time*0.9

        # 更新共享时间
        with afterTime.get_lock():  # 加锁避免竞态条件
            afterTime.value = (max(ocrTimeList)+0.1)/num_workers

        # 标记结果处理完成
        output_queue.task_done()

        count += 1

        if count%100 == 0:
            print(f"第{count}次任务结束")
    
    print("Result collector exiting...")

class MultiTask:
    def __init__(self,video_file,num_workers,use_cuda=False):
        """主函数：管理进程和队列，模拟图片输入"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)  # 输出到控制台
            ]
        )
        # 确保stdout无缓冲，实时输出
        sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
        
        # 创建进程池（根据CPU核心数调整）
        self.afterTime = multiprocessing.Value('d', 1.0)
        self.use_cuda = use_cuda


        self.num_workers = num_workers
        self.workers = []
        self.ocrTimeList = []
        # 创建队列和事件
        self.input_queue = multiprocessing.JoinableQueue(maxsize=10*num_workers)  # 限制队列大小避免内存溢出
        self.output_queue = multiprocessing.JoinableQueue()
        self.stop_event = multiprocessing.Event()
        self.video_file = video_file

        
        # 启动处理工作进程
        for i in range(self.num_workers):
            p = multiprocessing.Process(
                target=process_worker,
                args=(self.input_queue, self.output_queue, self.stop_event,i,use_cuda)
            )
            p.daemon = True  # 设置为守护进程，主程序退出时自动终止
            p.start()
            self.workers.append(p)
            self.ocrTimeList.append(1)

        # 启动结果收集进程
        self.collector = multiprocessing.Process(
            target=result_collector,
            args=(self.output_queue, self.stop_event,num_workers,self.afterTime)
        )
        self.collector.daemon = True
        self.collector.start()


    def work(self):
        try:
            self.multiVideoSolve()
        except KeyboardInterrupt:
            print("接收到中断信号，准备退出...")
        finally:
            # 通知所有进程停止
            self.stop_event.set()
            
            # 等待所有进程结束
            for p in self.workers:
                p.join(timeout=2)
            self.collector.join(timeout=2)
            
            # 清理队列
            while not self.input_queue.empty():
                self.input_queue.get()
                self.input_queue.task_done()
            while not self.output_queue.empty():
                self.output_queue.get()
                self.output_queue.task_done()
            
            print("程序已安全退出")


    def multiVideoSolve(self):
        begin = readResult() + 1
    
        cap = cv2.VideoCapture(self.video_file)
    
        if not cap.isOpened():
            logging.error(f"无法打开视频文件: {self.video_file}")
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
        print(f"开始从第 {begin} 帧处理视频，总剩余帧数: {remaining_frames}")

        logging.disable(logging.WARNING)

        while success:
            frame_start = time.time()
            success, frame = cap.read()

            self.input_queue.put((frame_count,time.time(), frame))

            # 等待时间
            time.sleep(self.afterTime.value) 

        
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
                    f"aftertime:{self.afterTime.value:.4f}s "
                    f"预计完成时间: {est_finish_time.strftime('%H:%M:%S')}", flush=True)
            
            frame_count += 1




def multiVideoTask():
    multiTask = MultiTask("./video/train.mp4",2,True)
    multiTask.work()
