
from util.anchor_box import squareDetect,preImageSolve,getImgByZeroMatrix,getSquareParamC,AnchorBound,draw_rectangles_on_image,createContours,draw_rectangles_on_image_with_num
from util.ocr import RapidOcr
from util.videoSolve import doOcr
from collections import deque
import time
import numpy as np
import cv2

class checkStatusModel():
    def __init__(self):
        # 0 未检查状态 1 已检查状态
        self.priorCheck = 0
        self.status =0
        self.ocrModel = RapidOcr()
        self.anchorModel = AnchorBound()
        self.h = 6
        self.w = 6
        self.priorMartrix = np.zeros((self.h,self.w))
        self.resX = -1
        self.resY = -1
        self.resDistance = -1
        self.bidirectional_array = deque()
        self.index =0

    def doCheckWithStatusOne(self,img):
        if self.status != 1:
            return 1,False,None

        self.anchorModel.change_rate( 0.65, 0.9, 0.05, 0.6)
        img_anchor, (rateW1,rateW2, rateH1,rateH2) = self.anchorModel.get(img)

        preImage = preImageSolve(img_anchor)
        res = squareDetect(preImage,self.resX,self.resY,self.resDistance)
        zero_matrix = np.zeros((self.h,self.w), dtype=int)
        flag = 0

        contours = createContours(self.resX,self.resY,self.resDistance,6,6)
        img_anchor_rect = draw_rectangles_on_image_with_num(img_anchor,contours,res,display=False)

        self.index += 1
        cv2.imwrite(f"./output/pic/img_anchor_{self.index}.jpg", img_anchor_rect)

        for i in range(self.h):
            for j in range(self.w):
                if self.priorMartrix[i][j] == 1 and res[i][j] == 0:
                    zero_matrix[i][j] = 1
                    self.priorMartrix[i][j] = 0
                    flag =1
        
        if flag == 0:
            return 1,True,None

        suc,boundImg =getImgByZeroMatrix(img_anchor,zero_matrix,self.resX,self.resY,self.resDistance)
        if suc==False:
            return 1,False,None
        cv2.imwrite(f"./output/pic/boundImg_{self.index}.jpg", boundImg)
        res = self.ocrModel.get(boundImg)

        if len(res.elapse_list) == 0:
            return 1,True,None  
        return 1,True,res.txts  

    def doCheckWithStatusZero(self,img):
        if self.status != 0:
            return 0,False,None
        
        self.anchorModel.change_rate( 0.65, 0.9, 0.05, 0.6)
        img_anchor, (rateW1,rateW2, rateH1,rateH2) = self.anchorModel.get(img)
        suc,x,y,distance = getSquareParamC(img_anchor,display=False,ocrModel=self.ocrModel)
        if suc == False:
            return 0,False,None
        self.resX = x
        self.resY = y
        self.resDistance = distance
        preImage = preImageSolve(img_anchor)
        res = squareDetect(preImage,self.resX,self.resY,self.resDistance)
        self.priorMartrix = res
        self.status = 1
        return 0,True,None

    def doCheckWithStatus(self,img):
        if self.status == 0:
            return self.doCheckWithStatusZero(img)
        else:
            return self.doCheckWithStatusOne(img)
    
    def doCheckQueue(self):
        start = time.time()
        while(len(self.bidirectional_array) > 0):
            img = self.bidirectional_array.popleft()
            tag,succ,res = self.doCheckWithStatus(img)
            if tag == 0 and succ == False:
                self.doQueueSkip(60)
                continue
            if succ == True and res != None:
                print(res)
        end = time.time()
        print(f"checkQueue cost {(end-start)*1000}ms")
    
    def doQueueClearRightBase(self):
        left, right = 0, len(self.bidirectional_array) - 1
        # 二分查找第一个不满足条件的位置
        while left < right:
            mid = (left + right) // 2
            if not self.doCheckIsOk(self.bidirectional_array[mid]):
                right = mid
            else:
                left = mid + 1
        return left

    def doQueueClearRight(self):
        """从右侧清理队列，移除第一个不满足条件及之后的所有元素"""
        if not self.bidirectional_array:
            return
        left = self.doQueueClearRightBase()
                
        if not self.doCheckIsOk(self.bidirectional_array[left]):
            while len(self.bidirectional_array) > left:
                self.bidirectional_array.pop()
        else:
            self.bidirectional_array.clear()

    def doQueueClearLeft(self):
        """从左侧清理队列，保留第一个满足条件及之后的所有元素"""
        if not self.bidirectional_array:
            return
            
        left, right = 0, len(self.bidirectional_array) - 1
        # 二分查找第一个满足条件的位置
        while left < right:
            mid = (left + right) // 2
            if self.doCheckIsOk(self.bidirectional_array[mid]):
                right = mid
            else:
                left = mid + 1
        
        if self.doCheckIsOk(self.bidirectional_array[left]):
            for _ in range(left):
                self.bidirectional_array.popleft()
        else:
            self.bidirectional_array.clear()

    def doQueueSkip(self,skipNum):
        while(len(self.bidirectional_array) > 0 and skipNum > 0):
            self.bidirectional_array.popleft()
            skipNum -= 1

    def doCheckIsOk(self, img):
        """检查图像是否满足条件，简化实现"""
        return doOcr(img, self.ocrModel, self.anchorModel) == 1

    def doCheck(self,img,boundLength):
        self.bidirectional_array.append(img)
        if len(self.bidirectional_array) < boundLength:
            return False,None

        check_res = self.doCheckIsOk(img)

        if self.priorCheck == 0:
            if not check_res:
                self.bidirectional_array.clear()
                self.status = 0
            else:
                self.doQueueClearLeft()
                self.doQueueSkip(10)
                self.doCheckQueue()
                self.priorCheck = 1
        else:
            if not check_res:
                self.doQueueClearRight()
                self.doCheckQueue()
                self.status = 0
                self.priorCheck = 0
            else:
                self.doCheckQueue()
