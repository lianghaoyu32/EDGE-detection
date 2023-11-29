import cv2
import sys
'''
使用方式：直接运行，运行后在键盘上按对应的功能建
----------------------------------------------
  功能键 | 功能
    Q   | 退出
    Z   | 预览镜像原图
    0   | 预览二值图像
    1   | 预览灰度图
    2   | 高斯滤波
    3   | 中值滤波
    4   | 双边滤波
    5   | 水平sobel
    6   | 垂直sobel
    7   | 图像sobel梯度的振幅
    8   | 水平DerivativeOfGaussian滤波检测边缘
    9   | 水平LaplacianOfGaussian滤波检测边缘
    A   | 垂直DerivativeOfGaussian滤波检测边缘
    S   | 垂直LaplacianOfGaussian滤波检测边缘
    D   | DerivativeOfGaussian滤波检测边缘
    F   | LaplacianOfGaussian滤波检测边缘
    W   | 二值图像膨胀
    E   | 二值图像腐蚀 
    R   | 二值图像开操作
    T   | 二值图像闭操作
    Y   | 灰度图像膨胀
    U   | 灰度图像腐蚀 
    I   | 灰度图像开操作
    O   | 灰度图像闭操作
    G   | 提取内边界（原图减腐蚀）
    H   | 提取外边界（膨胀减原图）
    J   | 形态学梯度（膨胀减腐蚀，即内外边界相加）
    P   | Canny算子
'''
print(
    '''
使用方式：直接运行，运行后在键盘上按对应的功能建
----------------------------------------------
 功能键  | 功能
----------------------------------------------
    Q    | 退出
    Z    | 预览镜像原图
    0    | 预览二值图像
    1    | 预览灰度图
    2    | 高斯滤波
    3    | 中值滤波
    4    | 双边滤波
    5    | 水平sobel
    6    | 垂直sobel
    7    | 图像sobel梯度的振幅
    8    | 水平DerivativeOfGaussian滤波检测边缘
    9    | 水平LaplacianOfGaussian滤波检测边缘
    A    | 垂直DerivativeOfGaussian滤波检测边缘
    S    | 垂直LaplacianOfGaussian滤波检测边缘
    D    | DerivativeOfGaussian滤波检测边缘
    F    | LaplacianOfGaussian滤波检测边缘
    W    | 二值图像膨胀
    E    | 二值图像腐蚀 
    R    | 二值图像开操作
    T    | 二值图像闭操作
    Y    | 灰度图像膨胀
    U    | 灰度图像腐蚀 
    I    | 灰度图像开操作
    O    | 灰度图像闭操作
    G    | 提取内边界（原图减腐蚀）
    H    | 提取外边界（膨胀减原图）
    J    | 形态学梯度（膨胀减腐蚀，即内外边界相加）
    P    | Canny算子
----------------------------------------------
'''
)
PREVIEW  = 0   # Preview Mode
PREVIEWBW = 1
GaussianBlur = 2 # GaussianBlur cv.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]	) ->	dst
MedianBlur = 3 # cv.medianBlur(src, ksize[, dst]	) ->	dst
BilateralFilter = 4 # cv.bilateralFilter(src, d, sigmaColor, sigmaSpace[, dst[, borderType]]	) ->	dst
SobelHorizon = 5 # cv.Sobel(	src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]	) ->	dst
SobelVertical = 6
Magnitude = 7
DerivativeOfGaussianHorizon = 8
LaplacianOfGaussianHorizon  = 9
DilateBW = 10 # cv.dilate(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]	) ->	dst
ErodeBW = 11 # cv.erode(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]	) ->	dst
OpenBW = 12
CloseBW = 13

Dilate = 14 # cv.dilate(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]	) ->	dst
Erode = 15 # cv.erode(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]	) ->	dst
Open = 16
Close = 17

DerivativeOfGaussianVertical = 18
LaplacianOfGaussianVertical = 19

DerivativeOfGaussian = 20
LaplacianOfGaussian  = 21

Innerboundary = 22
Outerboundary = 23
Morphologicalgradient = 24

GRAY = 25

Canny = 26

# ShiTomasi corner detection的参数
feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.2,
                       minDistance = 15,
                       blockSize = 9)
s = 0 # 调用笔记本摄像头


image_filter = PREVIEW # 摄像头初始显示
alive = True

win_name = 'Camera Filters'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
result = None

source = cv2.VideoCapture(s)

while alive:
    has_frame, frame = source.read()
    if not has_frame:
        break

    frame = cv2.flip(frame,1)
    frameBASE = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    if image_filter == GRAY: # 灰度图
        result = frameBASE
    if image_filter == PREVIEW: # 原图
        result = frame
    if image_filter == PREVIEWBW: # 二值图
        ret,result = cv2.threshold(frameBASE,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # 二进制阈值与大津法阈值相结合
    elif image_filter == GaussianBlur:
        result = cv2.GaussianBlur(src=frameBASE, ksize=(9,9), sigmaX=5, sigmaY=8)
    elif image_filter == MedianBlur: 
        result = cv2.medianBlur(src=frameBASE, ksize=9)
    elif image_filter == BilateralFilter:
        result = cv2.bilateralFilter(src=frameBASE, d=13,sigmaColor=29, sigmaSpace=29)
    elif image_filter == SobelHorizon:
        result = cv2.Sobel(src=frameBASE, ddepth=cv2.CV_8U, dx=1, dy=0, ksize=3)
    elif image_filter == SobelVertical:
        result = cv2.Sobel(src=frameBASE, ddepth=cv2.CV_8U, dx=0, dy=1, ksize=3)
    elif image_filter == Magnitude:
        resultHorizon = cv2.Sobel(src=frameBASE, ddepth=cv2.CV_8U, dx=1, dy=0, ksize=3)
        resultVertical = cv2.Sobel(src=frameBASE, ddepth=cv2.CV_8U, dx=0, dy=1, ksize=3)
        result = cv2.addWeighted(src1=resultHorizon, alpha=0.5, src2=resultVertical, beta=0.5, gamma=0)
    elif image_filter == DerivativeOfGaussianHorizon:
        result = cv2.GaussianBlur(src=frameBASE, ksize=(5,5), sigmaX=3, sigmaY=3)
        result = cv2.Sobel(src=result, ddepth=cv2.CV_8U, dx=0, dy=1, ksize=3)
    elif image_filter == LaplacianOfGaussianHorizon:
        result = cv2.GaussianBlur(src=frameBASE, ksize=(5,5), sigmaX=3, sigmaY=3)
        result = cv2.Sobel(src=result, ddepth=cv2.CV_8U, dx=0, dy=1, ksize=3)
        result = cv2.Sobel(src=result, ddepth=cv2.CV_8U, dx=0, dy=1, ksize=3)
    elif image_filter == DerivativeOfGaussianVertical:
        result = cv2.GaussianBlur(src=frameBASE, ksize=(5,5), sigmaX=3, sigmaY=3)
        result = cv2.Sobel(src=result, ddepth=cv2.CV_8U, dx=1, dy=0, ksize=3)
    elif image_filter == LaplacianOfGaussianVertical:
        result = cv2.GaussianBlur(src=frameBASE, ksize=(5,5), sigmaX=3, sigmaY=3)
        result = cv2.Sobel(src=result, ddepth=cv2.CV_8U, dx=1, dy=0, ksize=3)
        result = cv2.Sobel(src=result, ddepth=cv2.CV_8U, dx=1, dy=0, ksize=3)
    elif image_filter == DerivativeOfGaussian:
        result = cv2.GaussianBlur(src=frameBASE, ksize=(5,5), sigmaX=3, sigmaY=3)
        resultHorizon = cv2.Sobel(src=result, ddepth=cv2.CV_8U, dx=0, dy=1, ksize=3)
        result = cv2.GaussianBlur(src=frameBASE, ksize=(5,5), sigmaX=3, sigmaY=3)
        resultVertical = cv2.Sobel(src=result, ddepth=cv2.CV_8U, dx=1, dy=0, ksize=3)
        result = cv2.addWeighted(src1=resultHorizon, alpha=0.5, src2=resultVertical, beta=0.5, gamma=0)
    elif image_filter == LaplacianOfGaussian:
        result = cv2.GaussianBlur(src=frameBASE, ksize=(5,5), sigmaX=3, sigmaY=3)
        result = cv2.Sobel(src=result, ddepth=cv2.CV_8U, dx=0, dy=1, ksize=3)
        resultHorizon = cv2.Sobel(src=result, ddepth=cv2.CV_8U, dx=0, dy=1, ksize=3)
        result = cv2.GaussianBlur(src=frameBASE, ksize=(5,5), sigmaX=3, sigmaY=3)
        result = cv2.Sobel(src=result, ddepth=cv2.CV_8U, dx=1, dy=0, ksize=3)
        resultVertical = cv2.Sobel(src=result, ddepth=cv2.CV_8U, dx=1, dy=0, ksize=3)
        result = cv2.addWeighted(src1=resultHorizon, alpha=0.5, src2=resultVertical, beta=0.5, gamma=0)
    elif image_filter == DilateBW:
        ret,result = cv2.threshold(frameBASE,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        result = cv2.dilate(src=result,kernel=cv2.getStructuringElement(shape=cv2.MORPH_RECT,ksize=(10,10)))
    elif image_filter == ErodeBW:
        ret,result = cv2.threshold(frameBASE,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        result = cv2.erode(src=result,kernel=cv2.getStructuringElement(shape=cv2.MORPH_RECT,ksize=(10,10)))
    elif image_filter == OpenBW:
        ret,result = cv2.threshold(frameBASE,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        result = cv2.erode(src=result,kernel=cv2.getStructuringElement(shape=cv2.MORPH_RECT,ksize=(10,10)))
        result = cv2.dilate(src=result,kernel=cv2.getStructuringElement(shape=cv2.MORPH_RECT,ksize=(10,10)))
    elif image_filter == CloseBW:
        ret,result = cv2.threshold(frameBASE,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        result = cv2.dilate(src=result,kernel=cv2.getStructuringElement(shape=cv2.MORPH_RECT,ksize=(10,10)))
        result = cv2.erode(src=result,kernel=cv2.getStructuringElement(shape=cv2.MORPH_RECT,ksize=(10,10)))
    elif image_filter == Dilate:
        result = cv2.dilate(src=frameBASE,kernel=cv2.getStructuringElement(shape=cv2.MORPH_RECT,ksize=(10,10)))
    elif image_filter == Erode:
        result = cv2.erode(src=frameBASE,kernel=cv2.getStructuringElement(shape=cv2.MORPH_RECT,ksize=(10,10)))
    elif image_filter == Open:
        result = cv2.erode(src=frameBASE,kernel=cv2.getStructuringElement(shape=cv2.MORPH_RECT,ksize=(10,10)))
        result = cv2.dilate(src=result,kernel=cv2.getStructuringElement(shape=cv2.MORPH_RECT,ksize=(10,10)))
    elif image_filter == Close:
        result = cv2.dilate(src=frameBASE,kernel=cv2.getStructuringElement(shape=cv2.MORPH_RECT,ksize=(10,10)))
        result = cv2.erode(src=result,kernel=cv2.getStructuringElement(shape=cv2.MORPH_RECT,ksize=(10,10)))
    elif image_filter == Innerboundary:
        resulterode = cv2.erode(src=frameBASE,kernel=cv2.getStructuringElement(shape=cv2.MORPH_RECT,ksize=(10,10)))
        result = frameBASE - resulterode
    elif image_filter == Outerboundary:
        resultdilate = cv2.dilate(src=frameBASE,kernel=cv2.getStructuringElement(shape=cv2.MORPH_RECT,ksize=(10,10)))
        result = resultdilate - frameBASE
    elif image_filter == Morphologicalgradient:
        resulterode = cv2.erode(src=frameBASE,kernel=cv2.getStructuringElement(shape=cv2.MORPH_RECT,ksize=(10,10)))
        resultdilate = cv2.dilate(src=frameBASE,kernel=cv2.getStructuringElement(shape=cv2.MORPH_RECT,ksize=(10,10)))
        result = resultdilate - resulterode
    elif image_filter == Canny:
        result = cv2.Canny(frameBASE, 32, 128, apertureSize = 3, L2gradient = True)
    cv2.imshow(win_name, result)

    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q') or key == 27:
        alive = False
    elif key == ord('0'):   
        image_filter = PREVIEWBW
    elif key == ord('1'):   
        image_filter = GRAY
    elif key == ord('2'):   
        image_filter = GaussianBlur
    elif key == ord('3'):   
        image_filter = MedianBlur
    elif key == ord('4'):   
        image_filter = BilateralFilter
    elif key == ord('5'):   
        image_filter = SobelHorizon
    elif key == ord('6'):   
        image_filter = SobelVertical
    elif key == ord('7'):   
        image_filter = Magnitude
    elif key == ord('8'):   
        image_filter = DerivativeOfGaussianHorizon
    elif key == ord('9'):   
        image_filter = LaplacianOfGaussianHorizon
    elif key == ord('W') or key == ord('w'):   
        image_filter = DilateBW
    elif key == ord('E') or key == ord('e'):    
        image_filter = ErodeBW
    elif key == ord('R') or key == ord('r'):   
        image_filter = OpenBW
    elif key == ord('T') or key == ord('t'):    
        image_filter = CloseBW
    elif key == ord('Y') or key == ord('y'):   
        image_filter = Dilate
    elif key == ord('U') or key == ord('u'):    
        image_filter = Erode
    elif key == ord('I') or key == ord('i'):   
        image_filter = Open
    elif key == ord('O') or key == ord('o'):    
        image_filter = Close
    elif key == ord('A') or key == ord('a'):  
        image_filter = DerivativeOfGaussianVertical
    elif key == ord('S') or key == ord('s'):  
        image_filter = LaplacianOfGaussianVertical
    elif key == ord('D') or key == ord('d'):  
        image_filter = DerivativeOfGaussian
    elif key == ord('F') or key == ord('f'):  
        image_filter = LaplacianOfGaussian
    elif key == ord('G') or key == ord('g'):  
        image_filter = Innerboundary
    elif key == ord('H') or key == ord('h'):  
        image_filter = Outerboundary
    elif key == ord('J') or key == ord('j'):  
        image_filter = Morphologicalgradient
    elif key == ord('Z') or key == ord('z'):  
        image_filter = PREVIEW
    elif key == ord('P') or key == ord('p'):  
        image_filter = Canny
source.release()
cv2.destroyWindow(win_name)