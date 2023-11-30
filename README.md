### 运行后在键盘上按对应的功能建，运行对应的例程 (After running, according to the key on the keyboard, run the corresponding routine)

requirement: Opencv

使用方式：直接运行，运行后选择功能，在键盘上按对应的功能键

(1)输入“iris”,实现虹膜分割功能。
(2)输入“camera”,调用摄像头实时读取图片,按对应的功能键实现不同类型的操作。
(3)输入“location”,读取本地图片,按对应的功能键实现不同类型的操作。

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
