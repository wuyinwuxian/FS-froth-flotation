import cv2
import numpy as np
from pywt import wavedec2
from skimage import measure,morphology
from scipy import stats
from bisect import bisect_left
from numba import jit

'''
以下是原始版本的计算灰度频次表和计算邻域灰度相关矩阵，几乎完全按照matlab修改而来
'''
# def getgrayfrequency(Frame, d, a, t):             # 计算灰度频次表
#     row, col = Frame.shape[0],Frame.shape[1]
#     px = np.zeros((row - 2*d, col - 2*d))
#     py = np.zeros((row - 2*d, col - 2*d))
#     if t ==1:
#         for i in range(d, (row-d)):
#             for j in range(d, (col - d)):
#                 '''原版的按matlab修改'''
#                 # k = Frame[i, j]
#                 # count = -1
#                 # for m in range((i-d), (i+d+1)):     # 计算Frame[i,j] 附近8个像素点中有几个和它像素值相同的（8是由邻域d确定的）
#                 #     for n in range((j-d), (j+d+1)):
#                 #         if abs(k - Frame[m,n]) <= a:
#                 #             count = count + 1
#                 '''自己改写的'''
#                 count = np.sum(Frame[(i - d):(i + d + 1), (j - d):(j + d + 1)] == Frame[i, j]) - 1
#
#                 '''
#                 熟悉图像处理的人应该知道，如果不扩充像素，这样处理后的矩阵比原矩阵少2*d行，2*d列，上下左右各少d，所以最后的dx、dy会少两行两列
#                 '''
#                 px[i-d,j-d] = Frame[i, j]  # 这个就是Frame[i,j]像素值, 因为下标从0开始。而我的i、j都是从d开始的，所以减一个d，
#                 py[i-d, j-d] = count       # 这个就是Frame[i,j]附近8个像素点中和它像素值相同的点的个数
#     elif t ==2:
#         for i in range(d, (row-d)):
#             for j in range(d, (col - d)):
#                 k = Frame[i,j]
#                 if a == 0:
#                     count = -1
#                 else:
#                     count = 0
#                 for m in range((i-d), (i+d+1)):
#                     for n in range((j-d), (j+d+1)):
#                         if abs(k - Frame[m,n]) <= a:
#                             count = count + 1
#                 px[i-d,j-d] = k
#                 py[i-d, j-d] = count
#     px = px.astype(np.int16)
#     py = py.astype(np.int16)
#     return px, py
#
# def getgraymatrix(Frame,graylevel,d,a,t):                # 计算邻域灰度相关矩阵
#     neighbornums = (2*d + 1)**2                          # 邻域大小，本身加上旁边的构成一个2*d+1的方阵
#     [px, py] = getgrayfrequency(Frame, d, a, t)          # 得到灰度频次表
#     row, col = px.shape[0], px.shape[1]
#     gm = np.zeros((graylevel, neighbornums))
#     for i in range(row):
#         for j in range(col):
#             gm[px[i,j],py[i,j]] = gm[px[i,j],py[i,j]]+1   # 统计像素值是 px[i,j]，且附近8个像素点中和它像素值相同的点的个数为 py[i,j]的像素点个数
#     gm = gm.astype(np.int16)
#     return gm

'''         
自己理解后的更改版本 计算邻域灰度相关矩阵，少了计算灰度频次表的 48万次迭代
使用 jit 装饰器表明我们希望将该函数转换为机器代码，然后参数 nopython 指定我们希望 Numba 采用纯机器代码，
或者有必要的情况加入部分 Python 代码，这个参数必须设置为 True 来得到更好的性能，除非出现错误。
'''
@jit(nopython=True)
def getgraymatrix(Frame,graylevel,d):
    neighbornums = (2*d + 1)**2                 # 邻域大小，本身加上旁边的构成一个2*d+1的方阵
    row, col = Frame.shape[0], Frame.shape[1]
    gm = np.zeros((graylevel, neighbornums))
    for i in range(d, row-d):
        for j in range(d, col-d):
            count = np.sum(Frame[(i - d):(i + d + 1), (j - d):(j + d + 1)] == Frame[i, j]) - 1   # 统计
            gm[Frame[i,j],count] = gm[Frame[i,j],count]+1   # 统计像素值是 Frame[i,j]，且附近8个像素点中和它像素值相同的点的个数为count的像素点个数
    gm = gm.astype(np.int16)
    return gm

@jit(nopython=True)
def solveVelocity(nowFrame,preFrame):
    centralIm1 = nowFrame[300:400, 400:500,0]        # 利用红色通道的像素值进行上下帧图像宏块匹配过程
    sumRowCol1 = np.r_[np.sum(centralIm1,axis=0), np.transpose(np.sum(centralIm1, axis=1))]  # 对前帧泡沫图像中心宏块灰度值行列累计求和
    minDiff = float("inf")
    minDy, minDx = 0, 0
    for dy in range(-100,100):
         for dx in range(-150,150):
             centralIm2 = preFrame[300+dy:400+dy, 400+dx:500+dx,0]
             sumRowCol2 = np.r_[np.sum(centralIm2, axis=0), np.transpose(np.sum(centralIm2, axis=1))]
             dSum = np.sum(abs(sumRowCol1 - sumRowCol2))
             if dSum < minDiff:
                minDiff = dSum
                minDy = dy
                minDx = dx
    velocity = np.sqrt(minDx**2 + minDy**2)           # 11速度特征  速度特征的求取
    return velocity

def sloveHueMean(Frame):
    hsv_im = cv2.cvtColor(Frame, cv2.COLOR_RGB2HSV)   # 色相(Hue)、饱和度(Saturation)、明度(Value)
    hueMean = (np.mean(hsv_im[:,:,1])) /360           #3 色调均值  范围0~1  应该是饱和度均值
    return hueMean

def waveletDecomposition(grayIm, l, w):
    coeffs = wavedec2(data=grayIm, wavelet=w, level=l)
    [cl, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)] = coeffs   # 返回值介绍见 https://blog.csdn.net/qq_43657442/article/details/109380852
    cl = abs(np.round(cl)).astype(np.int16)           # 系数全转为整型，四舍五入成整数
    cD3 = abs(np.round(cD3)).astype(np.int16)
    return cl, cD3                                    # 返回低频系数和第3层对角线高频系数

def sloveCoarseness(ngldm):
    # ngldm = getgraymatrix(Frame, graylevel, d, a, t)
    row, col = ngldm.shape[0], ngldm.shape[1]
    sumQ = ngldm.sum()
    '''法1 ，按着matlab改过来的 '''
    # tempSum = 0
    # for i in range(row):
    #     for j in range(col):
    #         tempSum = tempSum + float(ngldm[i,j]) * ((j+1)**2)

    '''法2，利用 numpy 的矩阵计算好处'''
    # tempSum = 0
    # for j in range(col):
    #     tempSum = np.sum(ngldm[:,j] * ((j+1)**2)) + tempSum

    '''
    # 方法3，利用辅助矩阵
    生成[[ 1 2 3 ... col]
        [ 1 2 3 ... col]
        ......
        [ 1 2 3 ... col]]的一个辅助矩阵，然后再按元素相乘
    '''
    helpMatrix = np.tile((np.arange(1, col + 1)), (row, 1))   # https://blog.csdn.net/qq_43657442/article/details/109060986
    tempSum = np.sum((np.multiply(ngldm, helpMatrix ** 2)))

    coarseness = tempSum / sumQ                               # 泡沫粗度
    return coarseness

def sloveNonUnifiormaity(ngldm):
    # ngldm = getgraymatrix(Frame, graylevel, d, a, t)
    sumQ = ngldm.sum()
    '''
    法1 ，按着matlab改过来的
    col = ngldm.shape[1]
    tempSum = 0
    tempSS = 0
    for i in range(col):
        tempSum = np.sum(ngldm[:, i])
        tempSS = tempSum * tempSum + tempSS
    '''
    tempSum = np.sum(ngldm[:, -1])                            # 法2，利用 numpy 的矩阵计算好处
    nonUnifiormaity = tempSum / sumQ                          # 15 数值非均匀度
    return nonUnifiormaity

def sloveSecondMoment(ngldm):
    # ngldm = getgraymatrix(Frame, graylevel, d, a, t)
    sumQ = ngldm.sum()
    '''
    法1 ，按着matlab改过来的
    row, col = ngldm.shape[0], ngldm.shape[1]
    tempSum = 0
    for i in range(row):
        for j in range(col):
            tempSum = tempSum + ngldm[i,j]**2
    '''
    ngldm = ngldm.astype(np.int64)                            # 要转为64位的，不然的话平方会溢出
    tempSum = np.sum(ngldm**2)                                # 法2，利用 numpy 的矩阵计算好处
    secondMoment = tempSum / sumQ                             #  二阶矩
    return secondMoment

def sloveHF_energy(ngldm):
    # ngldm = getgraymatrix(Frame, graylevel, d, a, t)
    row, col = ngldm.shape[0], ngldm.shape[1]
    '''
    法1 ，按着matlab改过来的
    '''
    # tempSum = 0
    # for i in range(row):
    #     for j in range(col):
    #         tempSum = tempSum + (float(ngldm[i, j])**2) * (i+1)
    '''
    # 法2 利用辅助矩阵生成
            [[ 1 1 1 ... 1]
            [ 2 2 2 ... 2]
            ......
            [row row row ... row]]的一个辅助矩阵，然后再按元素相乘
    '''
    helpMatrix = np.tile((np.arange(1, row + 1)).reshape(-1, 1), (1, col))  # reshape(-1, 1)是将其变成列向量
    tempSum = np.sum((np.multiply(ngldm**2, helpMatrix)))                   # np.multiply 是按元素相乘

    HF_energy = tempSum                                       #18 高频能量
    return HF_energy

def sloveBubble_load(grayIm,row,col):
    number_2 = max([row,col])
    im = grayIm.reshape(-1,order='F')                         # 按列优先去将矩阵展开成一个向量
    number_l = len(np.argwhere(im < 80))                      # 找到灰度值小于80的，我们认为有矿
    transpantNum = max([number_l-number_2,0])
    bubble_load = 1 - transpantNum /row / col
    return bubble_load                                        #9  承载率

def sloveScale(number, width, height):
    scales = np.zeros(number)
    for i in range(number):
        temp = (width[i] / height[i]) if (width[i] > height[i]) else height[i] / width[i]
        scales[i] = temp
    scale = np.mean(scales)
    return scale                                              #8 长宽比

def slove_R_Bsize(labels,number):
    props = measure.regionprops(labels)
    area = np.array([props[i].area for i in range(number - 1)])  # 值为0的标签将被忽略。也就是背景不算
    area = area / np.max(area)
    radius = np.sqrt(area) * 0.3
    bubble_size, t = np.histogram(area, 20)  # 计算面积落在各个区间的数目，把最小面积和最大面积分成20个区间
    return bubble_size.astype(np.uint8), radius               # 泡沫大小与半径

def water(grayIm,nowFrame):
    # 测试完毕，成功
    def imadjust(src, tol=1, vin=[0, 255], vout=(0, 255)):
        # src : input one-layer image (numpy array)
        # tol : tolerance, from 0 to 100.
        # vin  : src image bounds
        # vout : dst image bounds
        # return : output img
        assert len(src.shape) == 2, 'Input image should be 2-dims'
        tol = max(0, min(100, tol))
        if tol > 0:
            # Compute in and out limits
            # Histogram
            hist = np.histogram(src, bins=list(range(256)), range=(0, 255))[0]
            # Cumulative histogram
            cum = hist.copy()
            for i in range(1, 255): cum[i] = cum[i - 1] + hist[i]
            # Compute bounds
            total = src.shape[0] * src.shape[1]
            low_bound = total * tol / 100
            upp_bound = total * (100 - tol) / 100
            vin[0] = bisect_left(cum, low_bound)
            vin[1] = bisect_left(cum, upp_bound)
        # Stretching
        scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
        vs = src - vin[0]
        vs[src < vin[0]] = 0
        vd = vs * scale + 0.5 + vout[0]
        vd[vd > vout[1]] = vout[1]
        dst = vd
        return dst

    grayIm = grayIm.astype(np.uint8)
    K = imadjust(grayIm).astype(np.uint8)  # 对比度增强
    Ds = cv2.subtract(K, 10)
    '''
    用于开运算的结构算子，python opencv里面没有与matlab strel('disk',r)对应的函数,该函数产生一个圆形的结构算子
    有一个cv2.getStructuringElement()函数，但只能产生矩形、椭圆、十字交叉三种，所以我将其写死了
    '''
    kernel = np.array([ [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
                      , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
                      , [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                      , [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                      , [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                      , [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                      , [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                      , [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                      , [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                      , [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                      , [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                      , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
                      , [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]).astype(np.uint8)

    Ioc = cv2.morphologyEx(Ds, cv2.MORPH_OPEN, kernel)  # 开运算，和matlab有一点区别，但基本只相差1，问题不大

    def imreconstruct(marker, mask, SE=np.ones([3, 3])):   # 开重构
        """
        描述：以mask为约束，连续膨胀marker，实现形态学重建，其中mask >= marker
        参数：
            - marker 标记图像，单通道/三通道图像
            - mask   模板图像，与marker同型
            - conn   联通性重建结构元，参照matlab::imreconstruct::conn参数，默认为8联通。默认结构算子是3*3，全1的
        """
        while True:
            marker_pre = marker
            dilation = cv2.dilate(marker, kernel=SE)
            marker = np.min((dilation, mask), axis=0)
            if (marker_pre == marker).all():
                break
        return marker

    Iobr = imreconstruct(Ioc, Ds)
    Iobrd = cv2.dilate(Iobr, kernel)

    def imcomplement(img):  # 负片函数，相当于对图片取反 0<->255, 10<->245, 21<->234
        table = np.array([255 - i for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)  # 使用OpenCV的查找表函数

    Iobrcbr = imreconstruct(imcomplement(Iobrd), imcomplement(Iobr))
    '''
    matlab watershed函数 https://blog.csdn.net/eric_hf_chen/article/details/80778047
    python watershed函数 https://blog.csdn.net/dcrmg/article/details/52498440
    两个不是一个东西
    '''
    def imregionalmin(image):
        """Similar to matlab's imregionalmin"""
        reg_max_loc = morphology.local_minima(image)
        return reg_max_loc.astype(np.uint8)

    bgm = imregionalmin(Iobrcbr)
    labels = measure.label(bgm).astype(np.uint8)
    # k = segmentation.watershed(grayIm,markers=labels)     # 分水岭算法测试不对，和matlab的输出相差太远，不知道怎么弄，所以简单的返回一个标签吧
    return labels


def getFeatures(nowFrame,velocity,d,w):
    row = nowFrame.shape[0]            # 600 图像的高
    col = nowFrame.shape[1]            # 800 图像的宽
    l = nowFrame.shape[2]              # 3   l是小波分解层次（也是通道数）
    grayIm = cv2.cvtColor(nowFrame, cv2.COLOR_RGB2GRAY)  # 转为灰度图
    grayMean = np.mean(grayIm)         # 1  灰度均值
    hueMean = sloveHueMean(nowFrame)   # 3  色调均值范围0~1
    rMean = np.mean(nowFrame[:,:,0])   # 5  R均值
    gMean = np.mean(nowFrame[:,:,1])   # 6  G均值
    bMean = np.mean(nowFrame[:,:,2])   # 7  B均值
    R_per = rMean / grayMean           # 10 相对红色分量

    grayIm = np.round(grayIm).astype(np.int16)    # 我也不知道为啥要转成16位整型
    im1, im4 = waveletDecomposition(grayIm, l, w)
    graylevel1 = np.max(im1)+1
    graylevel4 = np.max(im4)+1

    ngldm = getgraymatrix(grayIm, graylevel1, d)
    ngldm1 = getgraymatrix(im1, graylevel1, d)
    ngldm4 = getgraymatrix(im4, graylevel4, d)

    coarseness = sloveCoarseness(ngldm)                # 14 泡沫粗度
    nonUnifiormaity = sloveNonUnifiormaity(ngldm)      # 15 数值非均匀度
    secondMoment = sloveSecondMoment(ngldm)            # 16 二阶矩
    LF_Coarseness = sloveCoarseness(ngldm1)            # 17 低频子图粗度
    HF_energy = sloveHF_energy(ngldm4)                 # 18 高频能量
    bubble_load = sloveBubble_load(grayIm, row, col)   # 9  承载率

    Frame = water(grayIm,nowFrame)                                       # 做一个分水岭变换，得到泡沫的区域编号，可惜没有成功，下面的函数都测试过，基本和matlab同输入同输出
    number, labels, Statistics, centroids = cv2.connectedComponentsWithStats(Frame)   # labels背景为0,区域从1开始编号，https://blog.csdn.net/qq_40784418/article/details/106023288
    scale = sloveScale(number, Statistics[:, 2], Statistics[:, 3])       #8  长宽比
    bubble_size_mean = np.sqrt(row*col/number)*0.3                       #2  泡沫大小均值
    bubble_size, radius = slove_R_Bsize(labels,number)                   # 泡沫分布与半径
    bubble_size_std = np.std(radius)                                     #4  大小分布方差
    bubble_kurtois = stats.kurtosis(radius, fisher=False)                #12 陡峭度    https://blog.csdn.net/qq_43657442/article/details/109553149
    bubble_sknew = stats.skew(radius)                                    #13 偏斜度    https://blog.csdn.net/qq_43657442/article/details/109553381

    result = np.array([grayMean,bubble_size_mean,hueMean,bubble_size_std,rMean,gMean,bMean,     #'1灰度','2泡沫平均大小','3色调','4大小标准差','5红色均值','6绿色均值','7蓝色均值'
                   scale,bubble_load,R_per,velocity,bubble_kurtois,bubble_sknew,                #'8长宽比','9承载率','10相对红色分量','11速度','12陡峭度','13偏斜度'
                   coarseness,nonUnifiormaity,secondMoment,LF_Coarseness,HF_energy,             #'14粗度','15非均匀度','16二阶矩','17低频子图粗度','18高频能量'
                   *bubble_size])                                                               #泡沫大小
    return np.around(result, decimals=5)     #保留5位小数