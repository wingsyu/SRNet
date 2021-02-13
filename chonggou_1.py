# -*- encoding: utf-8 -*-

import argparse
import datetime
import multiprocessing
import random
import threading

from PIL import Image
from scipy import fft
import pandas as pd
from scipy.stats import norm
from scipy.signal import find_peaks
import scipy.stats
from statsmodels.tsa import tsatools, stattools

from scipy import stats
from statsmodels.tsa.stattools import adfuller, coint
import numpy as np
# import keras
import requests
# from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# from PIL import Image
# from keras.preprocessing import image
import cv2
from imgaug import augmenters as iaa
import torch, torchvision
import os
from random import sample
from torchvision import datasets
from torchvision import transforms
import torch.utils.data as utils
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
import shutil
import time
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import math
from pathlib import Path
from sklearn.metrics import mean_squared_error
from multiprocessing import Pool
import statsmodels.api as sm

# 本节介绍数据增强的几个方法：
## 首先是keras的随机干扰，生成一定数量的随机干扰图片
from main_1 import WORK_DIR, experience_id, shape, model_name

snake = ['n01728572', 'n01728920', 'n01729322', 'n01729977', 'n01734418', 'n01735189', 'n01737021',
         'n01739381', 'n01740131', 'n01742172', 'n01744401', 'n01748264', 'n01749939', 'n01751748',
         'n01753488', 'n01755581', 'n01756291']

butterfly = ['n02276258', 'n02277742', 'n02279972', 'n02280649',
             'n02281406', 'n02281787']

cat = ['n02123045', 'n02123159', 'n02123394', 'n02123597', 'n02124075',
       'n02125311', 'n02127052']

leopard = ['n02128385', 'n02128757', 'n02128925']

dog = ['n02085620', 'n02085782',
       'n02085936', 'n02086079', 'n02086240', 'n02086646', 'n02086910', 'n02087046', 'n02087394',
       'n02088094']

fish = ['n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041']

bird = ['n02002724', 'n02006656', 'n02007558', 'n02009229', 'n02009912', 'n02011460', 'n02012849',
        'n02013706']

spider = ['n01773157', 'n01773549', 'n01773797', 'n01774384', 'n01774750', 'n01775062',
          'n01776313']

monkey = ['n02483362', 'n02487347', 'n02494079', 'n02486410']

lizard = ['n01675722', 'n01677366', 'n01682714', 'n01685808', 'n01687978', 'n01688243', 'n01689811', 'n01692333',
          'n01693334', 'n01694178', 'n01695060']

wall_g = ['n01629819', 'n01630670', 'n01631663', 'n01632458', 'n01632777']

fox = ['n02119022', 'n02119789', 'n02120079', 'n02120505']

li = ['n02441942', 'n02442845', 'n02443114', 'n02443484', 'n02444819', 'n02445715', 'n02447366']

ox = ['n02403003', 'n02408429', 'n02410509']

sheep = ['n02412080', 'n02415577', 'n02417914', 'n02422106', 'n02422699', 'n02423022']

mushroom = ['n12985857', 'n12998815', 'n13037406', 'n13040303', 'n13044778', 'n13052670', 'n13054560']

violin = ["n04536866"]

# pool table, paintbrush, padlock, bowl, holster, baseball, bottlecap, pll bottle, toilet seat,coral fungus,corn
test_class = ['n03982430', 'n03876231', 'n03874599', 'n03775546', 'n03527444', 'n02799071', 'n02877765',
              'n03937543', 'n04447861', 'n12985857', 'n12144580']

concepts = [snake]
random_class = []
for concept in concepts:
    random_class.extend(concept)


# def keras_dataGenerator(source_path, target_path, img_num):
#     '''
#     # 函数作用是将source_path文件夹中的图片进行随机干扰变换，然后将其输出到目标文件夹。
#     :param source_path: 源文件夹，存放用于干扰的图片
#     :param target_path: 目标文件夹，存放源文件夹中图片干扰之后的图片
#     :param img_num: 需要生成的干扰图片的数目
#     :return: 无
#     '''
#
#     fill_mode = ["reflect", "wrap", "nearest"]
#     datagen = image.ImageDataGenerator(
#         zca_whitening=True,
#         rotation_range=30,
#         width_shift_range=0.03,
#         height_shift_range=0.03,
#         shear_range=0.5,
#         zoom_range=0.1,
#         channel_shift_range=100,
#         horizontal_flip=True,
#         fill_mode=fill_mode[np.random.randint(3)]
#     )
#     gen_data = datagen.flow_from_directory(source_path,
#                                            batch_size=1,
#                                            shuffle=False,
#                                            save_to_dir=target_path,
#                                            save_prefix="gen",
#                                            target_size=(224, 224))
#     while (img_num > 0):
#         gen_data.next()
#         img_num -= 1


## 生成固定干扰的图片，为了便于两个图片协同变换。方法弃用
def generate_coin(sigma):  # crop, fliplr, sigma, connor, flipud, x_s, y_s, x_t, y_t, rotate, shear, order, mode):

    seq = iaa.Sequential({  # 建立一个名为seq的实例，定义增强方法，用于增强
        # iaa.Crop(px=(crop, crop+1)),  # 对图像进行crop操作，随机在距离边缘的0到16像素中选择crop范围
        # # iaa.Fliplr(0.5),  # 对百分之五十的图像进行做左右翻转
        # iaa.GaussianBlur((0, 1.0)),  # 在模型上使用0均值1方差进行高斯模糊
        #
        # iaa.Fliplr(fliplr),
        iaa.GaussianBlur(sigma=sigma)
        # iaa.contrast.LinearContrast(connor, per_channel=True),
        # iaa.Flipud(flipud),
        #
        # iaa.Affine(
        #     scale={"x": x_s, "y": y_s},
        #     translate_percent={"x": x_t, "y": y_t},
        #     rotate=rotate,
        #     shear=shear,
        #     order=order,
        #     mode=mode
        # )
    })
    return seq


## 生成固定干扰的图片，为了便于两个图片协同变换。
def generate_coinT(aug_num, thread_num, img_list_dir="/home/hongwu/python/Image/exper_v1/marker_C/",
                   save_dir="/home/hongwu/python/Image/exper_v1/marker_C/"):
    # aug_num = args[0]
    # thread_num = args[1]

    print("线程序号：", thread_num)
    try:
        imglist = np.load(img_list_dir + "/result/data_imglist_" + str(thread_num) + ".npy",
                          allow_pickle=True)
        print(np.shape(imglist))  # (1, ..,.,.)
        img_num = 1

        # sleepawhile(3)

        imglist = [imglist]
        # crop_ind = 0
        # fliplr_ind = 0
        # sigma_ind = 1
        # connor_ind = 0
        # flipud_ind = 0
        # x_s_ind = 0
        # rotate_ind = 0
        # shear_ind = 0
        # order_ind = 0
        # mode_ind = 0

        for i in range(aug_num):
            #     crop =  random.randint(0, 20)
            # crop = (0.01 * np.abs(np.sin(i / 3.14))) if crop_ind else 0
            # crop = int(20 * (i/aug_num)) if crop_ind else 0
            # fliplr = (i % 2) if fliplr_ind else 0
            # # sigma = 5 * np.abs(np.cos(i / 3.14)) if sigma_ind else 0
            # sigma = 5 * (i/aug_num) # if sigma_ind else 0
            #
            # # (0.75-1.5)
            # # connor = np.sin(i / 3.14) * 0.75 + 0.75 if connor_ind else 1
            # connor = (i/aug_num) * 0.75 + 0.75 if connor_ind else 1
            #
            # flipud = i % 2 if flipud_ind else 0
            #
            # if x_s_ind:
            #     # x_s = np.sin(i / 3.14) * 0.4 + 0.8
            #     # y_s = np.sin(i / 3.14) * 0.4 + 0.8
            #     # x_t = np.sin(i / 3.14) * 0.4 - 0.2
            #     # y_t = np.sin(i / 3.14) * 0.4 - 0.2
            #     x_s = (i/aug_num) * 0.4 + 0.8
            #     y_s = (i/aug_num) * 0.4 + 0.8
            #     x_t = (i/aug_num) * 0.4 - 0.2
            #     y_t = (i/aug_num) * 0.4 - 0.2
            # else:
            #     x_s = 1
            #     y_s = 1
            #     x_t = 0
            #     y_t = 0
            #
            # # rotate = int(90 * np.sin(i / 3.14) - 45) if rotate_ind else 0
            # rotate = int(90 * (2*i/aug_num-1)) if rotate_ind else 0
            # # shear = int(32 * np.sin(i / 3.14) - 16) if shear_ind else 0
            # shear = int(32 * (2*i/aug_num-1) - 16) if shear_ind else 0
            # order = (i % 2) if order_ind else 0
            #
            # c = ["edge", "symmetric", "reflect", "wrap"]
            # mode = c[i % 4] if mode_ind else "constant"
            # mode = "constant"

            # if i % 50 == 0:
            #     print(i)

            # seq = generate_coin(sigma) #crop, fliplr, sigma, connor, flipud, x_s, y_s, x_t, y_t, rotate, shear, order, mode)

            # 变换的种类，可以在此选择，然后找出有意义的干扰。
            # seq = iaa.Sequential({  # 建立一个名为seq的实例，定义增强方法，用于增强
            #     # iaa.Crop(px=(crop, crop+1)),  # 对图像进行crop操作，随机在距离边缘的0到16像素中选择crop范围
            #     # # iaa.Fliplr(0.5),  # 对百分之五十的图像进行做左右翻转
            #     # iaa.GaussianBlur((0, 1.0)),  # 在模型上使用0均值1方差进行高斯模糊
            #     #
            #     # iaa.Fliplr(fliplr),
            #     iaa.GaussianBlur(sigma=sigma)
            #     # iaa.contrast.LinearContrast(connor, per_channel=True),
            #     # iaa.Flipud(flipud),
            #     #
            #     # iaa.Affine(
            #     #     scale={"x": x_s, "y": y_s},
            #     #     translate_percent={"x": x_t, "y": y_t},
            #     #     rotate=rotate,
            #     #     shear=shear,
            #     #     order=order,
            #     #     mode=mode
            #     # )
            # })

            # 作为marker_C的时候是需要进行调制的,但是marker_B不需要
            seq = iaa.GaussianBlur(sigma=(6 + 6 * np.sin(7 * i * np.pi / 180)))

            # marker_B
            # seq = iaa.GaussianBlur((0, 1.0))
            images_aug = seq.augment_images(imglist)
            #  这里得出的结果应该是50*图片尺寸，我们只需要直接进行保存就好了，不用全部进行存储
            for img in range(img_num):
                # cv2.imwrite("/home/hongwu/python/Image/mid_result/aug/" + str(thread_num) + "/" + str(img) + "/val/class/" + str(i) +".jpg", images_aug[img])
                # print(np.shape(images_aug[img]))
                cv2.imwrite(
                    save_dir + str(thread_num) + "/val/class/" + str(i) + ".jpg",
                    images_aug[img])

            # for j in range(img_num):
            #     aug_list[j].append(images_aug[j])

            if i % 333 == 0:
                print("线程", thread_num, "进度：", (i / aug_num))


    except Exception as e:
        print(e)
    # print("线程", thread_num, "开始存储")
    # for j in range(img_num):  # （0-49）
    #     for k in range(aug_num):
    #         cv2.imwrite("/home/hongwu/python/Image/mid_result/aug/" + str(thread_num) + '/' + str(j) + '/val/class/' + str(k) + '.jpg', aug_list[j][k])
    #


# 计算各种评估方式的方法
## 首先是预处理方法，为了删除左右异常值
def pre_handle(point):
    """
    方法是为了删除异常点，选择删除两边5%的点
    :param point: 要删除点的数组
    :return: 删除节点之后的数组
    """

    # v1
    # min_ = np.percentile(point, 2.5)  # 2.5%分位数
    # max_ = np.percentile(point, 97.5)
    # n = np.shape(point)[0]
    # count = int(n * 0.95)
    # z = count
    # point_new = np.zeros(n, )
    # k = 0
    # for i in range(n):
    #     if min_ <= point[i] < max_ and count > 0:
    #         point_new[k] = point[i]
    #         k += 1
    #         count -= 1
    #     elif count <= 0:
    #         point_new[k] = point[i]
    #         k += 1
    # point_new = point_new[:z]
    # point = point_new

    # v2
    # n = 0
    # miu = point.mean()
    # sigma = point.std()
    # if sigma == 0:
    #     return point
    # for i in range(np.shape(point)[0]):
    #     if np.abs(point[i] - miu) / sigma < 3:
    #         point[n] = point[i]
    #         n += 1
    # if n < 100:
    #     print(n)
    # return np.resize(point, (n,))

    # v3
    n = 0
    miu = point.mean()
    sigma = point.std()
    if sigma == 0:
        return point

    m_max = miu + 2 * sigma
    m_min = miu - 2 * sigma

    point = (point - m_min) / (m_max - m_min)

    return point


# 进行分箱
def fenxiang(sequence, Bin_number=10, max_=1, min_=0, del_zero=False):
    """
    将数据进行分箱，是为了将离散的数据进行概率化。
    有两种选择： 直接数字分箱或者将其拟合数据再分箱
    :param min_:
    :param max_:
    :param del_zero:
    :param sequence: 数据结果
    :param Bin_number: 箱的个数。默认值100
    :return: 返回分享结果，即为概率的结果数组。
    """
    sequence = sequence.reshape((-1,))

    sequence = (sequence - min_) / (max_ - min_)

    p = np.ones(Bin_number, )

    bins = []
    for low in range(0, 100, int(100 / Bin_number)):
        bins.append((low / 100, (low + 100 / Bin_number) / 100))
    #     print(bins)

    for j in range(np.shape(sequence)[0]):
        for i in range(0, len(bins)):
            if sequence[j] == 0 and del_zero:
                break
            if bins[i][0] <= sequence[j] < bins[i][1]:
                p[i] += 1
    for i in range(Bin_number):
        p[i] = p[i] / (np.shape(sequence)[0] + Bin_number)
    return p


# 计算js散度
def JS_divergence(point_1, point_2, del_zero=False, num_bins=100):
    """
    计算js散度的函数
    :param num_bins:
    :param del_zero:
    :param point_1: point表示要计算js散度的两个值，一般都是相同长度的概率数组。
    :param point_2:
    :return: 返回js散度
    """
    global js

    try:
        point_1 = point_1.reshape(-1, )
        point_2 = point_2.reshape(-1, )
        x_1 = pre_handle(point_1)
        x_2 = pre_handle(point_2)

        min_ = min(min(x_1), min(x_2))
        max_ = max(max(x_1), max(x_2))

        # p = fenxiang(x_1, 10, max_, min_, del_zero)
        # q = fenxiang(x_2, 10, max_, min_, del_zero)

        # # max0 = max(np.max(x_1), np.max(x_2))
        # # min0 = min(np.min(x_1), np.min(x_2))
        # # bins = np.linspace(min0 - 1e-4, max0 - 1e-4, num=num_bins)
        # # PDF1 = pd.cut(x_1, bins).value_counts() / len(x_1)
        # # PDF2 = pd.cut(x_2, bins).value_counts() / len(x_2)
        # # p = PDF1.values
        # # q = PDF2.values
        # M = (p + q) / 2
        # js = 0.5 * entropy(p, M) + 0.5 * entropy(q, M)

        bins = [n / 10 for n in range(0, 11, 1)]  # 箱子边界，0-1之内
        hist_1, bin_edges_1 = np.histogram(x_1, bins)
        hist_2, bin_edges_2 = np.histogram(x_2, bins)
        js = scipy.spatial.distance.jensenshannon(hist_1 + 1, hist_2 + 1)



    except Exception as e:
        print(e)
    return js


def JS_divergence_1(point_1, point_2):
    point_1 = (point_1 + 1e-10)  # /  (point_1.sum() + 1e-7)
    point_2 = (point_2 + 1e-10)  # / (point_2.sum() + 1e-7)
    # print(point_1.sum(), point_2.sum())
    # print(point_1)

    # M = (point_1 + point_2) / 2
    # distance = 0.5 * scipy.stats.entropy(point_1, M) + 0.5 * scipy.stats.entropy(point_2, M)

    n = np.shape(point_2)[0]
    a = np.arange(n)
    distance = wasserstein_distance(a, a, point_1, point_2)
    return distance


# 计算W距离
def W_divergence(point_1, point_2, del_zero=False):
    """
    计算W推土机距离的函数
    :param del_zero:
    :param point_1: 分别是两个相同长度概率数组
    :param point_2:
    :return: 返回W距离
    """

    x_1 = pre_handle(point_1)
    x_2 = pre_handle(point_2)

    p = fenxiang(x_1, 100, del_zero)
    q = fenxiang(x_2, 100, del_zero)
    w = wasserstein_distance(p, q)

    return w


# 计算序列的js散度，就是求每个序列与平均值js然后再平均
def kl_Bin_cal(sequence, Normalize=True, del_zero=False):
    '''
    首先计算n个序列的平均值，然后计算每个序列与平均值的js散度，然后将结果进行平均，作为返回值
    '''
    n = np.shape(sequence)[0]
    sequence = sequence.reshape(n, -1)

    if Normalize:
        sequence = (sequence - np.min(sequence)) / (np.max(sequence) - np.min(sequence))

    # 首先计算平均值，
    shape = np.shape(sequence)[1:]
    sum_ = np.zeros(shape=shape)

    for i in range(n):
        sum_ += sequence[i]
    average = sum_ / n

    kl_seq = np.zeros(n, )

    for i in range(n):
        kl_seq[i] = JS_divergence_1(average, sequence[i], del_zero=del_zero)

    # Bin_number = 100
    # q = fenxiang(average, Bin_number)
    #
    # #     print(q)
    #
    # kl_seq = np.zeros((n,))
    # for i in range(n):
    #     p = fenxiang(sequence[i], Bin_number)  # 返回分箱之后的概率
    #
    #     # for j in range(Bin_number - 1):
    #     #     kl_seq[i] += p[j] * np.log(p[j] / q[j])
    #     kl_seq[i] = JS_divergence_1(p, q, del_zero)
    #     if i % 20 == 0:
    #         print(kl_seq[i])
    # #         print(kl_seq[i])

    return kl_seq


# 计算数据的psnr
def psnr_cal(x, y):
    """
    计算两个向量或者矩阵的psnr
    :param x: 向量
    :param y:
    :return: psnr
    """
    PIXEL_MAX = 1
    #     x = x / PIXEL_MAX
    #     y = y /PIXEL_MAX
    #     PIXEL_MAX = 1
    ## mse = np.mean( (x/1 - y/1) ** 2 )
    mse = np.mean((x / 255. - y / 255.) ** 2)
    print(mse)
    if mse < 1.0e-10:
        return 100

    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def cal_psnr(data, Standard=False, Normalize=False):
    """
    计算多维数据的psnr
    :param data: 数据
    :param Standard: 是否对数据进行标准化
    :param Normalize: 是否对数据进行归一化
    :return:
    """

    # 定义两开关：standard 为标准化，Normalize为归一化
    if Standard:
        data = (data - np.mean(data)) / (np.std(data))

    if Normalize:
        data = (data - np.min(data)) / (np.max(data) - np.min(data))

    data = data.reshape((1000, -1))

    shape = np.shape(data)[1]

    sum_ = np.zeros((shape,))
    for i in range(1000):
        sum_ = sum_ + data[i]
    layer_average = sum_ / 1000
    psnr_layer = np.zeros((1000,))
    for i in range(1000):
        # 可选，采用官方实现的psnr或者自己实现的psnr计算法
        # psnr_layer[i] = psnr_cal(layer_average, data[i])
        psnr_layer[i] = psnr(layer_average, data[i])
    return psnr_layer.mean()


# 计算数据的SSIM
def cal_ssim(sequence1, sequence2, shape):
    # sequence 是指输入向量，比如，第一层输出为(64, 27, 27),，shape=(64, 27, 27)

    # 然后计算L， C, S
    # 先计算出两个分布的均值和方差
    miu_1 = np.mean(sequence1)
    miu_2 = np.mean(sequence2)
    sigma_1 = np.sqrt(np.var(sequence1))
    sigma_2 = np.sqrt(np.var(sequence2))
    cov_12 = np.cov(sequence1, sequence2)

    K1 = 0.01
    K2 = 0.03
    L = 255

    C1 = np.square(K1 * L)
    C2 = np.square(K2 * L)
    C3 = C2 / 2

    L_XY = (2 * miu_1 * miu_2 + C1) / (np.square(miu_1) + np.square(miu_2) + C1)
    C_XY = (2 * sigma_1 * sigma_2 + C2) / (np.square(sigma_1) + np.square(sigma_2) + C2)
    S_XY = (cov_12[0][1] + C3) / (sigma_1 * sigma_2 + C3)
    return L_XY * C_XY * S_XY


# 计算序列的SSIM
def cal_seq_ssim(sequence, layer, Standard=False, Normalize=False):
    '''
    计算一个序列的ssim
    计算步骤，将每个维度的数据与所有维度的平均值做ssim，然后取平均值，得出结果。
    :param sequence: 序列，
    :param layer: 层数
    :param Standard: 标准化
    :param Normalize: 归一化
    :return: 返回序列的ssim
    '''
    if (layer <= 4):
        win_size = 5
    else:
        win_size = 3

    n = np.shape(sequence)[0]
    #     sequence = sequence.reshape((n, sequence.shape[2],-1))
    sequence = sequence.reshape((n, -1))

    shape = np.shape(sequence)[1:]

    #     print(shape)

    # 定义两开关：standard 为标准化，Normalize为归一化

    if (Standard):
        sequence = (sequence - np.mean(sequence)) / (np.var(sequence))

    if (Normalize):
        sequence = (sequence - np.min(sequence)) / (np.max(sequence) - np.min(sequence))

    # 首先计算平均值，
    sum_ = np.zeros(shape=shape, dtype=np.float32)
    for i in range(n):
        sum_ += sequence[i]
    average = sum_ / n

    # 将序列中每个值与均值做ssim比较，得出ssim向量，最后求一个均值，
    ssim_seq = np.zeros((n))
    for i in range(n):
        #         ssim_seq[i] = cal_ssim(average, sequence[i], shape=shape)
        #         print(type(average[0])), print(type(sequence[i][0]))

        ssim_seq[i] = ssim(X=average, Y=sequence[i], win_size=win_size)
    #         if (i<10):
    #             print(ssim_seq[i])
    return ssim_seq.mean()


# 关于模型的一些方法，用来获取模型中间输出或者准确度
class AverageMeter(object):
    '''
        获取平均网络精度
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    '''
    获取模型准确度，一般获取前1个和前5个的准确度。
    :param output: 输出结果
    :param target: 目标
    :param topk: 前几个准确度
    :return: 返回准确度
    '''
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def data_loader(root, batch_size=64, workers=1, mode="val", pin_memory=False):
    '''
    使用数据集生成dataloader
    :param root: 数据集的文件夹
    :param batch_size:
    :param workers:
    :param pin_memory:
    :return: 返回dataloader
    '''
    valdir = os.path.join(root, mode)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return val_loader


def save_checkpoint(state, is_best, filename='alex_checkpoint.pth'):
    '''
    保存模型的权重
    :param state: 状态，
    :param is_best:  是否是最优权重，也就是当前结果是不是最好的
    :param filename: 输出的文件名
    :return: 无返回值
    '''
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'alex_model_best_1.pth')


def adjust_learning_rate(optimizer, epoch, init_lr):
    """
    适应性学习率
    :param optimizer: 选择的优化器
    :param epoch:
    :param init_lr: 初始学习率
    :return: 无返回值
    """
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def imagenet_class_index_dic():
    with open('imagenet_class_index.txt', 'r') as f:
        index_list = f.readlines()

    index_dic = {}
    for i in range(len(index_list)):
        index_list[i] = index_list[i].strip('\n')
        index_split = index_list[i].split(' ')
        index_dic[index_split[0]] = index_split[1]
    return index_dic


# 验证js，通过node来对节点进行删除，将node转化为0、1向量，直接采用与中间结果相乘即可，
def validate_js(val_loader, model, criterion, print_freq, node, tresh, thread_num, filename, class_name, is_cuda=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    node = torch.from_numpy(node)

    model.eval()

    # # 对node进行处理
    # if tresh != 1:
    #     for i in range(shape[0]):
    #         for j in range(shape[1]):
    #             for k in range(shape[2]):
    #                 if node[i, j, k] < tresh:
    #                     node[i, j, k] = 0
    #                 else:
    #                     node[i, j, k] = 1

    end = time.time()
    for i, (input_, _) in enumerate(val_loader):

        with torch.no_grad():
            # output = model(input_)
            if is_cuda:
                node = node.cuda()
                input_ = input_.cuda()

            try:
                # compute output
                #             output = model(input)
                if model_name == "res50":
                    mid = (model.layer4(model.layer3(model.layer2(model.layer1(model.maxpool(model.relu(
                        model.bn1(model.conv1(input_))
                    )))))))
                else:
                    mid = model.features(input_)
                # print(mid.dtype, node.dtype)
                for k in range(mid.size()[0]):
                    mid[k] = mid[k] * node

            except Exception as e:
                print(e)
            # print("mid_shape: ", (mid.size()))
            if model_name == "res50":
                mid = model.avgpool(mid)
                mid = np.reshape(mid, (mid.size()[0], -1))
                output = model.fc(mid)
            else:
                mid = np.reshape(mid, (mid.size()[0], -1))
                output = model.classifier(mid)

            # loss = criterion(output, target)
            # measure accuracy and record loss

            # print(output)
            im_dict = imagenet_class_index_dic()
            target = int(im_dict[class_name])
            # print(target)
            target = torch.tensor([target for i in range(np.shape(input_)[0])])

            if is_cuda:
                target = target.cuda()

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            # losses.update(loss.item(), input_.size(0))
            top1.update(prec1[0], input_.size(0))
            top5.update(prec5[0], input_.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % print_freq == 0:
                print(str(thread_num), 'Test: [{0}/{1}]\t'
                                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                       'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time,
                    top1=top1, top5=top5))

    with open("/home/hongwu/python/Image/" + experience_id + "/" + filename + ".txt",
              "a+") as f:
        f.write(
            'save ' + str(
                thread_num) + ", percent: " + str(
                node.mean()) + ' :   * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} \n'.format(
                top1=top1,
                top5=top5))

    # return top1.avg, top5.avg


# 直接验证
def validate(val_loader, model, criterion, print_freq, filename, is_cuda=False):
    '''
    使用验证集对当前模型进行评估
    :param val_loader: 使用验证集生成的dataloader
    :param model: 模型
    :param criterion: 评价标准
    :param print_freq: 打印频率
    :return: 最终的准确率
    '''
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    # a = [
    #     [0, 120, 143, 166, 189, 210, 233, 256, 279, 300, 323, 346, 369, 391, 413, 436, 459, 481, 503, 526, 549, 571,
    #      594, 616, 639, 661, 684, 706, 729, 751, 774, 797, 819, 841, 864, 887, 909, 931, 954, 977],
    #     [1, 121, 144, 167, 19, 211, 234, 257, 28, 301, 324, 347, 37, 392, 414, 437, 46, 482, 504, 527, 55, 572, 595,
    #      617, 64, 662, 685, 707, 73, 752, 775, 798, 82, 842, 865, 888, 91, 932, 955, 978],
    #     [10, 122, 145, 168, 190, 212, 235, 258, 280, 302, 325, 348, 370, 393, 415, 438, 460, 483, 505, 528, 550, 573,
    #      596, 618, 640, 663, 686, 708, 730, 753, 776, 799, 820, 843, 866, 889, 910, 933, 956, 979],
    #     [100, 123, 146, 169, 191, 213, 236, 259, 281, 303, 326, 349, 371, 394, 416, 439, 461, 484, 506, 529, 551, 574,
    #      597, 619, 641, 664, 687, 709, 731, 754, 777, 8, 821, 844, 867, 89, 911, 934, 957, 98],
    #     [101, 124, 147, 17, 192, 214, 237, 26, 282, 304, 327, 35, 372, 395, 417, 44, 462, 485, 507, 53, 552, 575, 598,
    #      62, 642, 665, 688, 71, 732, 755, 778, 80, 822, 845, 868, 890, 912, 935, 958, 980],
    #     [102, 125, 148, 170, 193, 215, 238, 260, 283, 305, 328, 350, 373, 396, 418, 440, 463, 486, 508, 530, 553, 576,
    #      599, 620, 643, 666, 689, 710, 733, 756, 779, 800, 823, 846, 869, 891, 913, 936, 959, 981],
    #     [103, 126, 149, 171, 194, 216, 239, 261, 284, 306, 329, 351, 374, 397, 419, 441, 464, 487, 509, 531, 554, 577,
    #      6, 621, 644, 667, 69, 711, 734, 757, 78, 801, 824, 847, 87, 892, 914, 937, 96, 982],
    #     [104, 127, 15, 172, 195, 217, 24, 262, 285, 307, 33, 352, 375, 398, 42, 442, 465, 488, 51, 532, 555, 578, 60,
    #      622, 645, 668, 690, 712, 735, 758, 780, 802, 825, 848, 870, 893, 915, 938, 960, 983],
    #     [105, 128, 150, 173, 196, 218, 240, 263, 286, 308, 330, 353, 376, 399, 420, 443, 466, 489, 510, 533, 556, 579,
    #      600, 623, 646, 669, 691, 713, 736, 759, 781, 803, 826, 849, 871, 894, 916, 939, 961, 984],
    #     [106, 129, 151, 174, 197, 219, 241, 264, 287, 309, 331, 354, 377, 4, 421, 444, 467, 49, 511, 534, 557, 58, 601,
    #      624, 647, 67, 692, 714, 737, 76, 782, 804, 827, 85, 872, 895, 917, 94, 962, 985],
    #     [107, 13, 152, 175, 198, 22, 242, 265, 288, 31, 332, 355, 378, 40, 422, 445, 468, 490, 512, 535, 558, 580, 602,
    #      625, 648, 670, 693, 715, 738, 760, 783, 805, 828, 850, 873, 896, 918, 940, 963, 986],
    #     [108, 130, 153, 176, 199, 220, 243, 266, 289, 310, 333, 356, 379, 400, 423, 446, 469, 491, 513, 536, 559, 581,
    #      603, 626, 649, 671, 694, 716, 739, 761, 784, 806, 829, 851, 874, 897, 919, 941, 964, 987],
    #     [109, 131, 154, 177, 2, 221, 244, 267, 29, 311, 334, 357, 38, 401, 424, 447, 47, 492, 514, 537, 56, 582, 604,
    #      627, 65, 672, 695, 717, 74, 762, 785, 807, 83, 852, 875, 898, 92, 942, 965, 988],
    #     [11, 132, 155, 178, 20, 222, 245, 268, 290, 312, 335, 358, 380, 402, 425, 448, 470, 493, 515, 538, 560, 583,
    #      605, 628, 650, 673, 696, 718, 740, 763, 786, 808, 830, 853, 876, 899, 920, 943, 966, 989],
    #     [110, 133, 156, 179, 200, 223, 246, 269, 291, 313, 336, 359, 381, 403, 426, 449, 471, 494, 516, 539, 561, 584,
    #      606, 629, 651, 674, 697, 719, 741, 764, 787, 809, 831, 854, 877, 9, 921, 944, 967, 99],
    #     [111, 134, 157, 18, 201, 224, 247, 27, 292, 314, 337, 36, 382, 404, 427, 45, 472, 495, 517, 54, 562, 585, 607,
    #      63, 652, 675, 698, 72, 742, 765, 788, 81, 832, 855, 878, 90, 922, 945, 968, 990],
    #     [112, 135, 158, 180, 202, 225, 248, 270, 293, 315, 338, 360, 383, 405, 428, 450, 473, 496, 518, 540, 563, 586,
    #      608, 630, 653, 676, 699, 720, 743, 766, 789, 810, 833, 856, 879, 900, 923, 946, 969, 991],
    #     [113, 136, 159, 181, 203, 226, 249, 271, 294, 316, 339, 361, 384, 406, 429, 451, 474, 497, 519, 541, 564, 587,
    #      609, 631, 654, 677, 7, 721, 744, 767, 79, 811, 834, 857, 88, 901, 924, 947, 97, 992],
    #     [114, 137, 16, 182, 204, 227, 25, 272, 295, 317, 34, 362, 385, 407, 43, 452, 475, 498, 52, 542, 565, 588, 61,
    #      632, 655, 678, 70, 722, 745, 768, 790, 812, 835, 858, 880, 902, 925, 948, 970, 993],
    #     [115, 138, 160, 183, 205, 228, 250, 273, 296, 318, 340, 363, 386, 408, 430, 453, 476, 499, 520, 543, 566, 589,
    #      610, 633, 656, 679, 700, 723, 746, 769, 791, 813, 836, 859, 881, 903, 926, 949, 971, 994],
    #     [116, 139, 161, 184, 206, 229, 251, 274, 297, 319, 341, 364, 387, 409, 431, 454, 477, 5, 521, 544, 567, 59,
    #      611, 634, 657, 68, 701, 724, 747, 77, 792, 814, 837, 86, 882, 904, 927, 95, 972, 995],
    #     [117, 14, 162, 185, 207, 23, 252, 275, 298, 32, 342, 365, 388, 41, 432, 455, 478, 50, 522, 545, 568, 590, 612,
    #      635, 658, 680, 702, 725, 748, 770, 793, 815, 838, 860, 883, 905, 928, 950, 973, 996],
    #     [118, 140, 163, 186, 208, 230, 253, 276, 299, 320, 343, 366, 389, 410, 433, 456, 479, 500, 523, 546, 569, 591,
    #      613, 636, 659, 681, 703, 726, 749, 771, 794, 816, 839, 861, 884, 906, 929, 951, 974, 997],
    #     [119, 141, 164, 187, 209, 231, 254, 277, 3, 321, 344, 367, 39, 411, 434, 457, 48, 501, 524, 547, 57, 592, 614,
    #      637, 66, 682, 704, 727, 75, 772, 795, 817, 84, 862, 885, 907, 93, 952, 975, 998],
    #     [12, 142, 165, 188, 21, 232, 255, 278, 30, 322, 345, 368, 390, 412, 435, 458, 480, 502, 525, 548, 570, 593,
    #      615, 638, 660, 683, 705, 728, 750, 773, 796, 818, 840, 863, 886, 908, 930, 953, 976, 999]]

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # bianhuan = np.array(a).T.reshape(-1, )

        # for s in range(np.shape(target)[0]):
        # target[s] = (torch.tensor(bianhuan[target[s]]))

        if (is_cuda):
            target = target.cuda()
            input = input.cuda()
            model = model.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        with torch.no_grad():
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    with open("/home/hongwu/python/Image/" + experience_id + "/" + filename + ".txt",
              "a+") as f:
        f.write('save 1, percent:   * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} \n'.format(
            top1=top1,
            top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def validate_class(model, criterion, print_freq, class_num, scr_dir="/home/hongwu/python/Image/val/",
                   val_dir="/home/hongwu/Image/valdate_class/val/", is_cuda=False):
    '''
    验证单个类别的准确率
    :param val_loader:
    :param model:
    :param criterion:
    :param print_freq:
    :param class_num: 表示类别的序号
    :param is_cuda:
    :return: 最终的准确率
    '''

    shutil.rmtree(val_dir)
    shutil.copytree(scr_dir + str(class_num),
                    val_dir + str(class_num))

    val_loader = data_loader(val_dir + "/../")

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    for i, (input, _) in enumerate(val_loader):
        target = torch.ones(np.shape(input)[0], dtype=torch.int64) * class_num
        if (is_cuda):
            target = target.cuda()
            input = input.cuda()
            model = model.cuda()
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg


# 直接获取第n层的中间输出
def get_n_mid(data_loader, model, layer, is_cuda=False):
    '''
    获取模型第n层的输出，此时模型直接表示为alexnet，结果直接用list
    :param data_loader:
    :param model:
    :param layer:
    :param cuda:
    :return:
    '''
    model.eval()
    geshi_1 = np.array(
        ((1, 3, 224, 224), (1, 64, 27, 27), (1, 192, 13, 13), (1, 384, 13, 13), (1, 256, 13, 13), (1, 256, 6, 6)))
    geshi_2 = np.array(((1, 4096), (1, 4096), (1, 1000)))

    if layer <= 5:
        layer_result = torch.zeros(geshi_1[layer].tolist())
    else:
        layer_result = torch.zeros(geshi_2[layer - 6].tolist())

    if (is_cuda):
        model = model.cuda()
        layer_result = layer_result.cuda()

    with torch.no_grad():
        for i, (input, _) in enumerate(data_loader):

            model_f = model.features
            # model_c = model.classifier

            input_var = torch.autograd.Variable(input)

            if is_cuda:
                input_var = input_var.cuda()

            if layer == 0:
                layer_result = torch.cat((layer_result, input_var), 0)
                continue

            cov_re_pool_1 = model_f[2](model_f[1](model_f[0](input_var)))
            if layer == 1:
                layer_result = torch.cat((layer_result, cov_re_pool_1), 0)
                continue

            cov_re_pool_2 = model_f[5](model_f[4](model_f[3](cov_re_pool_1)))
            if layer == 2:
                layer_result = torch.cat((layer_result, cov_re_pool_2), 0)
                continue

            cov_re_pool_3 = model_f[7](model_f[6](cov_re_pool_2))
            if layer == 3:
                layer_result = torch.cat((layer_result, cov_re_pool_3), 0)
                continue

            cov_re_pool_4 = model_f[9](model_f[8](cov_re_pool_3))
            if layer == 4:
                layer_result = torch.cat((layer_result, cov_re_pool_4), 0)
                continue

            cov_re_pool_5 = model_f[12](model_f[11](model_f[10](cov_re_pool_4)))
            # 添加avgpool层
            cov_re_pool_5 = model.avgpool(cov_re_pool_5)
            if layer == 5:
                layer_result = torch.cat((layer_result, cov_re_pool_5), 0)
                continue

            cov_re_pool_5 = cov_re_pool_5.view((-1, 6 * 6 * 256))

            linear_relu_1 = model.classifier[2](model.classifier[1](model.classifier[0](cov_re_pool_5)))
            if layer == 6:
                layer_result = torch.cat((layer_result, linear_relu_1), 0)
                continue

            linear_relu_2 = model.classifier[5](model.classifier[4](model.classifier[3](linear_relu_1)))
            if (layer == 7):
                layer_result = torch.cat((layer_result, linear_relu_2), 0)
                continue

            linear_relu_3 = model.classifier[6](linear_relu_2)
            if (layer == 8):
                layer_result = torch.cat((layer_result, linear_relu_3), 0)

    if layer <= 5:
        layer_result = layer_result[1:, :, :, :]
    else:
        layer_result = layer_result[1:, :]

    return layer_result.cpu().detach().numpy()


def get_midresult(data_loader, model, augnum, shape, batch_size, is_cuda=False):
    result = np.zeros(np.concatenate(([augnum], shape)))
    model = model.eval()
    if is_cuda:
        model = model.cuda()
    for i, (input_, _) in enumerate(data_loader):
        if is_cuda:
            input_ = input_.cuda()
        if model_name != "res50":
            data = model.features(input_).cpu().detach().numpy()
        else:
            data = (model.layer4(model.layer3(model.layer2(model.layer1(model.maxpool(model.relu(
                model.bn1(model.conv1(input_))
            ))))))).cpu().detach().numpy()
        for j in range(np.shape(input_)[0]):
            result[i * batch_size + j] = data[j]
    return result


# 从x到y层的过程
def from_layer_x_to_y(layer_x, x, model):
    # 将x层的输出输入到x+1层得到x+1层的输出
    model = model.cpu()

    model_f = model.features
    model_c = model.classifier

    if x == 0:
        layer_y = model_f[2](model_f[1](model_f[0](torch.Tensor(layer_x))))
    elif x == 1:
        layer_y = model_f[5](model_f[4](model_f[3](torch.Tensor(layer_x))))
    elif x == 2:
        layer_y = (model_f[7](model_f[6](torch.Tensor(layer_x))))
    elif x == 3:
        layer_y = (model_f[9](model_f[8](torch.Tensor(layer_x))))
    elif x == 4:
        layer_y = model_f[12](model_f[11](model_f[10](torch.Tensor(layer_x))))
    elif x == 5:
        n = np.shape(layer_x)[0]
        layer_x = layer_x.reshape(n, -1)
        layer_y = (model_c[2](model_c[1](torch.Tensor(layer_x))))
    elif x == 6:
        layer_y = model_c[5](model_c[4](model_c[3](torch.Tensor(layer_x))))
    else:
        layer_y = (model_c[6](torch.Tensor(layer_x)))

    return layer_y


# 从上一层直接经过标准化之后输入到下一层
def before_to_after(layer_result, x, model):
    '''
    将单层输出输入进行bn之后输出到下一层，并返回下一层的结果
    :param layer_result: 单层输出
    :param x: 层数
    :param model: 模型
    :return: 模型最终输出和经过bn后输出到下一层之后的结果
    '''

    # 将第0层结果进行标准化
    #     layer_result = nn.BatchNorm2d(torch.from_array(layer_0_result))
    # 按照通道进行标准化，第二个维度为通道
    if (np.ndim(layer_result) == 4):
        for i in range(np.shape(layer_result)[1]):
            layer_result[:, i, :, :] = (layer_result[:, i, :, :] - np.mean(layer_result[:, i, :, :])) / np.std(
                layer_result[:, i, :, :])
    else:
        layer_result = (layer_result - np.mean(layer_result)) / np.std(layer_result)

    layer_next = from_layer_x_to_y(layer_result, x, model)

    result_trans_x = from_layer_x_to_y(layer_x=layer_result, x=x, model=model)

    for i in range(x + 1, 8):
        result_trans_x = from_layer_x_to_y(result_trans_x, i, model)

    return [result_trans_x, layer_next]

    # np.save(trans_dir + "result_trans_" + str(x) + ".npy", result_trans_x.detach().cpu().numpy())
    #
    # np.save(trans_dir + "layer_" + str(x + 1) + ".npy", layer_next.detach().cpu().numpy())


# 检查数据的偏序关系
# - 考虑一下步骤
#     1. 首先得到三个类别的数据 (1, 100, 200)
#     2. 将其输入网络后得出中间层输出，例如输出4096层为ndarray为[3, 4096]
#     3. 然后考虑单个节点的偏序关系，查看基层之间时候改变，并与最终层比较，因为最终分类层的偏序关系才是正确的。


# 构建数据集文件夹
# train_dir = "/home/hongwu/python/Image/tiny_image/tiny/train/"


# 构建训练数据集
def make_train_dir(train_dir, class_num, image_num):
    sample_dir = sample(os.listdir(train_dir), class_num)
    others = []
    one = []
    for i in range(class_num):
        if (i == 0):
            img = sample(os.listdir(train_dir + sample_dir[i] + '/images'), image_num)
            for j in range(image_num):
                img[j] = train_dir + sample_dir[i] + '/images/' + img[j]
            one.extend(img)
        else:
            img = sample(os.listdir(train_dir + sample_dir[i] + '/images'), image_num)
            for j in range(image_num):
                img[j] = train_dir + sample_dir[i] + '/images/' + img[j]
            others.append(img)  # others.extend(img) 直接将很多类合并在一起，而这里我们不合并，之后再合并
    one_imglist = []
    others_imglist = []

    for img in one:
        one_imglist.append(cv2.imread(img))

    for link in others:
        img_list = []
        for img in link:
            img_list.append(cv2.imread(img))
        others_imglist.append(img_list)
    return [one_imglist, others_imglist]


# 构建训练数据集
def make_dir(train_dir, filename, class_num, image_num, save_dir):
    """
    直接构建class_num类的数据，平均每个类选取image_num个图片
    :param save_dir:
    :param train_dir: 训练文件夹
    :param class_num: 文件数量
    :param image_num: 图片数量
    :return:
    """
    # sample_dir = sample(os.listdir(train_dir), class_num)
    # global file
    data = []

    # print(sample_dir)

    # for _, _, files in os.walk(train_dir + "/n02088364/"):
    #     file = files

    for i in range(class_num):
        img = sample(os.listdir(train_dir + filename), image_num)
        # img = os.listdir(train_dir + filename)
        # print(img)
        for j in range(image_num):
            img[j] = train_dir + filename + "/" + img[j]
            # img[j] = train_dir + sample_dir[i] + '/' + img[j]
        # print(img)

        data.append(img)

    data_imglist = []

    for cls in data:
        data_img = []
        for img in cls:
            data_img.append(cv2.imread(img))
        data_imglist.append(data_img)

    # print(np.shape(data_imglist))
    # print(data_imglist)
    print("构建数据完成！！")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(np.shape(data_imglist)[0]):
        for j in range(np.shape(data_imglist)[1]):
            # np.save("/home/hongwu/python/Image/result/data_imglist_"+str(i)+".npy", data_imglist[i])
            np.save(save_dir + "/data_imglist_" + str(j) + ".npy", data_imglist[i][j])

    print("写入data_imglist完毕！")

    # return data_imglist


# 测试两个js散度的区别，按照js的大小关系进行区别。
def test_dif(js_old, js_new):
    # np.reshape(js_old, [-1, ])
    # np.reshape(js_new, [-1, ])
    # count = 0.0
    # for i in range(np.shape(js_old)[0]):
    #     # if float(js_old[i])==0:
    #     #     continue
    #     for j in range(np.shape(js_new)[0]):
    #         # if (float(js_old[j])==0):
    #         #     continue
    #         if js_new[i]-js_new[j]>=0.01 and (float(js_old[i]) - float(js_old[j]))>0.01 or (js_new[i] - js_new[j] < -0.01 and float(js_old[i]) - float(js_old[j]) < -0.01):
    #             print(js_new[i], js_new[j])
    #             print(js_old[i], js_old[j])
    #             count+=1
    # return count/(np.shape(js_new)[0]*(np.shape(js_old)[0])/2)

    old_sort = np.argsort(js_old)
    new_sort = np.argsort(js_new)
    return np.sqrt(mean_squared_error(new_sort, old_sort))  # 当随机选择一个验证集计算第五层结果是，两个差别的RMSE=407


# 
def js_criterion():
    parse = argparse.ArgumentParser()
    parse.add_argument("--learning_rate", type=float, default=0.01, help="initial learning rate")
    parse.add_argument("--test_dir", default="/home/hongwu/python/Image/data_test/", help="initial test dir")
    parse.add_argument("--train_dir", default="/home/hongwu/python/Image/tiny_image/tiny/train/",
                       help="initial train dir")
    parse.add_argument("--class_num", default=10, help="test class num")
    parse.add_argument("--image_num", default=50, help="test images num")
    parse.add_argument("--aug_num", default=1000, help="test augment num")
    parse.add_argument("--val_dir", default="/home/hongwu/python/Image", help="validate dir")
    parse.add_argument("--print_freq", default=100, help="print frequency")

    args, unparsed = parse.parse_known_args()
    data_dir = args.test_dir
    train_dir = args.train_dir
    class_num = args.class_num
    image_num = args.image_num
    aug_num = args.aug_num
    print_freq = args.print_freq
    val_dir = args.val_dir

    # 首先删除之前建立的数据目录
    # if os.listdir(data_dir):
    #     shutil.rmtree(data_dir)
    #     os.mkdir(data_dir)

    print("开始添加数据干扰！")
    one_list, others_list = make_train_dir(train_dir, class_num=class_num, image_num=image_num)
    print("干扰添加完成！")
    one = one_list
    others = others_list
    # 得出结果为onelist shape: (50, 1000, 64, 64, 3)
    # otherslist shape : (9, 50, 64, 64, 3)
    # 首先将其数据输出为图片，然后制作dataloader
    # 首先需要设置目录
    if not Path(data_dir):
        os.makedirs(path=data_dir)
    # 输出one和others数据
    # 目录结构为 -datadir  -one/others  -i  -val -class -images
    # 首先输出one

    # 进行10词交换对比，保留的是需要删除点的坐标。如果是第五层则维度为[256, 6, 6]
    del_point = []

    print("开始计算需要删除的点！")
    for t in range(10):
        print("第", t + 1, "次计算！")
        # 每次都要对数据做一次恢复
        one_list = one
        others_list = others

        # 并且删除之前建立的数据目录
        if os.listdir(data_dir):
            shutil.rmtree(data_dir)
            os.mkdir(data_dir)

        if t != 0:
            # 进行交换
            one_list, others_list[t - 1] = others_list[t - 1], one_list

        one_list = generate_coinT(one_list, aug_num=aug_num)
        for i in range(np.shape(one_list)[0]):
            os.makedirs(data_dir + '/one/' + str(i) + '/val/class/')
            for j in range(np.shape(one_list)[1]):
                cv2.imwrite(data_dir + '/one/' + str(i) + '/val/class/aug_' + str(j) + '.jpg', one_list[i][j])
        # 然后输出others
        os.makedirs(data_dir + '/others/val/class/')
        k = 0
        for i in range(np.shape(others_list)[0]):
            for j in range(np.shape(others_list)[1]):
                cv2.imwrite(data_dir + '/others/val/class/' + str(k) + '.jpg', others_list[i][j])
                k += 1

        print('文件结构梳理完成')

        # 构建data_loader
        # one_data_loader
        one_data_loader = []
        for i in range(image_num):
            one_data_loader.append(data_loader(data_dir + "/one/" + str(i)))
        others_data_loader = data_loader(data_dir + "/others/")

        print("构建dataloader完成 ----- ")

        model = torchvision.models.alexnet(pretrained=True)
        one_mid5_result = []
        for i in range(image_num):
            one_mid5_result.append(get_n_mid(one_data_loader[i], model=model, layer=5, is_cuda=False))
        others_mid5_result = get_n_mid(others_data_loader, model=model, layer=5, is_cuda=False)

        print("中间输出获取完成------ ")

        js = np.zeros([50, 256, 6, 6])
        for i in range(50):
            for j in range(256):
                for k in range(6):
                    for r in range(6):
                        js[i][j][k][r] = JS_divergence_1(one_mid5_result[i][:, j, k, r], others_mid5_result[:, j, k, r],
                                                         del_zero=True)

        print("JS 计算完毕")

        # 根据js方差来进行筛选

        jstd_tresh = np.percentile(js.std(axis=0), 75)
        del_p = np.zeros([256, 6, 6])
        for i in range(256):
            for j in range(6):
                for k in range(6):
                    if (js[:, i, j, k].std() > jstd_tresh):
                        del_p[i][j][k] = -1

        del_point.append(del_p)

        # val_loader = data_loader(val_dir)
        # criterion = torch.nn.CrossEntropyLoss().cuda()
        # validate_js(val_loader, criterion=criterion, js=js, model=model, print_freq=100,
        #             percent=75, is_cuda=True)

        # print("模型评估完成，可与原始模型对比")
    return del_point


def test_js_zero():
    # 首先得到一些中间输出
    for i in range(4):
        loader = data_loader("/home/hongwu/python/Image/test_stand/one/" + str(i + 2) + "/")
        data = get_n_mid(data_loader=loader, model=torchvision.models.alexnet(pretrained=True), layer=5, is_cuda=False)
        print(np.shape(data))

        js_old = kl_Bin_cal(data)
        js_new = kl_Bin_cal(data, del_zero=True)

        print(test_dif(js_old, js_new))


from random import randint


# 使用Mypool重写，让子线程也可以创建新的子线程
class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def sleepawhile(t):
    print("Sleeping %i seconds..." % t)
    time.sleep(t)
    return t


# 方法重写过了，弃用
def cal_mid(data_dir="/home/hongwu/python/Image/mid_result/", aug_num=1000):
    """
    这里是想得到每个类别进行相同转换之后的中间输出，并且不经过转化之后的中间输出，
    :param:  输入为数据文件名称，将其设计为文件夹，并且输入网络进行输出。shape: [class_num, image_num]
    :return: 结果定义为两个ndarray:
                    经过干扰之后的输出: aug_res: [class_num, image_num, aug_num, 256, 6, 6]
                    未经干扰之后的输出：ori_res: [class_num, image_num, 256, 6, 6]
    """
    # 构建数据存储变量
    class_num = 10  # np.shape(data_imglist)[0]
    image_num = 50  # np.shape(data_imglist)[1]
    # ori_res = np.zeros((class_num, image_num, 256, 6, 6))
    #
    # model = torchvision.models.alexnet(pretrained=True)
    #
    # # # 设计文件目录
    # shutil.rmtree(data_dir+"/aug/")
    #
    # # 将对应分类的图片输出到对应文件夹, 当文件存在时，下面代码可以不需要
    # # for i in range(class_num):
    # #     os.makedirs(data_dir + str(i) + "/val/class/")
    # # for i in range(class_num):
    # #     for j in range(image_num):
    # #         cv2.imwrite(data_dir + str(i) + '/val/class/' + str(j) + '.jpg', data_imglist[i][j])
    #
    # # print("原始数据输出至文件夹完毕！")
    #
    #
    # # 下面进行干扰数据生成，并得到中间结果
    # # 直接读取数据，然后干扰之后再输出到文件， 当文件存在时，下面代码可以注释掉
    # for i in range(class_num):
    #     for j in range(image_num):
    #         os.makedirs(data_dir + "/aug/" + str(i) + "/" + str(j) + "/val/class/")
    #
    # # for i in range(class_num): # （0-9）
    # #     print("干扰数据写入类序：" , i)
    # #     aug_img = generate_coinT(data_imglist[i], aug_num=aug_num)
    #
    #     # for j in range(image_num): # （0-49）
    #     #     for k in range(aug_num):
    #     #         cv2.imwrite(data_dir + '/aug/' + str(i) + '/' + str(j) + '/val/class/'+str(k)+'.jpg', aug_img[j][k])
    #
    # #
    # # for i in range(10):
    # #     args = [aug_num, i]
    # #     generate_coinT(args)
    #
    #
    #
    # print("多线程实现添加干扰！！")
    # #
    # # # 多线程实现
    # pool = Pool(processes=10)
    # # args = zip([aug_num for i in range(class_num)], [thread_num for thread_num in range(class_num)])
    # # pool.map(generate_coinT, args)
    #
    # for index in range(10):
    #     pool.apply_async(func=generate_coinT, args=((aug_num, index), ))
    #
    # pool.close()
    # pool.join()
    #
    #
    # print("干扰数据写入文件完毕！")

    # 制作原始数据的data_loader, 数目为num_class
    # loaders = []
    # for i in range(class_num):
    #     loaders.append(data_loader(data_dir + str(i)))
    #
    # # 然后根据loaders经过网络得到中间输出。
    # for i in range(class_num):
    #     ori_res[i] = get_n_mid(loaders[i], model=model, layer=5)
    #
    # np.save("/home/hongwu/python/Image/result/origin_res.npy", ori_res)

    # print("原始网络结果获取完毕")
    # 多线程实现
    pool_1 = MyPool(10)
    for index in range(10):
        pool_1.apply_async(func=augMidres, args=(index,))
    # pool_1.map_async(augMidres, range(10))
    pool_1.close()
    pool_1.join()


def mult_process(function, aug_num):
    print("多线程实现添加干扰！！")

    # 多线程实现
    pool = Pool(processes=10)
    # args = zip([data_imglist[i] for i in range(class_num)], [aug_num for i in range(class_num)], [thread_num for thread_num in range(class_num)])
    # pool.map(generate_coinT, args)

    for i in range(10):
        pool.apply_async(func=function, args=((aug_num, i),))

    pool.close()
    pool.join()


# 获取添加干扰后的数据的中间结果
def augMidres(index, model, augnum, shape, data_dir="/home/hongwu/python/Image/exper_v1/marker_C/", is_cuda=False):
    # 构建loaders

    global aug_res, layer_result
    try:

        print("augmid计算：", index)
        batch_size = 10
        data_load = data_loader(data_dir + str(index), batch_size=batch_size)

        # layer_result = torch.zeros((1, 256, 6, 6))
        # with torch.no_grad():
        #     for i, (input_, _) in enumerate(data_load):
        #         if (i % 3):
        #             print(index, i)
        #         input_var = torch.autograd.Variable(input_)
        #         output = model.features(input_var)
        #         # print(np.shape(output))
        #         layer_result = torch.cat((layer_result, output), 0)
        #     # print(np.shape(layer_result))
        # layer_result = (layer_result[1:, :, :, :].cpu().detach().numpy())

        layer_result = get_midresult(data_loader=data_load, model=model, batch_size=batch_size, augnum=augnum,
                                     shape=shape, is_cuda=is_cuda)

    except Exception as e:
        print(e)
    # assert np.shape(layer_result) == (1000, 256, 6, 6)
    if not os.path.exists(data_dir + "/result/"):
        os.makedirs(data_dir + "/result/")

    np.save(data_dir + "/result/mid_result_" + str(index) + ".npy", layer_result)

    print(index, "线程干扰数据结果计算并存储完毕")


# 获取干扰数据的中间输出与原始数据中间输出的js散度
def test_stand(z):
    """
    本函数作用在于将每个aug_data的数据与原始数据进行对比，并且每个类得出image_num个js散度
    在输入函数之前要将数据进行合并，取出单个类干扰数据和其他数据综合进行对比。
    :param z:
    :param thread_num:
    :param aug_data: 干扰之后数据的中间输出 shape: [image_num, aug_num, 256, 6, 6]
    :param origin_data: 原始数据中间输出 shape: [class_num*image_num, 256, 6, 6]
    :return: js散度结果， shape: [256, 6, 6, image_num]
    """
    thread_num = z[0]
    origin_data = z[1]

    print("这是第", thread_num, "条进程！！")
    aug_data = np.load("/home/hongwu/python/Image/result/aug_res_" + str(thread_num) + ".npy")

    image_num = np.shape(aug_data)[0]

    js = np.zeros((256, 6, 6, image_num))

    print("开始计算kl散度: 线程" + str(thread_num))
    for i in range(256):

        if i % 32 == 0:
            print("进程", thread_num, "进度为： ", i / 256)

        for j in range(6):
            for k in range(6):
                for s in range(image_num):
                    js[i][j][k][s] = JS_divergence_1(aug_data[s, :, i, j, k], origin_data[i, j, k])
    np.save("/home/hongwu/python/Image/result/js_" + str(thread_num) + ".npy", js)


def duoxiancheng():
    # print("开始读取文件")
    # aug_data = np.load("/home/hongwu/python/Image/aug_data.npy")
    # ori_data = np.load("/home/hongwu/python/Image/ori_data.npy")
    # print("读取文件完毕")

    ori_data = np.load("/home/hongwu/python/Image/result/origin_res.npy")

    class_num = np.shape(ori_data)[0]
    image_num = np.shape(ori_data)[1]

    origin_data = np.zeros((class_num * image_num, 256, 6, 6))
    for i in range(class_num):
        for j in range(image_num):
            origin_data[i * image_num + j] = ori_data[i][j]

    print("处理数据完成！")

    pool = Pool(processes=10)
    args = zip([i for i in range(10)], [origin_data for i in range(10)])
    pool.map(test_stand, args)
    pool.close()
    pool.join()
    print("结束")


def getNode(thread_num):
    js_i = np.load("/home/hongwu/python/Image/result/js_" + str(thread_num) + ".npy")
    # print(np.shape(js_i))
    js_tresh = np.percentile(np.std(js_i, axis=3), 50)
    # print(js_tresh)
    node = np.zeros((256, 6, 6))
    for i in range(256):
        for j in range(6):
            for k in range(6):
                if np.std(js_i[i, j, k, :]) > js_tresh:
                    node[i, j, k] = 1

    return node


def get_node():
    nodes = []
    for i in range(10):
        nodes.append(getNode(i))
    np.save("/home/hongwu/python/Image/result/nodes.npy", nodes)


# # 测试多线程
# def thread(url):
#     r = requests.get(url, headers=None, stream=True, timeout=30)
#     # print(r.status_code, r.headers)
#     headers = {}
#     all_thread = 1
#     # 获取视频大小
#     file_size = int(r.headers['content-length'])
#     # 如果获取到文件大小，创建一个和需要下载文件一样大小的文件
#     if file_size:
#         fp = open('2012train.tar', 'wb')
#         fp.truncate(file_size)
#         print('文件大小：' + str(int(file_size / 1024 / 1024)) + "MB")
#         fp.close()
#     # 每个线程每次下载大小为5M
#     size = 5242880
#
#     # 当前文件大小需大于5M
#     if file_size > size:
#         # 获取总线程数
#         all_thread = int(file_size / size)
#         # 设最大线程数为10，如总线程数大于10
#         # 线程数为10
#         if all_thread > 10:
#             all_thread = 10
#     part = file_size // all_thread
#     threads = []
#     starttime = datetime.datetime.now().replace(microsecond=0)
#     for i in range(all_thread):
#         # 获取每个线程开始时的文件位置
#         start = part * i
#         # 获取每个文件结束位置
#         if i == all_thread - 1:
#             end = file_size
#         else:
#             end = start + part
#         if i > 0:
#             start += 1
#         headers = headers.copy()
#         headers['Range'] = "bytes=%s-%s" % (start, end)
#         t = threading.Thread(target=Handler, name='th-' + str(i),
#                              kwargs={'start': start, 'end': end, 'url': url, 'filename': '2012train.tar',
#                                      'headers': headers})
#         t.setDaemon(True)
#         threads.append(t)
#     # 线程开始
#     for t in threads:
#         time.sleep(0.2)
#         t.start()
#     # 等待所有线程结束
#     for t in threads:
#         t.join()
#     endtime = datetime.datetime.now().replace(microsecond=0)
#     print('用时：%s' % (endtime - starttime))
#
# def Handler(start, end, url, filename, headers={}):
#     tt_name = threading.current_thread().getName()
#     print(tt_name + ' is begin')
#     r = requests.get(url, headers=headers, stream=True)
#     total_size = end - start
#     downsize = 0
#     startTime = time.time()
#     with open(filename, 'r+b') as fp:
#         fp.seek(start)
#         var = fp.tell()
#         for chunk in r.iter_content(204800):
#             if chunk:
#                 fp.write(chunk)
#                 downsize += len(chunk)
#                 line = tt_name + '-downloading %d KB/s - %.2f MB， 共 %.2f MB'
#                 line = line % (
#                     downsize / 1024 / (time.time() - startTime), downsize / 1024 / 1024,
#                     total_size / 1024 / 1024)
#                 print(line, end='\r')



# 通过node修改网络并进行网络性能评估
def critBynodes(val_loader, model, node, tresh, is_cuda=False):
    '''
    使用验证集对当前模型进行评估
    :param val_loader: 使用验证集生成的dataloader
    :param model: 模型
    :param criterion: 评价标准
    :param print_freq: 打印频率
    :return: 最终的准确率
    '''

    model.eval()
    model_f = model.features
    model_c = model.classifier

    output_ = []
    for i, (input, _) in enumerate(val_loader):

        if (is_cuda):
            input = input.cuda()
            model = model.cuda()
            model_c = model_c.cuda()
            model_f = model_f.cuda()

        with torch.no_grad():
            # compute output
            mid = model_f(input)
            for k in range(mid.size()[0]):
                for r in range(256):
                    for s in range(6):
                        for t in range(6):
                            if node[:, r, s, t].std() >= tresh:
                                mid[k][r][s][t] = 0

            mid = mid.reshape(mid.size()[0], -1)
            output = model_c(mid)
            output_.extend(output)

    target = np.argmax(output, axis=1)
    print(target)
    from scipy import stats
    zhongshu = (stats.mode(target)[0][0])
    count = 0
    for i in range(50):
        if target[i] == zhongshu:
            count += 1

    # print(count/50)

    return count / 50


# 直接进行
def get_align():
    # for i in range(100):
    #     os.makedirs("/home/hongwu/python/Image/class_100/"+str(i)+"/val/class/")

    # make_dir(train_dir="/home/hongwu/python/Image/train/train/", class_num=100, image_num=1)
    # os.makedirs("/home/hongwu/python/Image/class_100/val/class")

    pool = MyPool(100)
    for thread_num in range(100):
        pool.apply_async(generate_coinT, args=((1000, thread_num),))
    pool.close()
    pool.join()


# 批处理
def save_Snode(thread_num_1, thread_num, std_percent):
    try:
        print("进程序号：", thread_num_1, "---", thread_num)
        js = np.load("/home/hongwu/python/Image/result/js_" + str(thread_num) + ".npy", allow_pickle=True)
        js_std = np.std(js, axis=3)
        # print(np.shape(js_std))
        js_tresh = np.percentile(js_std, std_percent)
        node = np.ones((256, 6, 6))
        for i in range(256):
            for j in range(6):
                for k in range(6):
                    if js_std[i][j][k] > js_tresh:
                        node[i][j][k] = 0
        np.save("/home/hongwu/python/Image/result/" + str(thread_num_1) + "_node_" + str(thread_num) + ".npy", node)
        print(np.shape(js))
    except Exception as e:
        print(e)


def save_node(thread_num_1, std_percent):
    #
    try:
        # print("外部线程：", thread_num_1)
        # pool = MyPool(10)
        # for thread_num in range(10):
        #     pool.apply_async(func=save_Snode, args=(thread_num_1, thread_num, std_percent,))
        # pool.close()
        # pool.join()

        node = np.zeros((256, 6, 6))
        for i in range(10):
            node += (np.load("/home/hongwu/python/Image/result/" + str(thread_num_1) + "_node_" + str(i) + ".npy",
                             allow_pickle=True))

        np.save("/home/hongwu/python/Image/result/criterion_" + str(std_percent) + ".npy", node)
    except Exception as e:
        print(e)


def vali_js(thread_num):
    try:
        print(thread_num, "线程启动！！！")

        model = torchvision.models.alexnet(pretrained=True)
        criterion = torch.nn.CrossEntropyLoss()
        print_freq = 100
        train_dataloader = data_loader(root="/home/hongwu/python/Image/train/train", mode="val", pin_memory=True)
        node = np.load("/home/hongwu/python/Image/result/criterion_" + str(30 + 5 * (thread_num % 10)) + ".npy")
        validate_js(val_loader=train_dataloader, model=model, criterion=criterion,
                    print_freq=print_freq, node=node, tresh=int(thread_num / 10 + 1), thread_num=thread_num,
                    is_cuda=True)

        print(thread_num, "线程写入结果完毕!")
    except Exception as e:
        print(e)


def class_100_mid(index):
    # 构建loaders

    global aug_res, layer_result
    try:

        print("计算类序：", index)
        data_dir = "/home/hongwu/python/Image/class_100/"
        model = torchvision.models.alexnet(pretrained=True)
        loader = (data_loader(data_dir + str(index)))

        layer_result = torch.zeros((1, 256, 6, 6))
        with torch.no_grad():
            for i, (input_, _) in enumerate(loader):
                if (i == 0):
                    print(index, "----开始计算----")
                input_var = torch.autograd.Variable(input_)
                output = model.features(input_var)
                # print(np.shape(output))
                layer_result = torch.cat((layer_result, output), 0)
            # print(np.shape(layer_result))

        np.save("/home/hongwu/python/Image/class_100/result/class_" + str(index) + "_res.npy",
                layer_result[1:, :, :, :])

        print(index, "数据结果计算并存储完毕")

    except Exception as e:
        print(e)


# 获取100类的中间结果
def get_100_mid():
    pool = MyPool(100)
    for thread_num in range(100):
        pool.apply_async(class_100_mid, args=(thread_num,))
    pool.close()
    pool.join()


def doub_son(thread_num, data, mean):
    print("线程启动：", thread_num)

    try:
        js = np.zeros((1000, 6, 6))
        # print(np.shape(data))
        # print(np.shape(mean))
        for i in range(6):
            print("线程", thread_num, "计算：", i)
            for j in range(6):
                for k in range(np.shape(data)[0]):
                    # print(np.shape(data[k, :, i, j]), np.shape(mean[:, i, j]))
                    js[k][i][j] = JS_divergence_1(data[k, :, i, j], mean[:, i, j])
        np.save("/home/hongwu/python/Image/diff_class_2/result/gauss_ind/js_" + str(thread_num) + ".npy", js)
        print("线程完成：", thread_num)
    except Exception as e:
        print(e)


def cal_ttest(thread_num):
    print("ttest 线程：", thread_num)

    reason = [i + 1 for i in range(1000)]
    js = np.load("/home/hongwu/python/Image/class_100/result/gauss_ind/js_" + str(thread_num) + ".npy")
    # test_ind_ = np.zeros((6, 6, 2))
    coint_result = np.zeros((6, 6))

    for j in range(6):
        print("ttest线程", thread_num, "计算：", j)
        for k in range(6):
            # test_ind_[j][k] = scipy.stats.ttest_ind(reason, js[:, j, k], equal_var=False)

            # a_price_diff = np.diff(reason)
            # b_price_diff = np.diff(js[:, j, k])
            temp = coint(reason, js[:, j, k])
            print(temp)
            if temp[1] < 0.05:
                coint_result[j][k] = 1

    np.save("/home/hongwu/python/Image/class_100/result/gauss_ind/Gauss_blur_coint_" + str(thread_num) + ".npy",
            coint_result)

    print("ttest 计算完毕：", thread_num)


# 100类数据获取的结果应该是100个文件，每个维度为(1000, 256, 6, 6)
def doubleSamT():
    # 构建长度为1000的js序列， 然后对这个序列和1-1000的序列做双样本t检验

    # result = np.zeros((100, 1000, 256, 6, 6))
    # for i in range(100):
    #     result[i] = np.load("/home/hongwu/python/Image/class_100/result/class_" + str(i) + "_res.npy")
    # result = np.swapaxes(result, 0, 1)
    # # print(np.shape(result))
    # mean = np.mean(result, axis=0)
    # print(np.shape(mean))
    #
    # pool = MyPool(256)
    #
    # for i in range(256):
    #     pool.apply_async(doub_son, args=(i, result[:, :, i, :, :], mean[:, i, :, :],))
    # pool.close()
    # pool.join()

    # 对每个点都查看是否相关，使用双样本t检验, 或者协整

    pool_1 = MyPool(1)
    for i in range(1):
        pool_1.apply_async(cal_ttest, args=(i,))
    pool_1.close()
    pool_1.join()


# 将结果进行保存
def save_result():
    with open("/home/hongwu/python/Image/marker_C/result/one_class_result_1.txt", "a+") as savef:
        for i in range(9):
            with open("/home/hongwu/python/Image/marker_C/result/one_class_result_" + str(i * 5 + 30) + ".txt",
                      "r") as sourcef:
                lines = sourcef.readlines()
                for line in lines:
                    savef.write(line)
                savef.write("\n")
                print(i)


def get_covari():
    ttest_ind = np.zeros((256, 6, 6, 2))
    for i in range(256):
        ttest_ind[i] = np.load(
            "/home/hongwu/python/Image/class_100/result/gauss_ind/Gauss_blur_ttest_ind" + str(i) + ".npy")
    result = np.zeros((256, 6, 6))
    for i in range(256):
        for j in range(6):
            for k in range(6):
                if ttest_ind[i][j][k][1] > 0.05:
                    result[i][j][k] = 1
    np.save("/home/hongwu/python/Image/class_100/result/p_value_cri.npy", result)


def get_final():
    final = np.zeros((256, 6, 6))
    for i in range(256):
        final[i] = np.load("/home/hongwu/python/Image/class_100/result/gauss_ind/Gauss_blur_coint_" + str(i) + ".npy")
    for i in range(256):
        np.savetxt("/home/hongwu/python/Image/class_100/result/final_coint" + str(i) + ".txt", final[i])
    save_result()


# 计算序列的js， 与之前的方法功能相同。不过改写为适用于多线程
def cal_2_js(result, thread_num):
    print(np.shape(result))
    try:
        print(thread_num, "start: cal_2_js")

        js = np.zeros((6, 6, 49))
        for j in range(6):
            for k in range(6):
                # mean = np.mean(result[:, :, j, k], axis=0)
                mean = result[0, :, j, k]
                for s in range(49):
                    js[j][k][s] = JS_divergence_1(result[s + 1, :, j, k], mean)

        np.save(WORK_DIR + experience_id + "/marker_C/result/js_" + str(thread_num) + "_of256.npy", js)
    except Exception as e:
        print(e)


# 验证不同类之间差距区分导致最终性能的改变
def vali_2_js(thread_num):
    try:
        print(thread_num, "线程启动！！！")

        model = torchvision.models.alexnet(pretrained=True)
        criterion = torch.nn.CrossEntropyLoss()
        print_freq = 100
        train_dataloader = data_loader(root=WORK_DIR + "/tiny_image/tiny/", mode="train", pin_memory=True,
                                       batch_size=16)
        node = np.load(WORK_DIR + "/diff_class_2/result/node_" + str(30 + 5 * thread_num) + ".npy")
        validate_js(val_loader=train_dataloader, model=model, criterion=criterion,
                    print_freq=print_freq, node=node, tresh=1, thread_num=thread_num,
                    is_cuda=True)

        print(thread_num, "线程写入结果完毕!")
    except Exception as e:
        print(e)


def test_01():
    # make_dir(train_dir=WORK_DIR+"/train/train/val/", class_num=2, image_num=1)
    # pool = MyPool(2)
    # for i in range(2):
    #
    #     pool.apply_async(generate_coinT, args=(([1000, i]), ))
    #
    # pool.close()
    # pool.join()

    # result = np.zeros((2, 1000, 256, 6, 6))
    # for i in range(2):
    #     loader = data_loader(WORK_DIR+"/diff_class_2/"+str(i))
    #     result[i] = get_n_mid(data_loader=loader, model=torchvision.models.alexnet(pretrained=True), layer=5, is_cuda=True)
    # np.save(WORK_DIR+"/diff_class_2/result/mid_result.npy", result)
    #
    # result = np.load(WORK_DIR+"/diff_class_2/result/mid_result.npy")
    #
    # print(np.shape(result))
    # pool = MyPool(256)
    #
    # for i in range(256):
    #     print(i)
    #     pool.apply_async(cal_2_js, args=(result[:, :, i, :, :], i, ))
    # pool.close()
    # pool.join()

    # for i in range(256):
    #     print(i)
    #     for j in range(6):
    #         for k in range(6):
    #             js[i][j][k] = JS_divergence_1(result[0, :, i, j, k], result[1, :, i, j, k])
    # np.save(WORK_DIR+"/diff_class_2/result/js.npy")

    # js = np.zeros((256, 6, 6))
    # for i in range(256):
    #     js[i] = np.load(WORK_DIR+"/diff_class_2/result/js" + str(i) + ".npy")
    #
    # for num in range(10):
    #     node = np.ones((256, 6, 6))
    #     js_tresh = np.percentile(js, 30 + 5 * num)
    #     for i in range(256):
    #         for j in range(6):
    #             for k in range(6):
    #                 if js[i, j, k] < js_tresh:
    #                     node[i, j, k] = 0
    #     np.save(WORK_DIR+"/diff_class_2/result/node_" + str(30 + 5 * num) + ".npy", node)

    pool = MyPool(10)
    for thread_num in range(10):
        pool.apply_async(vali_2_js, args=(thread_num,))
    pool.close()
    pool.join()

    # for i in range(10):
    #     vali_2_js(i)


# 计算原始数据的中间结果
def cal_ori():
    model = torchvision.models.alexnet(pretrained=True)
    for i in range(200):
        progress(i / 200)
        if not os.path.exists(WORK_DIR + experience_id + "/marker_C/result/origin/" + str(i) + "/val/class"):
            os.makedirs(WORK_DIR + experience_id + "/marker_C/result/origin/" + str(i) + "/val/class")
        img = np.load(WORK_DIR + "exper_v1/marker_C/result/data_imglist_" + str(i) + ".npy")
        cv2.imwrite(WORK_DIR + "exper_v1/marker_C/result/origin/" + str(i) + "/val/class/origin.jpg", img)
        data_load = data_loader(WORK_DIR + "exper_v1/marker_C/result/origin/" + str(i))
        np.save(WORK_DIR + experience_id + "/marker_C/result/origin/origin_" + str(i) + ".npy",
                get_n_mid(model=model, data_loader=data_load, layer=5))


# 计算给定标签的数据的js散度，这个主要是计算序列js
def cal_js(thread_num, filename, compare, degree):
    # try:

    print("start: degree: ", degree, "thread_num:", thread_num, "compare:", compare)
    js = np.zeros(shape)
    count = 0
    data = np.load(WORK_DIR + experience_id + filename + "/marker_C/result/xulie/xulie_" + str(thread_num) + "_1.npy")
    comp = np.load(WORK_DIR + experience_id + filename + "/marker_C/result/xulie/xulie_" + str(compare) + "_0.npy")

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                js[i, j, k] = JS_divergence_1(data[:, i, j, k], comp[:, i, j, k])
                if js[i, j, k] < 1e-10:
                    # print((data[:, i, j, k] - comp[:, i, j, k]).mean())
                    count += 1
    print(thread_num, "js中为0的个数：", count)
    np.save(
        WORK_DIR + experience_id + filename + "/marker_C/result/js_" + str(degree) + "/js_" + str(thread_num) + ".npy",
        js)

    print("end: degree: ", degree, "thread_num:", thread_num)
    # except Exception as e:
    #     print(e)


# 通过方差计算删除节点，也就是以稳定性作为标准，结果保留为包含percent的文件。
def multi_get_acc(thread_num, filename, js_std):
    print("multi_get_acc, 计算：", thread_num)
    percent = thread_num * 5 + 30
    js_tresh = np.percentile(js_std, percent)
    node = np.ones(shape, dtype=np.float32)

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                # 这里是删除方差大的节点，可以理解就是对于同类图片来说，我们需要保留能提取稳定特征的节点，但是也可以试试删除方差小的节点
                if js_std[i][j][k] > js_tresh:
                    # if js_std[i][j][k] < js_tresh:
                    node[i, j, k] = 0

    np.save(WORK_DIR + experience_id + filename + "/marker_C/result/node_" + str(percent) + ".npy", node)
    # node = np.load(WORK_DIR + "/marker_C/result/node_" + str(percent) + ".npy")

    # 对js做标准化，然后将其中不在3倍标准差的点删除
    # try:
    #     js = (js - np.mean(js)) / np.std(js)
    #     # 范围： [μ+σ+(2σ/10)*i, μ+σ+(2σ/10)*i];
    #     miu = js.mean()
    #     sigma = js.std()
    #     range_0 = miu - (sigma + (2 * sigma / 10 * thread_num))
    #     range_1 = miu + (sigma + (2 * sigma / 10 * thread_num))
    #     print(thread_num, range_0, range_1)
    #     print(js.shape)
    #
    #     node = np.reshape(list(map(lambda x: 1 if (range_1 > x > range_0) else 0, js.reshape(-1, ))), (256, 6, 6))
    #
    #     np.save(WORK_DIR + "/marker_C/result/sigma_node_" + str(thread_num) + ".npy", node)
    #     print("计算node完毕：", thread_num)
    #
    # except Exception as e:
    #     print(e)
    return

    model = torchvision.models.alexnet(pretrained=True)
    criterion = torch.nn.CrossEntropyLoss()
    print_freq = 100
    # 整个数据集进行验证
    train_loader = data_loader(root=WORK_DIR + "/train/train/", mode="val")
    # 只采用选用的那个类进行验证
    # train_loader = data_loader(root="/home/hongwu/tmp/pycharm_project_490/result/0/")
    validate_js(val_loader=train_loader, model=model, criterion=criterion,
                print_freq=print_freq, node=node, tresh=1, thread_num=thread_num)


# 计算给定数据之间的js散度
def cal_js_1(thread_num, target_dir, output_dir):
    print("JS计算开始:", thread_num)

    try:
        target = np.load(target_dir)
        output = np.load(output_dir + str(thread_num) + ".npy")

        js = np.zeros((256, 6, 6))
        for i in range(256):
            if i % 64 == 0:
                print(thread_num, "进程：", i / 256)
            for j in range(6):
                for k in range(6):
                    js[i, j, k] = JS_divergence_1(target[:, i, j, k], output[:, i, j, k])
                    print(js[i, j, k])
                    # sleepawhile(4)
                    # print(js[i, j, k])
                    # if np.abs(js[i, j, k] - 0.0) < 1e-6:
                    #     print((output[:, i, j, k] == target[:, i, j, k]).mean())
                    #     print(output[:, i, j, k].mean())
        np.save(WORK_DIR + experience_id + "/marker_A/result/js_" + str(thread_num) + ".npy", js)
    except Exception as e:
        print(e)


def getSomeNode(thread_num, thread_num_1):
    try:
        print(thread_num, thread_num_1, 'start')

        js = np.load(WORK_DIR + experience_id + "/marker_A/result/js_" + str(thread_num) + ".npy")
        print(js.shape)

        # 标准化有必要吗。。。
        # 首先进行标准化
        # js = (js - js.mean()) / js.std()

        js_kur = np.zeros((256, 6, 6))
        for i in range(256):
            for j in range(6):
                for k in range(6):
                    js_kur[i, j, k] = get_kurto(js[:, i, j, k])

        node = np.ones((256, 6, 6))
        js_tresh = np.percentile(js_kur, 2.5 + 2.5 * thread_num_1)

        # 计算峰度作为节点筛选标准

        for i in range(256):
            for j in range(6):
                for k in range(6):
                    if js_kur[i, j, k] <= js_tresh:
                        node[i, j, k] = 0
        np.save(WORK_DIR + experience_id + "/marker_A/result/node_" + str(thread_num) + "_" + str(
            thread_num_1 * 2.5 + 2.5) + ".npy",
                node)
        print(thread_num, thread_num_1, "end")
    except Exception as e:
        print(e)


def handleJsGetNode(thread_num):
    try:
        print(thread_num, "start")

        # 标准化之后选取一定比例的点作为js几乎无变化的点
        # pool = MyPool()
        # for i in range(5):
        #     pool.apply_async(getSomeNode, args=(thread_num, i,))
        # pool.close()
        # pool.join()
        js = np.zeros((20, 256, 6, 6))
        for i in range(20):
            js[i] = np.load(WORK_DIR + experience_id + "/marker_A/result/js_" + str(i) + ".npy")

        js_kur = np.zeros((256, 6, 6))
        for i in range(256):
            for j in range(6):
                for k in range(6):
                    # print(js[:, i, j, k])
                    js_kur[i, j, k] = stats.kurtosis(js[:, i, j, k])
                    # print(js_kur[i, j, k])
                    # sleepawhile(2)
        # np.save(WORK_DIR+experience_id+"/marker_A/result/kurto_"+str(thread_num)+".npy", js_kur)
        print(js_kur)

        for percent_level in range(5):
            node = np.ones((256, 6, 6))
            js_tresh = np.percentile(js_kur, 2.5 + 2.5 * percent_level)
            # 计算峰度作为节点筛选标准
            for i in range(256):
                for j in range(6):
                    for k in range(6):
                        if js_kur[i, j, k] <= js_tresh:
                            node[i, j, k] = 0
            np.save(WORK_DIR + experience_id + "/marker_A/result/node_" + str(thread_num) + "_" + str(
                percent_level * 2.5 + 2.5) + ".npy",
                    node)

        print(thread_num, "end")
    except Exception as e:
        print(e)


# 频谱分析
def fft_test(js, thread_num):
    '''
    计算频谱
    :param thread_num:
    :param js: shape=[6, 6]
    :return:
    '''

    print(thread_num, "start")

    try:
        node = np.zeros((6, 6))
        node_4 = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                fft_y = fft(js[:, i, j])
                N = 1000
                x = np.arange(N)  # 频率个数
                abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
                angle_y = np.angle(fft_y)  # 取复数的角度
                normalization_y = abs_y / N  # 归一化处理（双边频谱）
                half_x = x[range(int(N / 2))]  # 取一半区间
                normalization_half_y = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）
                # return normalization_half_y
                peaks, _ = find_peaks(normalization_half_y, prominence=normalization_half_y[1:].max() / 2)

                if np.abs(peaks - 19).min() < 2:
                    node[i][j] = 1
                if np.abs(peaks - 19).min() < 4:
                    node_4[i][j] = 1
        np.save(WORK_DIR + experience_id + "/marker_D/result/fft_node_6_6_" + str(thread_num) + ".npy", node)
        np.save(WORK_DIR + experience_id + "/marker_D/result/fft_node_4_6_6_" + str(thread_num) + ".npy", node_4)
    except Exception as e:
        print(e)
    print(thread_num, "end")


# 通过js序列进行频谱分析得出满足条件的点
def get_fftNode():
    js = np.zeros((1000, 256, 6, 6))
    for i in range(1000):
        progress(i / 1000)
        js[i] = np.load(WORK_DIR + experience_id + "/marker_C/result/js_0/js_" + str(i + 1) + ".npy")
    pool = MyPool()
    for i in range(256):
        pool.apply_async(fft_test, args=(js[:, i, :, :], i,))
    pool.close()
    pool.join()


# 这是手写DW检验，不过后面用的是stats中的sapi
def DW_Stat(js_order):  # 德宾-瓦特逊检验， Durbin-Watson Statistics
    fenzi = 0
    fenmu = 0
    for i in range(1000):
        fenzi += np.square(js_order[i + 1] - js_order[i])
        fenmu += np.square(js_order[i])
    fenmu += np.square(js_order[1000])
    return fenzi / fenmu


# 通过js序列计算dw
def dw_stat_6_6(js_order, thread_num, mode="dw"):
    print("start:", thread_num)

    try:
        if mode == "dw":
            node = np.zeros((6, 6))
            for i in range(6):
                for j in range(6):
                    temp = sm.stats.durbin_watson(js_order[:, i, j])
                    if temp > 1.779 or temp < 1.758:
                        node[i][j] = 1
                    np.save(WORK_DIR + "/marker_D/result/dw_" + str(thread_num) + "_6_6.npy", node)
        else:
            node01 = np.zeros((6, 6))
            node05 = np.zeros((6, 6))
            node10 = np.zeros((6, 6))
            for i in range(6):
                for j in range(6):
                    temp = stattools.coint([(1 + np.sin(7 * k * np.pi / 180)) for k in np.arange(1000)],
                                           js_order[:, i, j])
                    if temp[0] < temp[2][0] and temp[1] < 0.01:
                        node01[i][j] = 1
                    if temp[0] < temp[2][1] and temp[1] < 0.05:
                        node05[i][j] = 1
                    if temp[0] < temp[2][2] and temp[1] < 0.10:
                        node10[i][j] = 1
                    np.save(WORK_DIR + "/marker_D/result/node_10_" + str(thread_num) + ".npy", node10)
                    np.save(WORK_DIR + "/marker_D/result/node_01_" + str(thread_num) + ".npy", node01)
                    np.save(WORK_DIR + "/marker_D/result/node_05_" + str(thread_num) + ".npy", node05)
    except Exception as e:
        print(e)

    print("end:", thread_num)
    # 直接根据dw结果计算节点保存与否


# 计算格兰杰因果关系检验的p值
def granger_test(js_order, thread_num):
    try:
        print("start:", thread_num)
        lag = np.zeros((6, 6))
        gt_6_6 = np.zeros((6, 6))
        for j in range(6):
            for k in range(6):
                gt = stattools.grangercausalitytests(np.vstack((js_order[:, k, j], [(1 + np.sin(7 * i * np.pi / 180))
                                                                                    for i in np.arange(1000)])).T,
                                                     maxlag=20, verbose=False)

                lag[j][k] = np.argmax([(gt[i + 1][0]['params_ftest'][0]) for i in range(20)]) + 1
                gt_6_6[j][k] = gt[lag[j][k]][0]['params_ftest'][1]

        np.save(WORK_DIR + experience_id + "/marker_D/result/gt_" + str(thread_num) + "_6_6.npy", gt_6_6)
        print("end:", thread_num)
    except Exception as e:
        print(e)


# 通过p值计算节点
def get_D_Node(p):
    gt_p = np.zeros((256, 6, 6))
    for i in range(256):
        progress(i / 256)
        gt_p[i] = np.load(WORK_DIR + "/marker_C/result/gt_" + str(i) + "_6_6.npy")
    # node_1 = np.reshape(list(map(lambda x: 1 if x < 0.01 else 0, gt_p.reshape((-1,)))), (256, 6, 6))
    # node_5 = np.reshape(list(map(lambda x: 1 if x < 0.05 else 0, gt_p.reshape((-1,)))), (256, 6, 6))
    # np.save(WORK_DIR+"/marker_D/result/node_0.01.npy", node_1)
    # np.save(WORK_DIR+"/marker_D/result/node_0.05.npy", node_5)

    node_10 = np.reshape(list(map(lambda x: 1 if x < p else 0, gt_p.reshape((-1,)))), (256, 6, 6))
    np.save(WORK_DIR + "/marker_D/result/node_" + str(p) + ".npy", node_10)


# 进度条
def progress(percent, width=50):
    if percent > 1:
        percent = 1
    show_str = (('[%%-%ds]' % width) % (int(percent * width) * '#'))
    print('\r%s %d%%' % (show_str, int(percent * 100)), end='')


# 测试原始数据，后面直接在summary中改变loader的root参数即可实现此功能
def test_ori():
    pool = MyPool()
    index = ["", "_4"]
    for i in range(10):
        for j in range(2):
            node_C = np.load(
                WORK_DIR + experience_id + "/marker_C/result/node/node_" + str(30 + i * 5) + ".npy").astype(np.int8)
            node_D = np.load(WORK_DIR + experience_id + "/marker_D/result/fft_node" + index[j] + ".npy").astype(np.int8)
            node = np.bitwise_or(node_C, node_D)

            # node = np.ones((256, 6, 6))

            percent = np.mean(node)
            model = torchvision.models.alexnet(pretrained=True)
            criterion = torch.nn.CrossEntropyLoss()
            print_freq = 1000
            node = node.astype(np.float32)
            loader = data_loader(root=WORK_DIR, mode="val")
            # loader = data_loader(root=WORK_DIR+"/team/", mode="train",
            #                      pin_memory=True)
            pool.apply_async(validate_js, args=(
                loader, model, criterion, print_freq, node, 1, 30 + i * 5 + j, percent, "CorD", False,))

    pool.close()
    pool.join()


# 测试hx的数据
def test_hx():
    js_1_test = np.load(WORK_DIR + "/marker_C/result/js_1/cat_js.npy")
    js_1_test = np.diff(js_1_test, axis=0)
    print(js_1_test.shape)
    ngt = np.zeros((43264,))
    lag = np.zeros((43264,))
    for i in range(43264):
        progress(i / 43264)
        gt = stattools.grangercausalitytests(np.vstack((js_1_test[:, i], [
            (1.1 + 0.9 * math.sin(math.pi * i * 7.0 / 180.0) + float(i) / 500.0 + np.random.randint(-10, 10) / 50.0) for
            i in np.arange(999)])).T, maxlag=20, verbose=False)
        lag[i] = np.argmax([(gt[i + 1][0]['params_ftest'][0]) for i in range(20)]) + 1
        ngt[i] = gt[lag[i]][0]['params_ftest'][1]
    np.save(WORK_DIR + "/marker_D/result/gt_hx.npy", ngt)


def get_kurto(data):
    mean_ = data.mean()
    std_ = data.std()
    kurto = np.mean((data - mean_) ** 4) / pow(std_, 4)
    return kurto


def generate_data(image, img_size, centre, col, row, index, main_dir):
    if not os.path.exists(WORK_DIR + experience_id + main_dir + "/marker_C/" + str(index) + "/val/class/"):
        os.makedirs("WORK_DIR+experience_id+main_dir" + "/marker_C/" + str(index) + "/val/class/")

    try:
        for j in range(col):
            if j % 2 == 0:
                for i in range(row):
                    x_0 = i * centre[0] / col
                    x_1 = j * centre[1] / row
                    y_0 = i * (img_size[0] - centre[2]) / col + centre[2]
                    y_1 = j * (img_size[1] - centre[3]) / row + centre[3]
                    # print(x_0, y_0, x_1, y_1)
                    imag2 = image.crop((x_0, x_1, y_0, y_1))
                    imag2.save(WORK_DIR + experience_id + main_dir + "/marker_C/" + str(index) + "/val/class/" + str(
                        j * row + i) + ".jpg")
            else:
                for i in range(row - 1, -1, -1):
                    x_0 = i * centre[0] / col
                    x_1 = j * centre[1] / row
                    y_0 = i * (img_size[0] - centre[2]) / col + centre[2]
                    y_1 = j * (img_size[1] - centre[3]) / row + centre[3]
                    # print(x_0, y_0, x_1, y_1)
                    imag2 = image.crop((x_0, x_1, y_0, y_1))
                    imag2.save(WORK_DIR + experience_id + main_dir + "/marker_C/" + str(index) + "/val/class/" + str(
                        j * row + row - 1 - i) + ".jpg")
    except Exception as e:
        print(e)


def generate_scale(img, img_size, center, augnum, index, main_dir):
    # print(index)
    # if not os.path.exists(WORK_DIR + experience_id + main_dir + "/marker_C/" + str(index) + "/val/class/"):
    #     print("文件夹不存在")
    #     os.makedirs("WORK_DIR+experience_id+main_dir" + "/marker_C/" + str(index) + "/val/class/")

    try:
        deltax = img_size[0] / 10 / augnum
        deltay = img_size[1] / 10 / augnum
        for i in range(augnum):
            x_0 = center[0] - deltax * i
            x_1 = center[1] - deltay * i
            y_0 = center[2] + deltax * i
            y_1 = center[3] + deltay * i
            img_temp = img.crop((x_0, x_1, y_0, y_1))
            img_temp.save(
                WORK_DIR + experience_id + main_dir + "/marker_C/" + str(index) + "/val/class/" + str(i) + ".jpg")
    except Exception as e:
        print(e)


def generate(filename):
    # main_dir=random.sample(os.listdir("./train/train/val/"), 1)
    # print(filename)

    # pool = MyPool()
    # for i in range(1):
    # print(random_class[i])

    # if filename != "None" and random_class[i] != filename:
    #     continue
    # main_dir = random_class[i]
    main_dir = filename
    # print(main_dir)
    if os.path.exists(WORK_DIR + experience_id + main_dir + "/marker_C/val/"):
        # print("目录存在")
        shutil.rmtree(WORK_DIR + experience_id + main_dir + "/marker_C/val/")
    # if not os.path.exists(WORK_DIR + experience_id + main_dir + "/marker_C/val/"):
    #     os.makedirs(WORK_DIR + experience_id + main_dir + "/marker_C/val/")

    shutil.copytree(WORK_DIR + experience_id + "/../val/" + main_dir,
                    WORK_DIR + experience_id + main_dir
                    + "/marker_C/val/" + main_dir)
    print("????start: ", main_dir)
    files = (random.sample(os.listdir(WORK_DIR + "/train/train/val/" + main_dir + "/"), 50))
    for index in range(50):
        if not os.path.exists(
                WORK_DIR + experience_id + main_dir + "/marker_C/" + "/" + str(index) + "/val/class/"):
            os.makedirs(WORK_DIR + experience_id + "/" + main_dir + "/marker_C/" + str(index) + "/val/class")
        file = files[index]
        # print("start: ", file)
        img = Image.open(WORK_DIR + "/train/train/val/" + main_dir + "/" + file)
        img_size = img.size
        center = np.array([img_size[0] / 5, img_size[1] / 5, img_size[0] * 4 / 5, img_size[1] * 4 / 5])
        # pool.apply_async(generate_data, args=(img, img_size, center, 10, 10, index, main_dir,))
        generate_data(img, img_size, center, 10, 10, index, main_dir)
        # pool.apply_async(generate_scale, args=(img, img_size, center, 50, index, main_dir,))
        # generate_scale(img, img_size, center, 50, index, main_dir)
        # print("end: ", file)

    print("end: ", filename)

    # pool.close()
    # pool.join()
