# encoding=utf-8
# import threading
# 来，说句话0.0
# import requests
from warnings import simplefilter

from chonggou import *
import warnings
import torch.nn as nn

import faulthandler
# 在import之后直接添加以下启用代码即可
faulthandler.enable()
# 后边正常写你的代码


# import multiprocessing as mp

warnings.filterwarnings('ignore')
simplefilter(action='ignore', category=FutureWarning)

WORK_DIR = "/data/hongwu/"
experience_id = "/gauss_test_res/"

alex_shape = [256, 6, 6]
vgg_shape = [512, 7, 7]
res50_shape = [2048, 7, 7]
shape = res50_shape
model_name = "res50"

model = torchvision.models.resnext50_32x4d(pretrained=True)


def marker_A():
    # 选择之前选定的result图片作为标准，然后再随机抽样其他十类，每类一张图片，对其做干扰，最后得出十组js，并进行标准化，
    # 对每组进行筛选，通过投票进行删除，参数为js阈值以及投票阈值

    # 选取十个类
    # make_dir(train_dir=WORK_DIR + "/train/train/val/", class_num=20, image_num=1,
    #          save_dir=WORK_DIR + experience_id+"/marker_A/result")
    # for i in range(10):
    #     os.makedirs(WORK_DIR + experience_id+"/marker_A/" + str(i+10) + "/val/class/")

    # ## 进行数据干扰
    # pool = MyPool()
    # for i in range(20):
    #     pool.apply_async(generate_coinT,
    #                      args=(1000, i, WORK_DIR + experience_id+"/marker_A/", WORK_DIR + experience_id+"/marker_A/",))
    # pool.close()
    # pool.join()
    # #
    # # # 获取十个类干扰之后的网络中间结果，其每个类的维度为(1000, 256, 6, 6)
    # pool_1 = MyPool()
    # for i in range(20):
    #     pool_1.apply_async(augMidres, args=(i, WORK_DIR + experience_id+"/marker_A/",))
    # pool_1.close()
    # pool_1.join()

    # 读取中间结果并计算js，其中基准类就选择之前marker_C中的第0张图片，直接读取其中间结果即可
    # pool_js = MyPool()
    # for i in range(20):
    #     pool_js.apply_async(func=cal_js_1, args=(i, WORK_DIR + experience_id + "/marker_C/result/mid_result_32.npy",
    #                                              WORK_DIR + experience_id + "/marker_A/result/mid_result_"))
    # pool_js.close()
    # pool_js.join()

    # 计算完js之后，需要对其做标准化，并选取其中每个部分点，并进行存储
    handleJsGetNode(0)


def marker_B():
    # 与C做法一致，不过进行干扰的时候不采用调制。
    # 生成图片并保存到文件夹中
    # make_dir(train_dir=WORK_DIR+"/train/train/val/", class_num=1, image_num=200,
    #     #          save_dir=WORK_DIR + "/marker_B/result/")
    #     #
    #     # # 保存原始图片并计算中间输出,保存到marker_B的result目录中的mid_result_10000.npy文件中
    #     # os.makedirs(WORK_DIR+"/marker_B/10000/val/class/")
    #     # for i in range(200):
    #     #     img = np.load(WORK_DIR+"/marker_B/result/data_imglist_"+str(i)+".npy")
    #     #     cv2.imwrite(WORK_DIR+"/marker_B/10000/val/class/"+str(i)+".jpg", img)
    #     # augMidres(index=10000, data_dir=WORK_DIR + "/marker_B/")
    #     #
    #     # # 然后将其他文件进行对比就好了
    #     #
    #     # for i in range(200):
    #     #     os.makedirs(WORK_DIR + "/marker_B/" + str(i) + "/val/class/")
    #     #
    #     # pool = MyPool()
    #     # for i in range(200):
    #     #     pool.apply_async(generate_coinT, args=(1000, i, WORK_DIR + "/marker_B/", WORK_DIR + "/marker_B/"))
    #     # pool.close()
    #     # pool.join()
    #     #
    # pool_mid = MyPool()
    # for i in range(200):
    #     pool_mid.apply_async(augMidres, args=(i, WORK_DIR + "/marker_B/",))
    # pool_mid.close()
    # pool_mid.join()

    # 中间结果输出有200个,每个维度为(1000, 256, 6, 6)
    mid_result = np.zeros((200, 1000, 256, 6, 6))
    for i in range(200):
        if i % 20 == 0:
            print(i)
        mid_result[i] = np.load(WORK_DIR + "/marker_B/result/mid_result_" + str(i) + ".npy")
    # os.makedirs(WORK_DIR + "/marker_B/result/js_0/")
    for i in range(1000):
        np.save(WORK_DIR + "/marker_B/result/js_0/mid_" + str(i) + ".npy", mid_result[:, i, :, :, :])

    # # 计算0阶js序列, 然后找出几乎不变的节点,还是可以采用方差来进行筛选. # 改变使用峰度进行筛选
    pool_js = MyPool()
    for thread_num in range(1000):
        pool_js.apply_async(cal_js_1, args=(thread_num, WORK_DIR + "/marker_B/result/mid_result_10000.npy",
                                            WORK_DIR + "/marker_B/result/mid_result_",))
    pool_js.close()
    pool_js.join()

    # 然后对的出来的js计算方差,然后根据方差得出最终node矩阵
    # pool_node = MyPool()
    # for i in range(256):
    #     pool_node.apply_async()
    # pool_node.close()
    # pool_node.join()

    js = np.zeros((1000, 256, 6, 6))
    # js_std = js.std(axis=0)

    # 计算峰度
    js_std = np.zeros((256, 6, 6))
    for i in range(256):
        for j in range(6):
            for k in range(6):
                js_std[i, j, k] = np.mean((js[:, i, j, k] - js[:, i, j, k].mean()) ** 4) / pow(js[:, i, j, k].std(), 4)

    for i in range(5):
        node = np.ones((256, 6, 6))
        percent = 2.5 + 2.5 * i
        js_tresh = np.percentile(js_std, percent)
        for i in range(256):
            for j in range(6):
                for k in range(6):
                    if js_std[i, j, k] > js_tresh:
                        node[i, j, k] = 0

        np.save(WORK_DIR + "/marker_B/result/node_" + str(percent) + ".npy", node)


def marker_C(filename):
    # # 生成图片并保存到文件夹中
    image_num = 1
    augnum = 100

    # if not os.path.exists(WORK_DIR+experience_id+filename+"/marker_C/val"):
    #     os.makedirs(WORK_DIR+experience_id+filename+"/marker_C/val/")
    # # # shutil.copytree(WORK_DIR+"/val/"+filename, WORK_DIR+experience_id+filename+"/marker_C/val/"+filename)
    # #
    # make_dir(train_dir="/data/imagenet_2012/train/", filename=filename, class_num=1, image_num=image_num,
    #          save_dir=WORK_DIR + experience_id + filename +
    #                   "/marker_C"
    #                   "/result/")
    #
    # for i in range(image_num):
    #     if not os.path.exists(WORK_DIR + experience_id + filename + "/marker_C/" + str(i) + "/val/class/"):
    #         os.makedirs(WORK_DIR + experience_id + filename + "/marker_C/" + str(i) + "/val/class/")
    #
    # pool = MyPool()
    # for i in range(image_num):
    #     pool.apply_async(generate_coinT, args=(
    #         augnum, i, WORK_DIR + experience_id + filename + "/marker_C/",
    #         WORK_DIR + experience_id + filename + "/marker_C/",))
    # pool.close()
    # pool.join()
    #
    # pool = MyPool()
    # for i in range(image_num):
    #     # def augMidres(index, model, augnum, shape, data_dir="/home/hongwu/python/Image/exper_v1/marker_C/"):
    #     pool.apply_async(augMidres,
    #                      args=(i, model, augnum, shape, WORK_DIR + experience_id + filename + "/marker_C/", False,))
    # pool.close()
    # pool.join()
    #
    # # 读取数据，维度为[200， 1000， 256. 6. 6]
    # data_shape = np.concatenate(([image_num], [augnum], shape))
    # data = np.zeros(data_shape)
    #
    # for i in range(image_num):
    #     progress(i / image_num)
    #     data[i] = np.load(WORK_DIR + experience_id + filename + "/marker_C/result/mid_result_" + str(i) + ".npy")
    #
    # if not os.path.exists(WORK_DIR + experience_id + filename + "/marker_C/result/xulie"):
    #     os.makedirs(WORK_DIR + experience_id + filename + "/marker_C/result/xulie")

    # # 获取原始文件的中间结果，并保存为xulie_0_0
    # num = get_n_mid(data_loader(WORK_DIR + experience_id+"data/1000", mode="val"),
    #                 model=torchvision.models.alexnet(pretrained=True), layer=5)
    # np.save(WORK_DIR + experience_id + "/marker_C/result/xulie/xulie_0_0.npy", num)
    # print("origin数据计算完毕！！")

    # # 将结果分成1000个文件，然后再进行求0阶和1阶js序列。
    # for i in range(augnum):
    #     progress(i / augnum)
    #     np.save(WORK_DIR + experience_id + filename + "/marker_C/result/xulie/xulie_" + str(i + 1) + "_0.npy",
    #             data[:, i, :, :, :])
    #     np.save(WORK_DIR + experience_id + filename + "/marker_C/result/xulie/xulie_" + str(i + 1) + "_1.npy",
    #             data[:, i, :, :, :])

    # 首先计算0阶js序列
    # if not os.path.exists(WORK_DIR+experience_id+"/marker_C/result/js_0/"):
    #     os.makedirs(WORK_DIR+experience_id+"/marker_C/result/js_0/")
    # pool = MyPool()
    # for thread_num in range(100):
    #     pool.apply_async(cal_js, args=(thread_num + 1, 0, 0,))
    # pool.close()
    # pool.join()

    # # 计算1阶js序列
    # if not os.path.exists(WORK_DIR + experience_id + filename + "/marker_C/result/js_1/"):
    #     os.makedirs(WORK_DIR + experience_id + filename + "/marker_C/result/js_1/")
    # pool = MyPool()
    # for thread_num in range(augnum - 1):
    #     pool.apply_async(cal_js, args=(thread_num + 2, filename, thread_num + 1, 1,))
    # pool.close()
    # pool.join()

    # 通过横向对比得出方差，最后找出稳定性较好的点，进行筛选
    mid_result = np.zeros(np.concatenate(([image_num], [augnum], shape)))
    for i in range(image_num):
        progress(i / image_num)
        mid_result[i] = np.load(WORK_DIR + experience_id + filename + "/marker_C/result/mid_result_" + str(i) + ".npy")

    pool_1 = MyPool()
    for thread_num in range(shape[0]):
        pool_1.apply_async(cal_2_js,
                           args=(mid_result[:, :, thread_num, :, :], thread_num, image_num, augnum, filename, shape,))
    pool_1.close()
    pool_1.join()

    # js = np.zeros(np.concatenate((shape, [image_num-1])))
    # for i in range(shape[0]):
    #     progress(i / shape[0])
    #     js[i] = np.load(WORK_DIR + experience_id + filename + "/marker_C/result/js_" + str(i) + "_of256.npy")
    # js_std = np.std(js, axis=3)
    # print("js数据加载完毕")
    #
    # js = np.mean(js, axis=3)
    #
    # pool_2 = MyPool()
    # for thread_num in range(10):
    #     pool_2.apply_async(multi_get_acc, args=(thread_num, js_std,))
    # pool_2.close()
    # pool_2.join()
    #
    # # for degree in range(2):
    # degree = 1
    # # node = np.zeros((256, 6, 6))
    # node = np.zeros(shape)
    # js_0 = np.zeros(np.concatenate(([augnum], shape)))
    # for i in range(augnum - 1):
    #     progress(i / augnum)
    #     js_0[i] = np.load(
    #         WORK_DIR + experience_id + filename + "/marker_C/result/js_" + str(degree) + "/js_" + str(i + 2) + ".npy")
    #
    # js_std = np.std(js_0, axis=0)
    # # for epo in range(10):
    # # js_tresh = np.percentile(js_std, 5 + epo * 5)
    # js_tresh = np.median(js_std) + 2 * np.std(
    #     np.sort(js_std.reshape(-1, ))[:int((shape[0] * shape[1] * shape[2]) * 4 / 5)])
    # # js_tresh = np.mean(js_std) - np.std(js_std)
    #
    # for i in range(shape[0]):
    #     progress(i / shape[0])
    #     for j in range(shape[1]):
    #         for k in range(shape[2]):
    #             if js_std[i][j][k] < js_tresh:
    #                 node[i, j, k] = 1
    # np.save(WORK_DIR + experience_id + filename + "/w_node.npy", node)
    #


#
# def marker_D():
#     # 通过计算其0阶或者1阶js是否随着干预有规律变化，筛选出marker_D的节点
#
#     js_1_test = np.zeros((1000, 256, 6, 6))
#     for i in range(1000):
#         progress(i / 1000)
#         js_1_test[i] = np.load(WORK_DIR + "/marker_C/result/js_1/js_" + str(i + 1) + ".npy")
#     pool_dw = MyPool()
#     for i in range(256):
#         # progress(i / 256)
#         pool_dw.apply_async(granger_test, args=(js_1_test[:, i, :, :], i,))
#         # pool_dw.apply_async(dw_stat_6_6, args=(js_1_test[:, i, :, :], i,))
#     pool_dw.close()
#     pool_dw.join()


def summary(filename, class_name):
    # index = ["01", "05", "10"]
    # for i in range(10):
    # j = 10
    # k = 0
    # r = 0
    # s = 0
    # for j in range(2):
    #     for k in range(2):
    # index_a = ["2.5", "5.0", "7.5", "10.0", "12.5"]
    # for k in range(2):
    # for a in range(5):

    #             for s in range(3):

    #       ----------------------------node_A--------------------------------------------
    # node_A = np.zeros((256, 6, 6))
    # for i in range(10):
    # node_A = np.load(WORK_DIR + "/marker_A/result/node_0_" + index_a[a] + ".npy")
    # node_A = node_A.astype(np.int8)

    #       ----------------------------node_B--------------------------------------------
    # node_B = np.load(WORK_DIR + "/marker_B/result/node_" + str(2.5 + 2.5 * k) + ".npy").astype(np.int8)

    #       ----------------------------node_C--------------------------------------------

    node_C = np.load(
        WORK_DIR + experience_id + class_name + "/w_node.npy").astype(np.int8)

    # #       ----------------------------node_D--------------------------------------------
    # node_D = np.load(WORK_DIR + experience_id + "/marker_D/result/fft.npy").astype(np.int8)
    # if k == 0:
    #     node_D = np.load(WORK_DIR + experience_id + "/marker_D/result/fft_node.npy").astype(np.int8)
    # else:
    #     node_D = np.load(WORK_DIR + experience_id + "/marker_D/result/fft_node_4.npy").astype(np.int8)

    #
    # # 将这些意见进行统一，首先试试做交集，也就是都同意删除的节点才进行删除
    # # node = np.bitwise_or(np.bitwise_or(node_A, node_B, node_C), node_D)
    # node = np.bitwise_or(node_C, node_D)
    # node = np.bitwise_and(node, node_A)

    node = node_C
    # node = np.ones((256, 6, 6))-node
    # node = np.ones((512, 7, 7))-node

    ## 测试三种不同方式的不同组合。
    # node_gauss = np.load(WORK_DIR + "gauss/" + class_name + "/node60.npy").astype(np.int8)
    # node_shift = np.load(WORK_DIR + "shift_test/" + class_name + "/node40.npy").astype(np.int8)
    # node_rotate = np.load(WORK_DIR + "rotate_test/node30.npy").astype(np.int8)

    # node = np.bitwise_or(np.bitwise_or(node_gauss, node_shift), node_rotate)

    # node = np.ones((256, 6, 6))
    criterion = torch.nn.CrossEntropyLoss()
    print_freq = 1000
    node = node.astype(np.float32)
    print(0, "==== ", node.mean())

    # loader = data_loader(root=WORK_DIR + "/", mode="val",
    #                      pin_memory=False)

    # loader = data_loader(root=WORK_DIR, mode="val", pin_memory=False)

    # def validate_js(val_loader, model, criterion, print_freq, node, tresh, thread_num,
    # is_cuda=False)
    return validate_js(model, criterion, print_freq, node, 1, 0, filename, class_name, False)
    # validate_js(loader, model, criterion, print_freq, node, 1, 0, True)


def get_precise(filename, class_name):
    node = np.ones(shape)
    criterion = torch.nn.CrossEntropyLoss()
    print_freq = 1000

    # loader = data_loader(root=WORK_DIR + "/", mode="val",
    #                      pin_memory=False)

    # loader = data_loader(root="/data/imagenet_2012/", mode="val", pin_memory=False,
    #                      batch_size=64)
    # loader = data_loader(root=WORK_DIR, mode="val", pin_memory=False)

    # def validate_js(val_loader, model, criterion, print_freq, node, tresh, thread_num,
    # is_cuda=False)
    # beita=np.load(WORK_DIR+experience_id+class_name+"/beita.npy")
    return validate_js(model, criterion, print_freq, node, 1, 0, filename, class_name, False)
    # validate_js(loader, model, criterion, print_freq, beita, 1, 0, filename, class_name, False)


def cal_w(point_1, point_2):
    point_1 = (point_1 + 1e-10)  # /  (point_1.sum() + 1e-7)
    point_2 = (point_2 + 1e-10)  # / (point_2.sum() + 1e-7)
    # print(point_1.sum(), point_2.sum())
    # print(point_1)

    # M = (point_1 + point_2) / 2
    # distance = 0.5 * scipy.stats.entropy(point_1, M) + 0.5 * scipy.stats.entropy(point_2, M)
    point_1 = relu(point_1)
    point_2 = relu(point_2)

    n = np.shape(point_2)[0]
    a = np.arange(n)
    distance = wasserstein_distance(a, a, point_1, point_2)
    return distance


def relu(point):
    data = np.array(point)
    for i in range(np.shape(data)[0]):
        if data[i] <= 0:
            data[i] = 0.00001
    return data

#
# def pool_cal_w(thread_num, outNum, data, standard):
#     try:
#         if thread_num % 1000 == 0:
#             print("start: ", thread_num)
#         w = np.zeros((7, 7, outNum))
#         for i in range(7):
#             for j in range(7):
#                 for k in range(outNum):
#                     w[i, j, k] = cal_w(data[k, :, i, j], standard[:, i, j])
#         np.save("/data/hongwu/result/cat_outjs/w_" + str(thread_num) + ".npy", w)
#         if thread_num % 1000 == 0:
#             print("end: ", thread_num)
#     except Exception as e:
#         print(e)


def get_outw(outNum):
    class_name = datasets.ImageFolder("/data/imagenet_2012/val/").classes
    out_name = set()
    while (1):
        if len(out_name) == outNum:
            break;
        name = sample(class_name, 1)
        #         print(name[0])
        if name[0] != 'n02123045':
            out_name.add(name[0])
    mid_result = np.zeros((outNum, 100, 2048, 7, 7))
    for i in range(outNum):
        a = np.random.randint(20)
        mid_result[i] = np.load(
            "/data/hongwu/gauss_test_res/" + str(class_name[i]) + "/marker_C/result/mid_result_" + str(a) + ".npy")
    np.save("/data/hongwu/result/cat_outjs/mid_result.npy", mid_result)
    # mid_result = np.load("/data/hongwu/result/cat_outjs/mid_result.npy")
    standard = np.load("/data/hongwu/gauss_test_res/n02123045/marker_C/result/mid_result_0.npy")
    pool = Pool()
    for i in range(2048):
        pool.apply_async(pool_cal_w, args=(i, outNum, mid_result[:, :, i, :, :], standard[:, i, :, :],))
    pool.close()
    pool.join()


def Optimaze_1(function, next_random=True):
    global node_num
    print("-----------------------------------------数据准备-------------------------------------------------------------")

    model = torchvision.models.resnext50_32x4d(pretrained=True)
    # standard = np.load("/data/hongwu/gauss_test_res/n02123045/marker_C/result/mid_result_0.npy")

    if next_random:
        dele = np.load("/data/hongwu/gauss_test_res/n02123045/w_node.npy").reshape(-1, )
        delete = []
        for i in range(np.shape(dele)[0]):
            if dele[i] == 0:
                delete.append(i)
        np.save("/data/hongwu/result/delete.npy", delete)
    else:
        delete = np.load("/data/hongwu/result/delete.npy")

    #     Inclass:

    if next_random:
        cat_mid_result = np.zeros((20, 100, 2048, 7, 7))
        for i in range(20):
            progress(title="读取cat中间结果：", percent=i / 20)
            cat_mid_result[i] = np.load(
                "/data/hongwu/gauss_test_res/n02123045/marker_C/result/m id_result_" + str(i) + ".npy")
        cat_mid_result_del = np.delete(cat_mid_result.reshape((20, 100, -1)), delete, 2)

        np.save("/data/hongwu/result/cat_mid_result_del.npy", cat_mid_result_del)
    else:
        cat_mid_result_del = np.load("/data/hongwu/result/cat_mid_result_del.npy")

    if next_random:
        mid_w = np.zeros((2048, 7, 7, 19))
        for i in range(2048):
            progress(title="读取cat类内w：", percent=i / 2048)
            mid_w[i] = np.load("/data/hongwu/gauss_test_res/n02123045/marker_C/result/js_" + str(i) + "withcat0.npy")
        mid_w = mid_w.reshape((-1, 19))
        mid_w_del = np.delete(mid_w, delete, 0)
        np.save("/data/hongwu/result/mid_w_del.npy", mid_w_del)
    else:
        mid_w_del = np.load("/data/hongwu/result/mid_w_del.npy")

    # last_result = np.zeros((20, 100, 1000))
    # for i in range(20):
    #             print(i)
    #     for j in range(100):
    #         last_result[i][j] = model.fc(model.avgpool(torch.Tensor(cat_mid_result[i][j])).view(-1)).detach().numpy()

    # last_res = last_result[:, :, 281]
    # last_w = np.zeros(19)
    # for i in range(19):
    #     last_w[i] = cal_w(last_res[i + 1], last_res[0])
    # standard = np.load("/data/hongwu/gauss_test_res/n02123045/marker_C/result/mid_result_0.npy")

    # 计算无风险收益
    # 将中间网络输出置为不同的随机数，然后输出到最终节点，获得w序列

    if next_random:
        rf_last = np.zeros((20, 100))
        for i in range(20):
            progress(title="计算无风险收益：", percent=i / 20)
            for j in range(100):
                random_res = np.random.random()
                ran_result = np.ones((2048, 7, 7)) * random_res
                rf_last[i][j] = model.fc(model.avgpool(torch.Tensor(ran_result)).view(-1)).detach().numpy()[281]

        rf_w = np.zeros(19, )
        for i in range(19):
            progress(title="计算无风险：", percent=i / 19)
            rf_w = cal_w(rf_last[i + 1], rf_last[0])
        Erf = np.mean(rf_w)

        np.save("/data/hongwu/result/Erf.npy", Erf)
    else:
        Erf = np.load("/data/hongwu/result/Erf.npy")

    # outClass:
    outNum = 100

    # 选取不同随机图片时打开
    if next_random:
        get_outw(outNum)
        out_mid_result = np.load("/data/hongwu/result/cat_outjs/mid_result.npy")
        out_mid_result_1 = out_mid_result.reshape((outNum, 100, -1))
        out_mid_result_del = np.delete(out_mid_result_1, delete, 2)
        np.save("/data/hongwu/result/cat_outjs/out_mid_result_del.npy", out_mid_result_del)

        out_w = np.zeros((2048, 7, 7, outNum))
        for i in range(2048):
            progress(title="读取类外w：", percent=i / 2048)
            out_w[i] = np.load("/data/hongwu/result/cat_outjs/w_" + str(i) + ".npy")
        out_w_del = np.delete(np.reshape(out_w, [-1, outNum]), delete, 0)
        np.save("/data/hongwu/result/cat_outjs/out_w_del.npy", out_w_del)

    else:
        out_mid_result_del = np.load("/data/hongwu/result/cat_outjs/out_mid_result_del.npy")
        out_w_del = np.load("/data/hongwu/result/cat_outjs/out_w_del.npy")

    # print("-----------------------------------------计算协方差-----------------------------------------------------------")
    # 如果是同样图片，可重复利用，不用计算
    node_num = np.shape(mid_w_del)[0]
    if next_random:
        covarience = np.cov(np.log(mid_w_del))
        for i in range(node_num):
            np.save("/data/hongwu/result/cat_injs/covarience_0_" + str(i) + ".npy", covarience[i])

    # outlast_w = np.zeros(outNum, )
    # last_result = np.zeros((outNum, 100))
    # print("-----------------------------------------计算收益-------------------------------------------------------------")
    # for i in range(outNum):
    # last_result[i] = model.fc(model.avgpool(torch.Tensor(out_mid_result[i])).view((100, -1)))[:,
    # 281].detach().numpy()
    # last_stand = model.fc(model.avgpool(torch.Tensor(standard)).view((100, -1)))[:, 281].detach().numpy()
    # for i in range(outNum):
    #     outlast_w[i] = cal_w(last_result[i], last_stand)

    print("-----------------------------------------开始优化-------------------------------------------------------------")

    for layer in range(3):

        print("开始第", layer, "层优化！！！")
        pool = MyPool()
        for i in range(90):
            pool.apply_async(multi_fun, args=(function, i, mid_w_del, out_w_del, Erf, layer,))
        pool.close()
        pool.join()

        mid_result = np.zeros((20, 100, node_num))
        out_result = np.zeros((outNum, 100, node_num))

        for i in range(node_num):
            if layer == 0: break
            progress("读取第" + str(layer) + "层权重：", i / node_num)
            weight = np.load("/data/hongwu/result/weights/weight_" + str(layer) + "_" + str(i) + ".npy")
            union = np.load("/data/hongwu/result/weights/union_" + str(layer) + "_" + str(i) + ".npy")
            for j in range(len(union)):
                mid_result[:, :, i] += (cat_mid_result_del[:, :, union[j]] * weight[j])
                out_result[:, :, i] += (out_mid_result_del[:, :, union[j]] * weight[j])

        # print("读取下一层结果：")
        # mid_result = np.load("/data/hongwu/result/mid_" + str(layer) + ".npy")
        # out_result = np.load("/data/hongwu/result/out_" + str(layer) + ".npy")

        mid_result = np.maximum(mid_result, 0.0000001)
        out_result = np.maximum(out_result, 0.0000001)

        print("保存下一层结果：")
        for i in range(20):
            np.save("/data/hongwu/result/layer/mid_" + str(layer) + "_" + str(i) + ".npy", mid_result[i, :, :])
        for i in range(outNum):
            np.save("/data/hongwu/result/layer/out_" + str(layer) + "_" + str(i) + ".npy", out_result[i, :, :])

        np.save("/data/hongwu/result/mid_" + str(layer) + ".npy", mid_result)
        np.save("/data/hongwu/result/out_" + str(layer) + ".npy", out_result)

        pool = MyPool(19 + outNum)
        for i in range(19 + outNum):
            # print(i)
            progress(title="计算类内类外w距离", percent=i / (19 + outNum))
            pool.apply_async(cal_mid_and_out_w, args=(node_num, i, layer,))
        pool.close()
        pool.join()

        for i in range(19):
            mid_w_del[:, i] = np.load("/data/hongwu/result/layer/mid_w_layer_" + str(layer) + "_" + str(i) + ".npy")

        for i in range(outNum):
            out_w_del[:, i] = np.load("/data/hongwu/result/layer/out_w_layer_" + str(layer) + "_" + str(i) + ".npy")

        print("计算协方差......")
        covarience = np.cov(np.log(mid_w_del))
        for i in range(np.shape(covarience)[0]):
            progress(title="保存协方差", percent=i / np.shape(covarience)[0])
            np.save("/data/hongwu/result/cat_injs/covarience_" + str(layer + 1) + "_" + str(i) + ".npy", covarience[i])
        np.save("/data/hongwu/result/midw_" + str(layer) + ".npy", mid_w_del)
        np.save("/data/hongwu/result/outw_" + str(layer) + ".npy", out_w_del)

    return mid_w_del, out_w_del


def cal_mid_and_out_w(node_num, k, layer):
    if (layer == 0): return
    print("线程内：", k)
    try:
        standart = np.load("/data/hongwu/result/layer/mid_" + str(layer) + "_" + str(0) + ".npy")
        w = np.zeros(node_num)
        if k < 19:
            print(k)
            mid_result = np.load("/data/hongwu/result/layer/mid_" + str(layer) + "_" + str(k + 1) + ".npy")
            for i in range(node_num):
                progress(title="计算类内W：" + str(k), percent=i / node_num)
                w[i] = cal_w(mid_result[:, i], standart[:, i])
            print("类内计算完成存储：")
            np.save("/data/hongwu/result/layer/mid_w_layer_" + str(layer) + "_" + str(k) + ".npy", w)
        else:
            out_result = np.load("/data/hongwu/result/layer/out_" + str(layer) + "_" + str(k - 19) + ".npy")
            print(k)
            for i in range(node_num):
                progress(title="计算类外W：" + str(k), percent=i / node_num)
                w[i] = cal_w(out_result[:, i], standart[:, i])
            print("类外计算完成存储：")
            np.save("/data/hongwu/result/layer/out_w_layer_" + str(layer) + "_" + str(k - 19) + ".npy", w)

    except Exception as e:
        print(e)

    print("线程结束：", k)


def multi_fun(function, threadNum, mid_w_del, out_w_del, Erf, layer):
    # print(threadNum, "start")
    for i in range(threadNum * 1000, min((threadNum + 1) * 1000, np.shape(mid_w_del)[0])):
        progress(title="第" + str(threadNum) + "个线程进度： ", percent=(i - threadNum * 1000) / 1000)
        # print("layer: ", layer, "threadNum: ", threadNum, "start: ", i)
        function(i, mid_w_del, out_w_del, Erf, layer)
        # print("layer: ", layer, "threadNum: ", threadNum, "end: ", i)


def get_weight_1(node_index, mid_w_del, out_w_del, Erf, layer):
    if layer == 0: return
    global weight
    try:
        '''
            mid_w_del shape:  (83125, 19)
            mid_result_del: 20, 100, 83125
        '''

        node = np.load("/data/hongwu/result/cat_injs/covarience_" + str(layer) + "_" + str(node_index) + ".npy")
        sampleNum = 19

        nodeNum = np.shape(out_w_del)[0]
        out_sample_w = np.zeros((sampleNum, nodeNum))
        sNum = 10
        for i in range(sNum):
            out_sample_w += np.array(sample(list(out_w_del.T), sampleNum))
        out_sample_w = out_sample_w / sNum

        numbers = 100
        union = []
        thresh = np.median(out_sample_w[node_index]) / (np.median(mid_w_del[node_index]) + 0.00000001)

        node_ind = np.argsort(node)

        posi_node_ind = []
        neg_node_ind = []
        neg = True
        for nodein in node_ind:
            if neg:
                neg_node_ind.append(nodein)
            else:
                neg = False
                posi_node_ind.append(nodein)

        for i in range(np.shape(posi_node_ind)[0] - 1, 0, -1):
            if len(union) >= numbers / 2: break
            if np.median(out_sample_w[i]) / (np.median(mid_w_del[i]) + 0.00000001) >= thresh:
                union.append(i)

        for i in range(0, np.shape(neg_node_ind)[0] - 1):
            if len(union) >= numbers: break
            if np.median(out_sample_w[i]) / (np.median(mid_w_del[i]) + 0.00000001) >= thresh:
                union.append(i)

        # for i in (np.argpartition(node, -x)[-x:]):
        #     if np.median(mid_w_del[i]) <= thresh:
        #         union.append(i)
        # for i in (np.argpartition(node, x)[:x]):
        #     if np.median(mid_w_del[i]) <= thresh:
        #         union.append(i)

        x = len(union)
        if x <= 1:
            weight = np.zeros(np.shape(node)[0], )
            weight[node_index] = 1
            return weight

        one_reward = np.zeros((x, sampleNum))
        for i in range(x):
            for j in range(sampleNum):
                if mid_w_del[union[i]][j] == 0:
                    one_reward[i][j] = 100
                else:
                    one_reward[i][j] = np.log((out_sample_w[j][union[i]] + 0.000001) / (
                        mid_w_del[union[i]][j]) / np.var(mid_w_del[union[i]]) + 0.000001)

        V = np.cov(one_reward)
        Er = np.mean(one_reward, 1).reshape((x, 1))
        e = np.ones((x, 1))

        a = np.dot(np.dot(Er.T, np.linalg.inv(V)), Er)
        b = np.dot(np.dot(Er.T, np.linalg.inv(V)), e)
        c = np.dot(np.dot(e.T, np.linalg.inv(V)), e)
        # d = a * c - b * b
        #     print(a, b, c, d)
        A = np.dot(np.dot(np.hstack((Er, e)).T, np.linalg.inv(V)), np.hstack((Er, e)))
        miuP = (a * c - b * c * Erf) / (b * c - c * c * Erf)

        #     print(miuP)
        k = np.hstack((Er, e))
        c = np.dot(np.linalg.inv(V), k)
        d = np.dot(c, np.linalg.inv(A))
        w_star = np.dot(d, np.vstack((miuP, [1])))
        # weight = np.zeros((np.shape(node)[0],))
        # for index in range(np.shape(union)[0]):
        #     weight[union[index]] = w_star[index]
        # print(np.nonzero(weight))
        # print(weight)
        np.save("/data/hongwu/result/weights/weight_" + str(layer) + "_" + str(node_index) + ".npy", w_star)
        np.save("/data/hongwu/result/weights/union_" + str(layer) + "_" + str(node_index) + ".npy", union)

    except Exception as e:
        print(e)
    # return weight


if __name__ == '__main__':

    """2:00 ---       """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    # np.save("/data/hongwu/hongwu.npy", class_name)

    # torch.cuda.set_device(3)
    # for x in range(10000000000000):
    print("------------程序开始------------")
        # sleepawhile(2)
    # mkdirsds()
    # getNormal()
    # cal_wasserstein()
    # cw()
    # cal_weight(class_name[0])
    # pool_cal_w("n01697457", 3, "in")
    # getTrain()
    # getVal()
    # consData()
    train()
    # test()
        # inResult, outResult = Optimaze_1(get_weight_1, next_random=False)
    # np.save("/data/hongwu/inResult.npy", inResult)
    # np.save("/data/hongwu/outResult.npy", outResult)
    # pool = MyPool(40)
    # for thread_num in range(40):
    #     pool.apply_async(vali_js, args=(thread_num, ))
    # pool.close()
    # pool.join()
    # doubleSamT()
    # get_100_mid()
    #
    # get_final()
    # #
    # get_covari()
    #
    # # doubleSamT()
    # vali_js(10)
    # print("wings!!")
    #
    # cal_ori()
    # mp.set_start_method('spawn')

    # concepts = [snake, dog, leopard, fish, cat, mushroom, spider]
    # random_class = []
    # for concept in concepts:
    #     random_class.extend(concept)
    # random_class = sample(get_class_name(), 250)
    # print(random_class.index("n03903868"))
    # print(random_class)

    # count = 1
    # for i in range(np.shape(random_class)[0]):
    #     filename = random_class[i]
    #     if filename in cat[0]:
    #         break
    #     count += 1
    #     if count > 200:
    #         break
    #     #     # filename="n03982430"
    #     #     #
    #     if not os.path.exists("/data/imagenet_2012/train/" + filename):
    #         print("????")
    #         continue
    #     print(filename)
    # generate(filename)
    #     # test_01()
    #     # get_fftNode()
    #     # marker_A()
    #     # marker_B()
    #
    # marker_C(filename)
    #     # marker_D()
    #     #
    #     # test_hx()
    # with open(WORK_DIR + experience_id + "/result.txt", 'a+') as f:
    #     f.write(filename + "\n")
    # recall1, prec1, fpr1 = summary("result", class_name=filename)
    # recall2, prec2, fpr2 = get_precise("result", class_name=filename)
    # f11 = 2 * recall1 * prec1 / (recall1 + prec1)
    # f12 = 2 * recall2 * prec2 / (recall2 + prec2)
    #
    # temp = np.array((recall1, prec1, f11, fpr1, recall2, prec2, f12, fpr2))
    # np.save("/data/hongwu/result/rap_f1/" + filename + ".npy", temp)

    # model=torchvision.models.resnext50_32x4d(pretrained=True)
    # val_loader=data_loader("/data/imagenet_2012/", mode="val")
    # save_output(model, val_loader)

    # test_ori()
    # save_result()
    #    
    # get_D_Node(0.1)
    #
    # # train_dataloader = data_loader("/home/hongwu/tmp/pycharm_project_490/result/0/")
    # train_dataloader = data_loader(WORK_DIR+"/train/train/", mode="val")
    # model = torchvision.models.alexnet(pretrained=True)
    # criterion = torch.nn.CrossEntropyLoss()
    # node = np.ones((256, 6, 6), dtype=np.float32)
    # # node = np.load(WORK_DIR+"/marker_C/result/node_75.npy").astype(np.float32)
    # validate_js(train_dataloader, model, criterion, print_freq=10, node=node, tresh=1, thread_num=120, is_cuda=False)
    #
    # c:q:qlass_num = 100
    #
    #
    # # 计算中间输出并保存
    # #
    # start = datetime.datetime.now()
    # train_dir = WORK_DIR+"/train/train/"
    # make_dir(train_dir, class_num=10, image_num=50)
    #
    # print("====================================================================================")
    # print((datetime.datetime.now()-start).seconds)
    # print("====================================================================================")
    # data_imglist = np.zeros((10, 50))
    # cal_mid()
    # np.save(WORK_DIR+"/aug_data.npy", aug_data)
    # np.save(WORK_DIR+"/ori_data.npy", ori_data)
    #
    # duoxiancheng()
    #
    # get_node()
    # val_loader = data_loader(WORK_DIR+"/mid_result/1/")
    # model = torchvision.models.alexnet(pretrained=True)
    # node = np.load(WORK_DIR+"/result/nodes.npy")
    # nodes=np.zeros(np.shape(node)[1:])
    # for i in range(np.shape(node)[0]):
    #     nodes+=node[i]
    # # critBynodes(val_loader, model, node=None, tresh=8)
    # val_loader = data_loader(WORK_DIR+"/tiny_image/tiny/")
    # print(critBynodes(val_loader, model, nodes, 8, is_cuda=True))
    #
    # image = cv2.imread(WORK_DIR+"/train/train/n01484850/n01484850_21685.JPEG")
    # # print(image)
    # image = [image]
    # print(np.shape(image))
    # args=[]
    # args.append(image)
    # args.append(20)
    # args.append(1)
    # generate_coinT(args)
