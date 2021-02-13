# import threading
# 来，说句话0.0
# import requests
from warnings import simplefilter

from chonggou_1 import *
import warnings

# import multiprocessing as mp

# warnings.filterwarnings('ignore')
# simplefilter(action='ignore', category=FutureWarning)

WORK_DIR = "/home/hongwu/python/Image/"
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
    image_num = 20
    augnum = 100

    # model = torchvision.models.alexnet(pretrained=True)

    if not os.path.exists(WORK_DIR+experience_id+filename+"/marker_C/val"):
        os.makedirs(WORK_DIR+experience_id+filename+"/marker_C/val/")
    # shutil.copytree(WORK_DIR+"/val/"+filename, WORK_DIR+experience_id+filename+"/marker_C/val/"+filename)
    #
    make_dir(train_dir=WORK_DIR+"/train/train/val/", filename=filename, class_num=1, image_num=image_num, save_dir=WORK_DIR + experience_id + filename +
                                                                                             "/marker_C"
                                                                                             "/result/")
    for i in range(image_num):
        if not os.path.exists(WORK_DIR + experience_id + filename + "/marker_C/" + str(i) + "/val/class/"):
            os.makedirs(WORK_DIR + experience_id + filename + "/marker_C/" + str(i) + "/val/class/")

    pool = MyPool()
    for i in range(image_num):
        pool.apply_async(generate_coinT, args=(
            augnum, i, WORK_DIR + experience_id + filename + "/marker_C/", WORK_DIR + experience_id + filename + "/marker_C/",))
    pool.close()
    pool.join()

    pool = MyPool()
    for i in range(image_num):
        # def augMidres(index, model, augnum, shape, data_dir="/home/hongwu/python/Image/exper_v1/marker_C/"):
        pool.apply_async(augMidres,
                         args=(i, model, augnum, shape, WORK_DIR + experience_id + filename + "/marker_C/", False,))
    pool.close()
    pool.join()

    # 读取数据，维度为[200， 1000， 256. 6. 6]
    data_shape = np.concatenate(([image_num], [augnum], shape))
    data = np.zeros(data_shape)

    for i in range(image_num):
        progress(i / image_num)
        data[i] = np.load(WORK_DIR + experience_id + filename + "/marker_C/result/mid_result_" + str(i) + ".npy")

    if not os.path.exists(WORK_DIR + experience_id + filename + "/marker_C/result/xulie"):
        os.makedirs(WORK_DIR + experience_id + filename + "/marker_C/result/xulie")

    # 获取原始文件的中间结果，并保存为xulie_0_0
    # num = get_n_mid(data_loader(WORK_DIR + experience_id+"data/1000", mode="val"),
    #                 model=torchvision.models.alexnet(pretrained=True), layer=5)
    # np.save(WORK_DIR + experience_id + "/marker_C/result/xulie/xulie_0_0.npy", num)
    # print("origin数据计算完毕！！")

    # 将结果分成1000个文件，然后再进行求0阶和1阶js序列。
    for i in range(augnum):
        progress(i / augnum)
        np.save(WORK_DIR + experience_id + filename + "/marker_C/result/xulie/xulie_" + str(i + 1) + "_0.npy",
                data[:, i, :, :, :])
        np.save(WORK_DIR + experience_id + filename + "/marker_C/result/xulie/xulie_" + str(i + 1) + "_1.npy",
                data[:, i, :, :, :])

    # 首先计算0阶js序列
    # if not os.path.exists(WORK_DIR+experience_id+"/marker_C/result/js_0/"):
    #     os.makedirs(WORK_DIR+experience_id+"/marker_C/result/js_0/")
    # pool = MyPool()
    # for thread_num in range(100):
    #     pool.apply_async(cal_js, args=(thread_num + 1, 0, 0,))
    # pool.close()
    # pool.join()

    # 计算1阶js序列
    if not os.path.exists(WORK_DIR + experience_id + filename + "/marker_C/result/js_1/"):
        os.makedirs(WORK_DIR + experience_id + filename + "/marker_C/result/js_1/")
    pool = MyPool()
    for thread_num in range(augnum - 1):
        pool.apply_async(cal_js, args=(thread_num + 2, filename, thread_num + 1, 1,))
    pool.close()
    pool.join()

    # # 通过横向对比得出方差，最后找出稳定性较好的点，进行筛选
    mid_result = np.zeros(np.concatenate((image_num, augnum, shape)))
    for i in range(image_num):
        progress(i / image_num)
        mid_result[i] = np.load(WORK_DIR + experience_id + "/marker_C/result/mid_result_" + str(i) + ".npy")

    pool_1 = MyPool()
    for thread_num in range(shape[0]):
        pool_1.apply_async(cal_2_js, args=(mid_result[:, :, thread_num, :, :], thread_num,))
    pool_1.close()
    pool_1.join()

    js = np.zeros(np.concatenate((shape, 49)))
    for i in range(shape[0]):
        progress(i / shape[0])
        js[i] = np.load(WORK_DIR + experience_id + "/marker_C/result/js_" + str(i) + "_of256.npy")
    js_std = np.std(js, axis=3)
    print("js数据加载完毕")

    # js = np.mean(js, axis=3)

    # pool_2 = MyPool()
    # for thread_num in range(10):
    #     pool_2.apply_async(multi_get_acc, args=(thread_num, js_std,))
    # pool_2.close()
    # pool_2.join()

    # for degree in range(2):
    degree = 1
    # node = np.zeros((256, 6, 6))
    node = np.zeros(shape)
    js_0 = np.zeros(np.concatenate(([augnum], shape)))
    for i in range(augnum - 1):
        progress(i / augnum)
        js_0[i] = np.load(
            WORK_DIR + experience_id + filename + "/marker_C/result/js_" + str(degree) + "/js_" + str(i + 2) + ".npy")

    js_std = np.std(js_0, axis=0)
    for epo in range(10):
        js_tresh = np.percentile(js_std, 5 + epo * 5)
        # js_tresh = np.mean(js_std) - np.std(js_std)

        for i in range(shape[0]):
            progress(i / shape[0])
            for j in range(shape[1]):
                for k in range(shape[2]):
                    if js_std[i][j][k] < js_tresh:
                        node[i, j, k] = 1
        np.save(WORK_DIR + experience_id + filename + "/w_node" + str(5 + epo * 5) + ".npy", node)


def marker_D():
    # 通过计算其0阶或者1阶js是否随着干预有规律变化，筛选出marker_D的节点

    js_1_test = np.zeros((1000, 256, 6, 6))
    for i in range(1000):
        progress(i / 1000)
        js_1_test[i] = np.load(WORK_DIR + "/marker_C/result/js_1/js_" + str(i + 1) + ".npy")
    pool_dw = MyPool()
    for i in range(256):
        # progress(i / 256)
        pool_dw.apply_async(granger_test, args=(js_1_test[:, i, :, :], i,))
        # pool_dw.apply_async(dw_stat_6_6, args=(js_1_test[:, i, :, :], i,))
    pool_dw.close()
    pool_dw.join()


def summary(filename, class_name):
    pool = MyPool()
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
    for j in range(10):
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
            WORK_DIR + experience_id + class_name + "/w_node" + str(5 + 5 * j) + ".npy").astype(np.int8)

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
        print(j * 10, "==== ", node.mean())

        # loader = data_loader(root=WORK_DIR + "/", mode="val",
        #                      pin_memory=False)

        loader = data_loader(root="/data/imagenet_2012/", mode="val", pin_memory=True)
        # loader = data_loader(root=WORK_DIR, mode="val", pin_memory=False)

        # def validate_js(val_loader, model, criterion, print_freq, node, tresh, thread_num,
        # is_cuda=False)
        pool.apply_async(validate_js, args=(loader, model, criterion,
                                            print_freq, node, 1,
                                            j * 10, filename, class_name, True))
        # validate_js(loader, model, criterion, print_freq, node, 1, 0, True)
    pool.close()
    pool.join()


def get_precise(filename, class_name):
    node = np.ones(shape)
    criterion = torch.nn.CrossEntropyLoss()
    print_freq = 1000

    # loader = data_loader(root=WORK_DIR + "/", mode="val",
    #                      pin_memory=False)

    loader = data_loader(root=WORK_DIR + experience_id + class_name + "/marker_C/", mode="val", pin_memory=False,
                         batch_size=16)
    # loader = data_loader(root=WORK_DIR, mode="val", pin_memory=False)

    # def validate_js(val_loader, model, criterion, print_freq, node, tresh, thread_num,
    # is_cuda=False)
    # beita=np.load(WORK_DIR+experience_id+class_name+"/beita.npy")
    validate_js(loader, model, criterion, print_freq, node, 1, 0, filename, class_name, False)
    # validate_js(loader, model, criterion, print_freq, beita, 1, 0, filename, class_name, False)


if __name__ == '__main__':
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
    for filename in random_class:
    # filename="n03982430"
    #
        if not os.path.exists(WORK_DIR+"/train/train/val/" + filename):
            print("????")
            continue
        print("111")
    # generate(filename)
    # test_01()
    # get_fftNode()
    # marker_A()
    # marker_B()

        marker_C(filename)
    # marker_D()
    #
    # test_hx()
        with open(WORK_DIR + experience_id + "/w_snake_gauss.txt", 'a+') as f:
            f.write(filename + "\n")
        summary("w_snake_gauss", class_name=filename)

        get_precise("w_snake_gauss", class_name=filename)

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
