import os
import time
import pretrainedmodels
import torch
import torchvision
import yaml
import math
import matplotlib
import scipy.io
import argparse
import numpy as np

from apex.fp16_utils import *
from model import view_net
import utils

QUERY_PATH = './data/test'
QUERY_NAME_DEFINE = 'label_view'
# QUERY_NAME_DEFINE = 'view_all'


def init_options():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
    parser.add_argument('--test_dir', default=QUERY_PATH, type=str, help='./test_data')
    parser.add_argument('--name', default='view', type=str, help='save model path')
    parser.add_argument('--pool', default='avg', type=str, help='avg|max')
    parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
    parser.add_argument('--h', default=384, type=int, help='height')
    parser.add_argument('--w', default=384, type=int, help='width')
    parser.add_argument('--views', default=2, type=int, help='views')
    parser.add_argument('--use_dense', action='store_true', help='use densenet')
    parser.add_argument('--PCB', action='store_true', help='use PCB')
    parser.add_argument('--multi', action='store_true', help='use multiple query')
    parser.add_argument('--fp16', action='store_true', help='use fp16.')
    parser.add_argument('--ms', default='1', type=str, help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
    opt = parser.parse_args()
    return opt


def init_load_train_config(opt):
    config_path = os.path.join('./model', opt.name, 'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream)
    opt.fp16 = config['fp16']
    opt.use_dense = config['use_dense']
    opt.use_NAS = config['use_NAS']
    opt.stride = config['stride']
    opt.views = config['views']  # 这里view已经没用了
    if 'h' in config:
        opt.h = config['h']
        opt.w = config['w']
    if 'nclasses' in config:
        opt.nclasses = config['nclasses']
    else:
        opt.nclasses = 21

    str_ids = opt.gpu_ids.split(',')
    name = opt.name  # view
    test_dir = opt.test_dir
    return opt, str_ids, name, test_dir


def choose_gpu(opt, str_ids):
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)

    print('We use the scale: %s' % opt.ms)
    str_ms = opt.ms.split(',')
    ms = []
    for s in str_ms:
        s_f = float(s)
        ms.append(math.sqrt(s_f))

    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        torch.backends.cudnn.benchmark = True
    return ms


def load_data(opt, test_dir):
    data_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((opt.h, opt.w), interpolation=3),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data_dir = test_dir
    image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in
                      [QUERY_NAME_DEFINE]}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=False, num_workers=16) for x in
                   [QUERY_NAME_DEFINE]}
    use_gpu = torch.cuda.is_available()
    return image_datasets, dataloaders, use_gpu


def flip_horizontal(img):
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def get_labels_paths(img_path):
    camera_id = []
    labels = []
    paths = []
    for path, v in img_path:
        # print("path =", path)
        # print("v =", v)
        paths.append(path)
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
    return labels, paths


def load_train_model(opt, RESNET101=False, RESNET152=True, VGG19=False, VGG16=False):
    model, _, epoch = utils.load_network_teacher(opt.name, opt, RESNET101=RESNET101, RESNET152=RESNET152, VGG19=VGG19, VGG16=VGG16)
    # model.classifier.classifier = torch.nn.Sequential()
    model = model.eval()
    model = model.cuda()
    return model


def load_train_model_feature(opt, RESNET101=False, RESNET152=True, VGG19=False, VGG16=False):
    model, _, epoch = utils.load_network_teacher(opt.name, opt, RESNET101=RESNET101, RESNET152=RESNET152, VGG19=VGG19, VGG16=VGG16)
    model.classifier.classifier = torch.nn.Sequential()
    model = model.eval()
    model = model.cuda()
    return model


def save_matlab(query_features: torch.FloatTensor, query_labels: list, query_paths: list, mat_name: str):
    result_mat = {'query_feature': query_features.numpy(), 'query_label': query_labels, 'query_path': query_paths}
    scipy.io.savemat(mat_name, result_mat)
    return result_mat


def extract_feature(model, dataloaders, ms, opt):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        image, label = data
        n, c, h, w = image.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n, 512).zero_().cuda()
        for i in range(2):
            if (i == 1):
                image = flip_horizontal(image)
            input_image = torch.autograd.Variable(image.cuda())
            # print("input_image.shape =", input_image.shape)
            # time.sleep(1000)
            for scale in ms:
                outputs = model(input_image)
                ff += outputs
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff.data.cpu()), 0)
    return features


'''
    TP：(true_label == x && pre_label == x)
    FP: (true_label != x && pre_label == x)
    TN: (true_label != x && pre_label != x)
    FN: (true_label == x && pre_label != x)
'''


def compare_label(model, dataloaders, query_labels):
    count = 0
    true_count = 0
    print("dataloaders =", dataloaders)
    TP = [0 for i in range(0, 21)]
    FP = [0 for i in range(0, 21)]
    TN = [0 for i in range(0, 21)]
    FN = [0 for i in range(0, 21)]
    P = [0 for i in range(0, 21)]
    R = [0 for i in range(0, 21)]
    F1 = [0 for i in range(0, 21)]
    for data in dataloaders:
        image_tensor, label = data
        n, c, h, w = image_tensor.size()
        input_image_tensor = torch.autograd.Variable(image_tensor.cuda())
        print("input_image_tensor.shape =", input_image_tensor.shape)
        outputs = model(input_image_tensor)
        pre_label = torch.argmax(outputs, dim=1).cpu().numpy()
        true_label = query_labels[count: count + n]

        TP = [(TP[i] + np.sum((pre_label == i) & (np.array(true_label) == i))) for i in range(0, 21)]
        FP = [(FP[i] + np.sum((pre_label == i) & (np.array(true_label) != i))) for i in range(0, 21)]
        TN = [(TN[i] + np.sum((pre_label != i) & (np.array(true_label) != i))) for i in range(0, 21)]
        FN = [(FN[i] + np.sum((pre_label != i) & (np.array(true_label) == i))) for i in range(0, 21)]

        true_count += np.sum(pre_label == np.array(true_label))
        # print(np.sum(outputs_query == np.array(data_query)))
        # query_labels取[count,count+n]
        count += n
        print("count =", count)
        print("true_count =", true_count)

    P = [(TP[i] / (TP[i] + FP[i])) for i in range(0, 21)]
    R = [(TP[i] / (TP[i] + FN[i])) for i in range(0, 21)]
    F1 = [((2 * P[i] * R[i]) / (R[i] + P[i])) for i in range(0, 21)]

    tmp_p = [P[i] for i in range(0, 21) if (P[i] > 1e-6)]
    # label_count = len([P[i] for i in range(0, 21) if (P[i] > 1e-6)])
    label_count = len(tmp_p)
    P_avg = sum(tmp_p) / label_count

    tmp_r = [R[i] for i in range(0, 21) if (R[i] > 1e-6)]
    label_count = len(tmp_r)
    R_avg = sum(tmp_r) / label_count

    tmp_f1 = [F1[i] for i in range(0, 21) if (F1[i] > 1e-6)]
    label_count = len(tmp_f1)
    F1_avg = sum(tmp_f1) / label_count

    # print(label_count, P_avg, R_avg, F1_avg)

    true_acc = true_count / count
    return true_acc, P_avg, R_avg, F1_avg
    # print("torch.argmax(outputs, dim=1).cpu().numpy() =", torch.argmax(outputs, dim=1).cpu().numpy())
    # print("outputs.shape =", outputs.shape)
    # print("query_labels =", query_labels)
    # print("image_tensor =", image_tensor)
    # print("label =", label)
    # print("n =", n)
    # print("c =", c)
    # print("h =", h)
    # print("w =", w)
    # print("count =", count)
    # break

    #
    #
    # pass


def save_to_txt(query_name, query_path, true_acc, P_avg, R_avg, F1_avg):
    file = open('AHU.txt', mode="a+")
    file.writelines("query_name: ")
    file.writelines(query_name)
    file.writelines("; ")
    file.writelines("query_path: ")
    file.writelines(query_path)
    file.writelines("; ")
    file.writelines("true_acc: ")
    file.writelines(str(true_acc))
    file.writelines("; ")
    file.writelines("P_avg: ")
    file.writelines(str(P_avg))
    file.writelines("; ")
    file.writelines("R_avg: ")
    file.writelines(str(R_avg))
    file.writelines("; ")
    file.writelines("F1_avg: ")
    file.writelines(str(F1_avg))
    file.writelines("; ")
    file.writelines('\n')


def infer_feature(model, dataloaders, ms, opt, query_labels, query_paths):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        image_tensor, label = data
        n, c, h, w = image_tensor.size()
        count += n
        print("count =", count)
        ff = torch.FloatTensor(n, 512).zero_().cuda()
        for i in range(2):
            if (i == 1):
                image_tensor = flip_horizontal(image_tensor)
            input_image_tensor = torch.autograd.Variable(image_tensor.cuda())
            for scale in ms:
                outputs_tensor = model(input_image_tensor)
                ff += outputs_tensor
        # input_image_tensor = torch.autograd.Variable(image_tensor.cuda())
        # outputs_tensor = model(input_image_tensor)
        # ff += outputs_tensor

        # torch.norm(input, p, dim, keepdim)
        # input: 输入的tensor, p: 范数的维度，p=2表示二维距离, dim表示在哪一维度上计算, keepdim表示计算后不丢弃原有维度
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        # print("ff.shape =", ff.shape)
        # print("fnorm.shape =", fnorm.shape)
        # time.sleep(1000)

        # ff = ff.div(fnorm.expand_as(ff))
        features_tensor = torch.div(ff, fnorm.expand_as(ff))
        features = torch.cat((features, features_tensor.data.cpu()), dim=0)
    # count*out_features(512)
    return features


def load_student_kd_model_feature(name, opt):
    model, _, epoch = utils.load_network_student(opt.name, opt)
    model.classifier.classifier = torch.nn.Sequential()
    model = model.eval()
    model = model.cuda()
    return model


if __name__ == '__main__':
    opt = init_options()
    opt, str_ids, name, test_dir = init_load_train_config(opt)
    ms = choose_gpu(opt, str_ids)
    image_datasets, dataloaders, use_gpu = load_data(opt, test_dir)
    start_time = time.time()

    Acc_statistics = False
    Feature_Savemat = True

    # Acc_statistics = True
    # Feature_Savemat = False

    if Acc_statistics:
        # model = load_train_model(opt, RESNET101=False, RESNET152=True, VGG19=False, VGG16=False)

        model, _, _ = utils.load_network_student(opt.name, opt)
        model = model.eval()
        model = model.cuda()

        query_name = QUERY_NAME_DEFINE
        # print("image_datasets[drone].imgs =", image_datasets["drone"].imgs)
        query_labels, query_paths = get_labels_paths(image_datasets[query_name].imgs)
        # torch.no_grad():
        #     用于停止autograd模块的工作，起到加速和节省显存的作用（具体行为就是停止gradient计算，从而节省了GPU算力和显存）
        #     不会影响 dropout 和 batchnorm 层的行为
        with torch.no_grad():
            print("dataloaders[query_name] =", dataloaders[query_name])
            true_acc, P_avg, R_avg, F1_avg = compare_label(model, dataloaders[query_name], query_labels)
        # save_to_txt(QUERY_NAME_DEFINE, os.path.curdir + "/data/teacher" + QUERY_NAME_DEFINE, true_acc, P_avg, R_avg,
        #             F1_avg)
        save_to_txt(QUERY_NAME_DEFINE, os.path.curdir + "/data/student" + QUERY_NAME_DEFINE, true_acc, P_avg, R_avg,
                    F1_avg)
        print("true_acc=%s, P_avg=%s, R_avg=%s, F1_avg=%s" % (true_acc, P_avg, R_avg, F1_avg))
    elif Feature_Savemat:
        # model = load_train_model_feature(opt, RESNET101=False, RESNET152=True, VGG19=False, VGG16=False)
        model = load_student_kd_model_feature(opt.name, opt)
        print(model)
        query_name = QUERY_NAME_DEFINE
        query_labels, query_paths = get_labels_paths(image_datasets[query_name].imgs)
        with torch.no_grad():
            print("dataloaders[query_name] =", dataloaders[query_name])
            query_features = infer_feature(model, dataloaders[query_name], ms, opt, query_labels, query_paths)
        # mat_name = "CNN_100+_6798_"+QUERY_NAME_DEFINE + ".mat"
        # mat_name = "logits_159_" + QUERY_NAME_DEFINE + ".mat"
        # mat_name = "test_conv_159_"+QUERY_NAME_DEFINE + ".mat"
        mat_name = "test_4CNN_159_200+train_" + QUERY_NAME_DEFINE + ".mat"
        result_mat = save_matlab(query_features=query_features, query_labels=query_labels, query_paths=query_paths,
                                 mat_name=mat_name)
        os.system("python evaluate_gpu_1_ctx.py")
    # with torch.no_grad():
    #     query_feature = extract_feature(model, dataloaders[query_name], ms, opt)
    # end_time = time.time()
    # total_time = end_time - start_time
    # print('Test complete in {:.0f}m {:.0f}s'.format(
    #     total_time // 60, total_time % 60))
    # result_mat = save_matlab(query_feature, query_labels, query_paths)
    # print(opt.name)
    # result = './model/%s/result.txt' % opt.name

'''
net_7CNN.pth:
true_acc=0.9371069182389937, P_avg=0.9417744996549344, R_avg=0.9364818295739348, F1_avg=0.9368847257389515
Recall@1:93.71 Recall@5:96.86 Recall@10:96.86 Recall@top1:96.86 AP:84.04

net_res_18.pth
true_acc=0.949685534591195, P_avg=0.9567070158102766, R_avg=0.9499060150375939, F1_avg=0.9500847377836362
Recall@1:98.74 Recall@5:100.00 Recall@10:100.00 Recall@top1:100.00 AP:81.99
'''


'''
net_100+.pth:
true_acc=0.7044025157232704, P_avg=0.7655662454375689, R_avg=0.7019893483709273, F1_avg=0.7068251063951133
Recall@1:78.62 Recall@5:91.19 Recall@10:94.34 Recall@top1:89.94 AP:37.97

net_200+.pth:
true_acc=0.7735849056603774, P_avg=0.7953907852484078, R_avg=0.7733709273182958, F1_avg=0.77171601230715
Recall@1:88.68 Recall@5:96.86 Recall@10:98.74 Recall@top1:94.97 AP:42.30

net_300+.pth:
true_acc=0.8238993710691824, P_avg=0.8631528850278851, R_avg=0.8272556390977444, F1_avg=0.8306750427493462
Recall@1:89.31 Recall@5:97.48 Recall@10:98.74 Recall@top1:93.08 AP:44.82

net_400+.pth:
true_acc=0.8930817610062893, P_avg=0.9069183375104428, R_avg=0.89484649122807, F1_avg=0.8948964106618149
Recall@1:91.82 Recall@5:96.23 Recall@10:100.00 Recall@top1:95.60 AP:45.61



net_999.pth(ST):
true_acc=0.9622641509433962, P_avg=0.9651848866236739, R_avg=0.962468671679198, F1_avg=0.9625454739681387
Recall@1:92.45 Recall@5:98.11 Recall@10:100.00 Recall@top1:97.48 AP:54.09
'''



'''
net_800.pth(6-layer 2-conv 2-pool 2-fc CNN)
true_acc=0.8679245283018868, P_avg=0.8917921335200747, R_avg=0.8701441102756892, F1_avg=0.8710413855150698
Recall@1:89.31 Recall@5:96.86 Recall@10:98.74 Recall@top1:93.08 AP:45.62

net_900.pth(7-layer 3-conv 2-pool 2-fc CNN)
true_acc=0.8867924528301887, P_avg=0.8945833333333334, R_avg=0.8870457393483709, F1_avg=0.8851527584365113
Recall@1:88.05 Recall@5:95.60 Recall@10:98.11 Recall@top1:94.34 AP:42.96
'''
'''
Teacher at view_all: 6798 pictures
true_acc=0.9967637540453075, P_avg=0.9967660983367029, R_avg=0.9968275856403586, F1_avg=0.9967833504280225
Recall@1:99.93 Recall@5:99.93 Recall@10:99.93 Recall@top1:99.94 AP:99.37

net_101.pth
true_acc=0.9451309208590762, P_avg=0.9478910924572184, R_avg=0.9444067987859067, F1_avg=0.9446860176205463
Recall@1:99.88 Recall@5:99.96 Recall@10:99.96 Recall@top1:100.00 AP:65.89

net_152.pth
true_acc=0.9582230067666961, P_avg=0.9594975762823902, R_avg=0.9576663379779711, F1_avg=0.9575972875413324
Recall@1:99.93 Recall@5:99.99 Recall@10:99.99 Recall@top1:100.00 AP:74.69

net_116.pth
true_acc=0.9720506031185643, P_avg=0.9747165303664447, R_avg=0.9715782843016385, F1_avg=0.9725558789937051
Recall@1:99.59 Recall@5:99.81 Recall@10:99.91 Recall@top1:99.97 AP:93.86

net_119.pth
true_acc=0.9571932921447485, P_avg=0.9614941984839452, R_avg=0.9563725410752164, F1_avg=0.9572984997364173
Recall@1:99.65 Recall@5:99.88 Recall@10:99.91 Recall@top1:99.97 AP:88.74

# 2CNN_200+train_st
net_1459.pth
Recall@1:94.34 Recall@5:97.48 Recall@10:98.11 Recall@top1:97.48 AP:50.68
true_acc=0.9308176100628931, P_avg=0.9442971380471381, R_avg=0.9318139097744361, F1_avg=0.9326391129063677

# 2CNN_200+train_logits
net_269.pth
Recall@1:91.19 Recall@5:97.48 Recall@10:98.74 Recall@top1:96.86 AP:53.39
true_acc=0.8616352201257862, P_avg=0.8688042186571598, R_avg=0.8619517543859648, F1_avg=0.8623239082694406

# 2CNN_300+train_st
net_619.pth
Recall@1:94.34 Recall@5:98.11 Recall@10:98.74 Recall@top1:98.11 AP:51.17
true_acc=0.9308176100628931, P_avg=0.9410943223443223, R_avg=0.9324404761904762, F1_avg=0.9317752866864745

'''






# label_view: 159 pictures
# net_50.pth(Teacher):
# true_acc=0.9937106918238994, P_avg=1.0, R_avg=0.9940476190476191, F1_avg=0.9969512195121951 Recall@1:100.00
# net_200.pth(Student):
# true_acc=0.8364779874213837, P_avg=0.8517617072572737, R_avg=0.8397556390977443, F1_avg=0.8363099058622164 Recall@1:86.16
# net_400.pth(ST):
# true_acc=0.9308176100628931, P_avg=0.9386320915926178, R_avg=0.9321115288220552, F1_avg=0.9308368379206813 Recall@1:89.94
# net_600.pth(Logits):
# true_acc=0.8930817610062893, P_avg=0.9039468794617536, R_avg=0.8951754385964912, F1_avg=0.8930194424187559 Recall@1:86.79
#
# net_300+(300+ train)
# true_acc=0.8238993710691824, P_avg=0.8290762186981816, R_avg=0.8234335839598997, F1_avg=0.8222807991435552
# Recall@1:84.28 Recall@5:93.08 Recall@10:96.86 Recall@top1:90.57 AP:39.93
#
# net_200+(200+ train)
# true_acc=0.7547169811320755, P_avg=0.7704408212560386, R_avg=0.7530388471177945, F1_avg=0.7544683068541016
# Recall@1:83.65 Recall@5:91.19 Recall@10:97.48 Recall@top1:88.68 AP:39.16
#
# net_100+(100+ train)
# true_acc=0.5849056603773585, P_avg=0.6335807656395892, R_avg=0.5855889724310778, F1_avg=0.5875416883189757
# Recall@1:78.62 Recall@5:90.57 Recall@10:94.34 Recall@top1:87.42 AP:37.03


'''训练集449张图片，知识蒸馏方法：SoftTarget。 温度：4，Alpha：0.85。测试集准确率如下'''
# net_200.pth(Student) 0.850412249705536
# net_400.pth(BestKD) 0.9128386336866903
# net_50.pth(Teacher) 0.9956811935610522
# net_600.pth(Logits) 0.8657243816254417

'''使用含有8个label的全部测试图片组成的测试集进行测试的结果，特征提取对比'''
# teacher_len = 2547
# round(teacher_len * 0.01) = 25
# Recall@1:99.96 Recall@5:99.96 Recall@10:99.96 Recall@top1:100.00 AP:99.58
# student_len = 2547
# round(student_len * 0.01) = 25
# Recall@1:98.19 Recall@5:99.29 Recall@10:99.57 Recall@top1:99.80 AP:43.12
# kd_len = 2547
# round(kd_len * 0.01) = 25
# Recall@1:99.06 Recall@5:99.61 Recall@10:99.73 Recall@top1:99.84 AP:50.92

'''裁减后测试集label_view, 159张测试集照片'''
# net_200.pth(Student) 0.8364779874213837
# net_400.pth(BestKD) 0.9308176100628931
# net_50.pth(Teacher) 0.9937106918238994
# net_600.pth(Logits) 0.8930817610062893

# resnet_18(400 epoch)
# true_acc=0.9559748427672956, P_avg=0.9589348866236738, R_avg=0.9558897243107769, F1_avg=0.9557742796969445
# Recall@1:96.23 Recall@5:97.48 Recall@10:97.48 Recall@top1:96.86 AP:87.23


# net_res18.pth
# true_acc=0.9433962264150944, P_avg=0.9482142857142858, R_avg=0.9461309523809525, F1_avg=0.9426203877423389
# Recall@1:100.00 Recall@5:100.00 Recall@10:100.00 Recall@top1:100.00 AP:86.59


'''使用含有8个label的裁减后测试集label_view, 159张测试集照片进行测试的结果，特征提取对比'''
# teacher_len = 159
# round(teacher_len * 0.01) = 2
# Recall@1:100.00 Recall@5:100.00 Recall@10:100.00 Recall@top1:100.00 AP:95.58
# student_len = 159
# round(student_len * 0.01) = 2
# Recall@1:86.16 Recall@5:93.71 Recall@10:95.60 Recall@top1:93.08 AP:41.71
# kd_len = 159
# round(kd_len * 0.01) = 2
# Recall@1:89.94 Recall@5:96.23 Recall@10:98.11 Recall@top1:94.97 AP:47.57
# logits_len = 159
# round(logits_len * 0.01) = 2
# Recall@1:86.79 Recall@5:95.60 Recall@10:96.23 Recall@top1:93.71 AP:43.73


# net_500.pth(Logits,error):
# true_acc=0.9182389937106918, P_avg=0.927237477752352, R_avg=0.9205043859649124, F1_avg=0.9186471981722303
# Recall@1:91.19 Recall@5:96.23 Recall@10:98.74 Recall@top1:94.97 AP:47.52

'''
# 2CNN_200+train_st
net_215.pth
Recall@1:93.08 Recall@5:97.48 Recall@10:98.74 Recall@top1:96.86 AP:49.57
true_acc=0.9308176100628931, P_avg=0.9396135265700483, R_avg=0.9311873433583959, F1_avg=0.9308085554796081

# 2CNN_200+train_logits
net_269.pth
Recall@1:91.19 Recall@5:97.48 Recall@10:98.74 Recall@top1:96.86 AP:53.39
true_acc=0.8616352201257862, P_avg=0.8688042186571598, R_avg=0.8619517543859648, F1_avg=0.8623239082694406



# 2CNN_300+train_st
net_619.pth
Recall@1:94.34 Recall@5:98.11 Recall@10:98.74 Recall@top1:98.11 AP:51.17
true_acc=0.9308176100628931, P_avg=0.9410943223443223, R_avg=0.9324404761904762, F1_avg=0.9317752866864745

# 2CNN_300+train_logits
net_364.pth
Recall@1:93.71 Recall@5:97.48 Recall@10:99.37 Recall@top1:96.86 AP:55.20
true_acc=0.8930817610062893, P_avg=0.9056745398850663, R_avg=0.894251253132832, F1_avg=0.8942368157881782



# 2CNN_400+train_st
net_499.pth
Recall@1:92.45 Recall@5:98.11 Recall@10:100.00 Recall@top1:97.48 AP:54.09
true_acc=0.9622641509433962, P_avg=0.9651848866236739, R_avg=0.962468671679198, F1_avg=0.9625454739681387

# 2CNN_400+train_logits
net_466.pth
Recall@1:94.34 Recall@5:98.11 Recall@10:98.74 Recall@top1:96.23 AP:56.40
true_acc=0.9182389937106918, P_avg=0.9251875668789116, R_avg=0.9199091478696743, F1_avg=0.9182240427650463
'''