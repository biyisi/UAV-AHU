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

QUERY_PATH ='./data/test'
QUERY_NAME_DEFINE = 'view'


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


def load_train_model(opt, RESNET18=False, RESNET152=True, VGG19=False):
    model, _, epoch = utils.load_network_teacher(opt.name, opt, RESNET18=RESNET18, RESNET152=RESNET152, VGG19=VGG19)
    # model.classifier.classifier = torch.nn.Sequential()
    model = model.eval()
    model = model.cuda()
    return model


def load_train_model_feature(opt, RESNET18=False, RESNET152=True, VGG19=False):
    model, _, epoch = utils.load_network_teacher(opt.name, opt, RESNET18=RESNET18, RESNET152=RESNET152, VGG19=VGG19)
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


def compare_label(model, dataloaders, query_labels):
    count = 0
    true_count = 0
    print("dataloaders =", dataloaders)
    for data in dataloaders:
        image_tensor, label = data
        n, c, h, w = image_tensor.size()
        input_image_tensor = torch.autograd.Variable(image_tensor.cuda())
        print("input_image_tensor.shape =", input_image_tensor.shape)
        outputs = model(input_image_tensor)
        outputs_query = torch.argmax(outputs, dim=1).cpu().numpy()
        data_query = query_labels[count: count + n]
        true_count += np.sum(outputs_query == np.array(data_query))
        # print(np.sum(outputs_query == np.array(data_query)))
        # query_labels取[count,count+n]
        count += n
        print("count =", count)
        print("true_count =", true_count)

    true_acc = true_count / count
    return true_acc
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


def save_to_txt(query_name, query_path, true_acc):
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


if __name__ == '__main__':
    opt = init_options()
    opt, str_ids, name, test_dir = init_load_train_config(opt)
    ms = choose_gpu(opt, str_ids)
    image_datasets, dataloaders, use_gpu = load_data(opt, test_dir)
    start_time = time.time()

    # Acc_statistics = False
    # Feature_Savemat = True

    Acc_statistics = True
    Feature_Savemat = False


    if Acc_statistics:
        # model = load_train_model(opt, RESNET18=False, RESNET152=True, VGG19=False)

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
            true_acc = compare_label(model, dataloaders[query_name], query_labels)
        save_to_txt(QUERY_NAME_DEFINE, os.path.curdir + "/data/student" + QUERY_NAME_DEFINE, true_acc)
        print(true_acc)
    elif Feature_Savemat:
        model = load_train_model_feature(opt, RESNET18=False, RESNET152=True, VGG19=False)
        query_name = QUERY_NAME_DEFINE
        query_labels, query_paths = get_labels_paths(image_datasets[query_name].imgs)
        with torch.no_grad():
            print("dataloaders[query_name] =", dataloaders[query_name])
            query_features = infer_feature(model, dataloaders[query_name], ms, opt, query_labels, query_paths)
        mat_name = QUERY_NAME_DEFINE + ".mat"
        result_mat = save_matlab(query_features=query_features, query_labels=query_labels, query_paths=query_paths,
                                 mat_name=mat_name)
    # with torch.no_grad():
    #     query_feature = extract_feature(model, dataloaders[query_name], ms, opt)
    # end_time = time.time()
    # total_time = end_time - start_time
    # print('Test complete in {:.0f}m {:.0f}s'.format(
    #     total_time // 60, total_time % 60))
    # result_mat = save_matlab(query_feature, query_labels, query_paths)
    # print(opt.name)
    # result = './model/%s/result.txt' % opt.name

# net_79.pth 0.8973227419829362
# net_124.pth(蒸馏) 0.8342159458664313
# net_139.pth(蒸馏) 0.8698146513680495
# net_144.pth(蒸馏) 0.8601059135039718
# net_184.pth(蒸馏) 0.8689320388349514