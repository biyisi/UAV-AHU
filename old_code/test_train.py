# -*- coding: utf-8 -*-

from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from model import view_net
from utils import load_network
from apex.fp16_utils import *


def init_options():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
    parser.add_argument('--test_dir', default='./data/test_pure', type=str, help='./test_data')
    parser.add_argument('--name', default='view', type=str, help='save model path')
    parser.add_argument('--pool', default='avg', type=str, help='avg|max')
    parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
    parser.add_argument('--h', default=256, type=int, help='height')
    parser.add_argument('--w', default=256, type=int, help='width')
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
    opt.views = config['views']
    if 'h' in config:
        opt.h = config['h']
        opt.w = config['w']

    if 'nclasses' in config:  # tp compatible with old config files
        opt.nclasses = config['nclasses']
    else:
        opt.nclasses = 729
    str_ids = opt.gpu_ids.split(',')
    # which_epoch = opt.which_epoch
    name = opt.name
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
        cudnn.benchmark = True
    return ms


def load_data(opt, test_dir):
    data_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data_dir = test_dir
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in
                      ['gallery_satellite', 'gallery_drone', 'gallery_street', 'query_satellite', 'query_drone',
                       'query_street']}
    # print("image_datasets=", image_datasets)
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=False, num_workers=16) for x in
                   ['gallery_satellite', 'gallery_drone', 'gallery_street', 'query_satellite', 'query_drone',
                    'query_street']}
    use_gpu = torch.cuda.is_available()
    return image_datasets, dataloaders, use_gpu


def flip_horizontal(img):
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def which_view(name):
    if 'satellite' in name:
        return 1
    elif 'street' in name:
        return 2
    elif 'drone' in name:
        return 3
    else:
        print('unknown view')
    return -1


def extract_feature_one(model, data_image, view_index):
    return


def extract_feature(model, dataloaders, ms, opt, view_index=1):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        # print("img=", img)
        # print("type(img)=", type(img))
        # print("type(label)=", type(label))
        # print("label=", label)
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n, 512).zero_().cuda()

        for i in range(2):
            if (i == 1):
                img = flip_horizontal(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                outputs = model(input_img)
                # if opt.views == 2:
                #     if view_index == 1:
                #         outputs, _ = model(input_img, None)
                #     elif view_index == 2:
                #         _, outputs = model(None, input_img)
                # elif opt.views == 3:
                #     if view_index == 1:
                #         outputs, _, _ = model(input_img, None, None)
                #     elif view_index == 2:
                #         _, outputs, _ = model(None, input_img, None)
                #     elif view_index == 3:
                #         _, _, outputs = model(None, None, input_img)
                ff += outputs
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff.data.cpu()), 0)
    return features


def get_labels_paths(img_path):
    camera_id = []
    labels = []
    paths = []
    for path, v in img_path:
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
        paths.append(path)
    return labels, paths


def load_train_model(opt, image_datasets):
    print("-----------Test: Load Collected data Trained model-----------")
    model, _, epoch = load_network(opt.name, opt)
    # print("model=",model)
    model.classifier.classifier = nn.Sequential()
    model = model.eval()
    model = model.cuda()

    # gallery_name = 'gallery_street'
    # query_name = 'query_drone'

    # gallery_name = 'gallery_drone'
    # query_name = 'query_street'

    # gallery_name = 'gallery_satellite'
    # query_name = 'query_drone'

    gallery_name = 'gallery_drone'
    query_name = 'query_satellite'

    # which_gallery = which_view(gallery_name)
    # which_query = which_view(query_name)
    # print('%d -> %d: %s -> %s' % (which_query, which_gallery, query_name, gallery_name))

    gallery_path = image_datasets[gallery_name].imgs
    f = open('gallery_name.txt', 'w')
    for p in gallery_path:
        f.write(p[0] + '\n')
    query_path = image_datasets[query_name].imgs
    f = open('query_name.txt', 'w')
    for p in query_path:
        f.write(p[0] + '\n')

    gallery_label, gallery_path = get_labels_paths(gallery_path)
    query_label, query_path = get_labels_paths(query_path)
    # return model, query_name, query_label, query_path, gallery_name, gallery_label, gallery_path, which_query, which_gallery
    return model, query_name, query_label, query_path, gallery_name, gallery_label, gallery_path

def save_matlab(gallery_feature, gallery_label, gallery_path, query_feature, query_label, query_path):
    # Save to Matlab for check
    result_mat = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_path': gallery_path,
              'query_f': query_feature.numpy(), 'query_label': query_label, 'query_path': query_path}
    # scipy.io.savemat('satellite->drone.mat', result_mat)
    # scipy.io.savemat('street->drone.mat', result_mat)
    # scipy.io.savemat('drone->satellite_new.mat', result_mat)
    scipy.io.savemat('new_satellite->drone.mat', result_mat)
    return result_mat

if __name__ == '__main__':
    opt = init_options()
    opt, str_ids, name, test_dir = init_load_train_config(opt)
    ms = choose_gpu(opt, str_ids)
    image_datasets, dataloaders, use_gpu = load_data(opt, test_dir)
    start_time = time.time()
    model, query_name, query_label, query_path, gallery_name, gallery_label, gallery_path = load_train_model(
        opt, image_datasets)


    with torch.no_grad():
        query_feature = extract_feature(model, dataloaders[query_name], ms, opt)
        gallery_feature = extract_feature(model, dataloaders[gallery_name], ms, opt)

    end_time = time.time()
    total_time = end_time-start_time
    print('Test complete in {:.0f}m {:.0f}s'.format(
        total_time // 60, total_time % 60))

    result_mat = save_matlab(gallery_feature, gallery_label, gallery_path, query_feature, query_label, query_path)
    print(opt.name)
    result = './model/%s/result.txt' % opt.name
    os.system('python evaluate_gpu_1_ctx.py | tee -a %s' % result)
