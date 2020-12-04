# -*- coding: utf-8 -*-

from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
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
from utils import load_network
from apex.fp16_utils import *


def init_options():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
    parser.add_argument('--test_dir', default='./data/test', type=str, help='./test_data')
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

def flip_horizontal(img):
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


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
    return model


def save_matlab(gallery_feature, gallery_label, gallery_path, query_feature, query_label, query_path):
    # Save to Matlab for check
    result_mat = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_path': gallery_path,
                  'query_f': query_feature.numpy(), 'query_label': query_label, 'query_path': query_path}
    scipy.io.savemat('drone->satellite.mat', result_mat)
    return result_mat


def load_image(model, img_path):
    print(img_path)
    image = Image.open(img_path).convert('RGB')
    # image.show()
    data_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_Tensor = data_transforms(image)
    print(type(image_Tensor))
    feature = torch.FloatTensor()
    ff = torch.FloatTensor(1, 512).zero_().cuda()
    # input_img = Variable(torch.unsqueeze(image_Tensor.cuda(), dim=0).float(), requires_grad=False)
    # image_Tensor = flip_horizontal(image_Tensor.cuda())
    # input_img = Variable(image_Tensor.cuda())
    for i in range(2):
        # if (i == 1):
        #     input_img = Variable(torch.unsqueeze(image_Tensor.cuda(), dim=0).float(), requires_grad=False)
        input_img = Variable(torch.unsqueeze(image_Tensor.cuda(), dim=0).float(), requires_grad=False)
        for scale in ms:
            outputs = model(input_img)
            ff += outputs
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))
    feature = torch.cat((feature, ff.data.cpu()), 0)
    return feature


def cal_score(query_feature, gallery_feature):
    query_feature = query_feature.view(-1, 1)
    score = torch.mm(gallery_feature, query_feature)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    print("score=", score)
    return score


# def save_to_txt(query_img_path, gallery_img_path, score):
#     file = open('AHU.txt', mode="a+")
#     file.writelines("query_img_path: ")
#     file.writelines(query_img_path)
#     file.writelines("; ")
#     file.writelines("gallery_img_path: ")
#     file.writelines(gallery_img_path)
#     file.writelines("; ")
#     file.writelines("score: ")
#     file.writelines(str(score))
#     file.writelines("; ")
#     file.writelines('\n')


if __name__ == '__main__':
    opt = init_options()
    opt, str_ids, name, test_dir = init_load_train_config(opt)
    ms = choose_gpu(opt, str_ids)
    model, _, epoch = load_network("drone", opt)
    # print("model=",model)
    model.classifier.classifier = nn.Sequential()
    model = model.eval()
    model = model.cuda()

    start_time = time.time()

    query_img_path = './data/new_street/15/image_1.jpg'
    query_feature = load_image(model, query_img_path)
    print("type(query_feature)=", type(query_feature))

    gallery_img_path = './data/new_street/00/image_1.jpg'
    gallery_feature = load_image(model, gallery_img_path)
    print("type(gallery_feature)=", type(gallery_feature))
    score = cal_score(query_feature, gallery_feature)

    end_time = time.time()
    totaltime = end_time - start_time
    print("totaltime=%f" % totaltime)

    # save_to_txt(query_img_path, gallery_img_path, score)

