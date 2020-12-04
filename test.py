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

from apex.fp16_utils import *
from model import view_net
from utils import load_network


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
                      ['drone']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=False, num_workers=16) for x in
                   ['drone']}
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
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
        paths.append(path)
    return labels, paths


def load_train_model(opt):
    model, _, epoch = load_network(opt.name, opt)
    model.classifier.classifier = torch.nn.Sequential()
    model = model.eval()
    model = model.cuda()
    return model


def save_matlab(query_feature, query_label, query_path):
    result_mat = {'query_feature': query_feature.numpy(), 'query_label': query_label, 'query_path': query_path}
    scipy.io.savemat('query.mat', result_mat)
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
                img = flip_horizontal(img)
            input_img = torch.autograd.Variable(img.cuda())
            for scale in ms:
                outputs = model(input_img)
                ff += outputs
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff.data.cpu()), 0)
    return features

if __name__ == '__main__':
    opt = init_options()
    opt, str_ids, name, test_dir = init_load_train_config(opt)
    ms = choose_gpu(opt, str_ids)
    image_datasets, dataloaders, use_gpu = load_data(opt, test_dir)
    start_time = time.time()
    model = load_train_model(opt)
    query_name = 'drone'
    query_label, query_path = get_labels_paths(image_datasets[query_name].imgs)
    with torch.no_grad():
        query_feature = extract_feature(model, dataloaders[query_name], ms, opt)
    end_time = time.time()
    total_time = end_time - start_time
    print('Test complete in {:.0f}m {:.0f}s'.format(
        total_time // 60, total_time % 60))
    result_mat = save_matlab(query_feature, query_label, query_path)
    print(opt.name)
    result = './model/%s/result.txt' % opt.name