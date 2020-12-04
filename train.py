from __future__ import print_function, division

from model import view_net
from random_erasing import RandomErasing
from autoaugment import ImageNetPolicy, CIFAR10Policy

import shutil
import time
import utils
import argparse
import torch
import torchvision
import matplotlib
import PIL
import copy
import os
import yaml

import matplotlib.pyplot as plt

matplotlib.use('agg')
# print(torch.__version__)
torch_version = torch.__version__

TRAIN_FILE_NAME = 'drone'

try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError:
    print("Warning: 没有apex包")


def init_options():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--name', default='view', type=str, help='output model name')
    parser.add_argument('--pool', default='avg', type=str, help='pool avg')
    parser.add_argument('--data_dir', default='./data/train', type=str, help='training dir path')
    parser.add_argument('--train_all', action='store_true', help='use all training data')
    parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
    parser.add_argument('--batchsize', default=4, type=int, help='batchsize')
    parser.add_argument('--stride', default=2, type=int, help='stride')
    parser.add_argument('--pad', default=10, type=int, help='padding')
    parser.add_argument('--h', default=384, type=int, help='height')
    parser.add_argument('--w', default=384, type=int, help='width')
    parser.add_argument('--views', default=2, type=int, help='the number of views')
    parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
    parser.add_argument('--use_dense', action='store_true', help='use densenet')
    parser.add_argument('--use_NAS', action='store_true', help='use NAS')
    parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--moving_avg', default=1.0, type=float, help='moving average')
    parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
    parser.add_argument('--DA', action='store_true', help='use Color Data Augmentation')
    parser.add_argument('--resume', action='store_true', help='use resume training')
    parser.add_argument('--share', action='store_true', help='share weight between different view')
    parser.add_argument('--extra_Google', default="true", action='store_true', help='using extra noise Google')
    parser.add_argument('--fp16', action='store_true',
                        help='use float16 instead of float32, which will save about 50% memory')
    opt = parser.parse_args()
    return opt


def opt_resume(opt):
    model, opt, start_epoch = utils.load_network(opt.name, opt)
    return model, opt, start_epoch


def opt_not_resume():
    start_epoch = 0
    return start_epoch


def init_parameter(opt):
    fp16 = opt.fp16
    data_dir = opt.data_dir
    name = opt.name
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)

    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        torch.backends.cudnn.benchmark = True

    return fp16, data_dir, name


def init_transform_train_list():
    '''
        裁剪（Crop）——
            中心裁剪：transforms.CenterCrop
            随机裁剪：transforms.RandomCrop
            随机长宽比裁剪：transforms.RandomResizedCrop
            上下左右中心裁剪：transforms.FiveCrop
            上下左右中心裁剪后翻转，transforms.TenCrop
        翻转和旋转（Flip and Rotation） ——
            依概率p水平翻转：transforms.RandomHorizontalFlip(p=0.5)
            依概率p垂直翻转：transforms.RandomVerticalFlip(p=0.5)
            随机旋转：transforms.RandomRotation
        图像变换（resize） ——transforms.Resize
        标准化：transforms.Normalize
        转为tensor，并归一化至[0-1]：transforms.ToTensor
        填充：transforms.Pad
        修改亮度、对比度和饱和度：transforms.ColorJitter
        转灰度图：transforms.Grayscale
        线性变换：transforms.LinearTransformation()
        仿射变换：transforms.RandomAffine
        依概率p转为灰度图：transforms.RandomGrayscale
        将数据转换为PILImage：transforms.ToPILImage
        transforms.Lambda：Apply a user-defined lambda as a transform.
        对transforms操作，使数据增强更灵活
            transforms.RandomChoice(transforms)， 从给定的一系列transforms中选一个进行操作
            transforms.RandomApply(transforms, p=0.5)，给一个transform加上概率，依概率进行操作
            transforms.RandomOrder，将transforms中的操作随机打乱
    '''
    transform_train_list = [
        # torchvision.transforms.RandomResizedCrop(size=(opt.h, opt.w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        torchvision.transforms.Resize((opt.h, opt.w), interpolation=3),
        torchvision.transforms.Pad(opt.pad, padding_mode='edge'),
        torchvision.transforms.RandomCrop((opt.h, opt.w)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    return transform_train_list


def init_transform_val_list():
    transform_val_list = [
        torchvision.transforms.Resize(size=(opt.h, opt.w), interpolation=3),  # Image.BICUBIC
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    return transform_val_list


def init_data_transforms(transform_train_list, transform_val_list):
    data_transforms = {
        'train': torchvision.transforms.Compose(transform_train_list),
        'val': torchvision.transforms.Compose(transform_val_list)
    }
    return data_transforms


def load_data(opt):
    transform_train_list = init_transform_train_list()
    transform_val_list = init_transform_val_list()
    if opt.erasing_p > 0:
        transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]
    if opt.color_jitter:
        transform_train_list = [torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                                   hue=0)] + transform_train_list
    if opt.DA:
        transform_train_list = [ImageNetPolicy()] + transform_train_list
    # print(transform_train_list)

    data_transforms = init_data_transforms(transform_train_list, transform_val_list)
    train_all = ''
    if opt.train_all:
        train_all = '_all'

    image_datasets = {}
    image_datasets[TRAIN_FILE_NAME] = torchvision.datasets.ImageFolder(os.path.join(data_dir, TRAIN_FILE_NAME), data_transforms['train'])
    # 8 workers may work faster
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize, shuffle=True, num_workers=2,
                                       pin_memory=True)
        for x in [TRAIN_FILE_NAME]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN_FILE_NAME]}
    class_names = image_datasets[TRAIN_FILE_NAME].classes
    # print(class_names)
    print('dataset_sizes=', dataset_sizes)
    use_gpu = torch.cuda.is_available()
    return transform_train_list, data_transforms, image_datasets, dataloaders, dataset_sizes, class_names, use_gpu


def init_loss_err():
    y_loss = {}
    y_loss['train'] = []
    y_loss['val'] = []
    y_err = {}
    y_err['train'] = []
    y_err['val'] = []
    return y_loss, y_err


def init_running():
    running_loss = 0.0
    running_corrects = 0.0
    return running_loss, running_corrects


def train_model(model, model_test, criterion_lr, optimizer_view, exp_lr_scheduler, dataset_sizes, start_epoch, opt,
                num_epochs=25):
    start_time = time.time()
    start_warm_lr_up = 0.1
    start_warm_iteration = round(dataset_sizes[TRAIN_FILE_NAME] / opt.batchsize) * opt.warm_epoch

    for epoch in range(num_epochs - start_epoch):
        epoch = epoch + start_epoch
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)
            running_loss, running_corrects = init_running()
            # for data, data2 in zip(dataloaders['view'], dataloaders['google']):
            for data in dataloaders[TRAIN_FILE_NAME]:
                inputs, labels = data
                # inputs2, labels2 = data2
                now_batch_size, c, h, w = inputs.shape
                if now_batch_size < opt.batchsize:
                    continue
                if use_gpu:
                    inputs = torch.autograd.Variable(inputs.cuda().detach())
                    labels = torch.autograd.Variable(labels.cuda().detach())
                    # inputs = inputs.cuda()
                    # if opt.extra_Google:
                    #     inputs2 = torch.autograd.Variable(inputs2.cuda().detach())
                    #     labels2 = torch.autograd.Variable(labels2.cuda().detach())
                else:
                    inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
                optimizer_view.zero_grad()  # zero the parameter gradients

                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                    # if opt.extra_Google:
                    #     outputs, outputs2 = model(inputs, inputs2)
                    # else:
                    #     outputs = model(inputs)
                _, predicts = torch.max(outputs.data, 1)
                loss = criterion_lr(outputs, labels)
                # if opt.extra_Google:
                #     loss += criterion_lr(outputs2, labels2)
                # backward+optimize only if in training phase
                if epoch < opt.warm_epoch and phase == 'train':
                    start_warm_lr_up = min(1.0, start_warm_lr_up + 0.9 / start_warm_iteration)
                    loss *= start_warm_lr_up

                if phase == 'train':
                    if fp16:
                        with amp.scale_loss(loss, optimizer_view) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    optimizer_view.step()

                    if opt.moving_avg < 1.0:
                        utils.update_average(model_test, model, opt.moving_avg)

                running_loss += loss.item() * now_batch_size
                running_corrects += float(torch.sum(predicts == labels.data))

            epoch_loss = running_loss / dataset_sizes[TRAIN_FILE_NAME]
            epoch_acc = running_corrects / dataset_sizes[TRAIN_FILE_NAME]
            # epoch_loss = running_loss / len(os.listdir(opt.data_dir))
            # epoch_acc = running_corrects / len(os.listdir(opt.data_dir))
            print('{} Loss: {:.4f} View_Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)
            # deep copy the model
            if phase == 'train':
                exp_lr_scheduler.step()
            last_model_weights = model.state_dict()
            if epoch % 5 == 4:
                utils.save_network(model, opt.name, epoch)

        time_elapsed = time.time() - start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    utils.save_network(model, opt.name, num_epochs)

    return model


def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join('./model', name, 'train.jpg'))


if __name__ == '__main__':
    opt = init_options()
    fp16, data_dir, name = init_parameter(opt)
    transform_train_list, data_transforms, image_datasets, dataloaders, dataset_sizes, class_names, use_gpu = load_data(
        opt)
    y_loss, y_err = init_loss_err()

    if opt.resume:
        model, opt, start_epoch = opt_resume(opt)
    else:
        start_epoch = opt_not_resume()
        model = view_net(len(class_names), droprate=opt.droprate, stride=opt.stride, pool=opt.pool, share_weight=opt.share, VGG19=False, RESNET152=True)


    x_epoch = []
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="top1err")
    opt.nclasses = len(class_names)
    print(model)

    if start_epoch >= 40:
        opt.lr = opt.lr * 0.1
    ignored_params = list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    optimizer_view = torch.optim.SGD([
        {'params': base_params, 'lr': 0.1 * opt.lr},
        {'params': model.classifier.parameters(), 'lr': opt.lr}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_view, step_size=80, gamma=0.1)

    # dir_name = os.path.join('./model', name)
    dir_name = os.path.join('./model', name)
    if not opt.resume:
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        # record every run
        shutil.copyfile('./train.py', dir_name + '/train.py')
        shutil.copyfile('./model.py', dir_name + '/model.py')
        # save opts
        with open('%s/opts.yaml' % dir_name, 'w') as fp:
            yaml.dump(vars(opt), fp, default_flow_style=False)
    model = model.cuda()

    if fp16:
        model, optimizer_view = amp.initialize(model, optimizer_view, opt_level="O1")

    criterion_lr = torch.nn.CrossEntropyLoss()
    if opt.moving_avg < 1.0:
        model_test = copy.deepcopy(model)
        num_epochs = 140
    else:
        model_test = None
        num_epochs = 120

    train_model(model, model_test, criterion_lr, optimizer_view, exp_lr_scheduler, dataset_sizes, start_epoch, opt,
                num_epochs=20)

# python train.py --name drone --droprate 0.75 --batchsize 8 --stride 1 --h 384  --w 384 --fp16;
