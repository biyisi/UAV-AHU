from __future__ import print_function, division

from model import view_net, simple_CNN, simple_2CNN, simple_3CNN, view_net_152, simple_resnet_18, simple_10CNN, simple_resnet_50
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

# 指定训练的文件名
TRAIN_FILE_NAME = 'view'

try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError:
    print("Warning: 没有apex包")


def init_options():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--out_model_name', default='view', type=str, help='output model name')
    parser.add_argument('--pool', default='avg', type=str, help='pool avg')
    parser.add_argument('--data_dir', default='./data/train', type=str, help='training dir path')
    parser.add_argument('--train_all', action='store_true', help='use all training data')
    parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
    parser.add_argument('--batchsize', default=4, type=int, help='batchsize')
    parser.add_argument('--stride', default=1, type=int, help='stride')
    parser.add_argument('--pad', default=10, type=int, help='padding')
    parser.add_argument('--h', default=384, type=int, help='height')
    parser.add_argument('--w', default=384, type=int, help='width')
    parser.add_argument('--views', default=2, type=int, help='the number of views')
    parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
    parser.add_argument('--use_dense', action='store_true', help='use densenet')
    parser.add_argument('--use_NAS', action='store_true', help='use NAS')
    parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--moving_avg', default=1.0, type=float, help='moving average')
    parser.add_argument('--droprate', default=0.75, type=float, help='drop rate')
    parser.add_argument('--DA', action='store_true', help='use Color Data Augmentation')
    parser.add_argument('--resume', action='store_true', help='use resume training')
    parser.add_argument('--share', action='store_true', help='share weight between different view')
    parser.add_argument('--extra_Google', default="true", action='store_true', help='using extra noise Google')
    parser.add_argument('--fp16', default="true", action='store_true',
                        help='use float16 instead of float32, which will save about 50% memory')
    parser.add_argument('--net_type', default='student', type=str, help='choose train teacher_net or student_net')
    opt = parser.parse_args()
    return opt


def opt_resume(opt):
    model, opt, start_epoch = utils.load_network_teacher(opt.out_model_name, opt, RESNET152=True, RESNET18=False,
                                                         VGG19=False)
    return model, opt, start_epoch


def opt_resume_student(opt):
    model, opt, start_epoch = utils.load_network_student(opt.out_model_name, opt)
    return model, opt, start_epoch


def init_parameter(opt):
    fp16 = opt.fp16
    data_dir = opt.data_dir
    out_model_name = opt.out_model_name
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

    return fp16, data_dir, out_model_name


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


def init_transform_test_list():
    transform_test_list = [
        torchvision.transforms.Resize(size=(opt.h, opt.w), interpolation=3),  # Image.BICUBIC
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    return transform_test_list


def init_data_transforms(transform_train_list, transform_test_list):
    data_transforms = {
        'train': torchvision.transforms.Compose(transform_train_list),
        'test': torchvision.transforms.Compose(transform_test_list)
    }
    return data_transforms


def load_data(opt):
    transform_train_list = init_transform_train_list()
    transform_test_list = init_transform_test_list()
    if opt.erasing_p > 0:
        transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]
    if opt.color_jitter:
        transform_train_list = [torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                                   hue=0)] + transform_train_list
    if opt.DA:
        transform_train_list = [ImageNetPolicy()] + transform_train_list
    # print(transform_train_list)

    data_transforms = init_data_transforms(transform_train_list, transform_test_list)
    train_all = ''
    if opt.train_all:
        train_all = '_all'

    image_datasets = {}
    image_datasets[TRAIN_FILE_NAME] = torchvision.datasets.ImageFolder(os.path.join(data_dir, TRAIN_FILE_NAME),
                                                                       data_transforms['train'])
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
    y_loss['test'] = []
    y_err = {}
    y_err['train'] = []
    y_err['test'] = []
    return y_loss, y_err


'''指定以知识蒸馏的方式训练网络'''


def train_model_kd(teacher_model, student_model, criterion_lr, optimizer_view, exp_lr_scheduler, dataset_sizes,
                   start_epoch, opt,
                   num_epochs=25):
    start_time = time.time()
    # criterionKD = utils.Logits()
    criterionKD = utils.SoftTarget(4.0)

    for epoch in range(num_epochs - start_epoch):
        epoch = epoch + start_epoch
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        cnt = 0

        for phase in ['train']:
            teacher_model.cuda()
            teacher_model.eval()
            student_model.cuda()
            student_model.train()

            running_loss = 0.0
            running_corrects = 0.0

            for data in dataloaders[TRAIN_FILE_NAME]:
                inputs, labels = data
                now_batch_size, c, h, w = inputs.shape

                cnt += now_batch_size
                if cnt % 60 == 0:
                    print("cnt =", cnt)
                # print("this.cnt =", cnt)
                # print("now_batch_size =", now_batch_size)
                # print("opt.batchsize =", opt.batchsize)

                if now_batch_size < opt.batchsize:
                    continue
                # TODO: this
                inputs = torch.autograd.Variable(inputs.cuda().detach())
                labels = torch.autograd.Variable(labels.cuda().detach())

                # TODO: this
                # 学生网络训练结果
                outputs_student = student_model(inputs)
                _, predicts = torch.max(outputs_student.data, 1)
                loss_student = criterion_lr(outputs_student, labels)
                # 教师网络推理结果
                with torch.no_grad():
                    outputs_teacher = teacher_model(inputs)
                # outputs_teacher1 = teacher_model(inputs)

                loss_kd = criterionKD(outputs_student, outputs_teacher.detach()) * 1.0
                # loss_kd = criterionKD(outputs_student, outputs_teacher.detach()) * 0.5
                # loss_kd = criterion_KD()
                loss = 0.15 * loss_student + 0.85 * loss_kd
                # loss = 0.08*loss_student + 0.92*loss_kd

                # print(loss_student) #
                # print(loss_kd) #
                # print(loss) #
                # print(optimizer_view)
                optimizer_view.zero_grad()  # zero the parameter gradients
                # print("this")
                if phase == 'train':
                    # print("this1")
                    # TODO: this
                    if fp16:
                        # print("this2")
                        with amp.scale_loss(loss, optimizer_view) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        # print("this3")
                        loss.backward()

                    optimizer_view.step()

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
            exp_lr_scheduler.step()
            last_model_weights = student_model.state_dict()
            # if epoch % 20 == 19:
            #     utils.save_network_kd(student_model, opt.out_model_name, epoch)
            utils.save_network_kd(student_model, opt.out_model_name, epoch)

        time_elapsed = time.time() - start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    utils.save_network_kd(student_model, opt.out_model_name, num_epochs)

    return student_model


'''指定正常训练一个网络模型'''


def train_model(model, criterion_lr, optimizer_view, exp_lr_scheduler, dataset_sizes, start_epoch, opt,
                num_epochs=25):
    start_time = time.time()
    start_warm_lr_up = 0.1
    start_warm_iteration = round(dataset_sizes[TRAIN_FILE_NAME] / opt.batchsize) * opt.warm_epoch

    for epoch in range(num_epochs - start_epoch):
        cnt = 0

        epoch = epoch + start_epoch
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0.0

            for data in dataloaders[TRAIN_FILE_NAME]:
                inputs, labels = data
                # inputs2, labels2 = data2
                now_batch_size, c, h, w = inputs.shape
                cnt += now_batch_size
                if cnt % 60 == 0:
                    print("cnt =", cnt)

                if now_batch_size < opt.batchsize:
                    continue
                if use_gpu:
                    # TODO: this
                    inputs = torch.autograd.Variable(inputs.cuda().detach())
                    labels = torch.autograd.Variable(labels.cuda().detach())
                    # inputs = inputs.cuda()
                else:
                    inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)

                # forward
                if phase == 'test':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    # TODO: this
                    outputs = model(inputs)

                _, predicts = torch.max(outputs.data, 1)

                loss = criterion_lr(outputs, labels)

                # backward+optimize only if in training phase
                if epoch < opt.warm_epoch and phase == 'train':
                    start_warm_lr_up = min(1.0, start_warm_lr_up + 0.9 / start_warm_iteration)
                    loss *= start_warm_lr_up

                optimizer_view.zero_grad()  # zero the parameter gradients
                if phase == 'train':
                    # TODO: this
                    if fp16:
                        with amp.scale_loss(loss, optimizer_view) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    optimizer_view.step()

                    # if opt.moving_avg < 1.0:
                    #     utils.update_average(model_test, model, opt.moving_avg)

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
            if epoch % 10 == 0:
                if opt.net_type == 'teacher':
                    utils.save_network_teacher(model, opt.out_model_name, epoch)
                else:
                    utils.save_network_student(model, opt.out_model_name, epoch)

        time_elapsed = time.time() - start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    if opt.net_type == 'teacher':
        utils.save_network_teacher(model, opt.out_model_name, num_epochs)
    else:
        utils.save_network_student(model, opt.out_model_name, num_epochs)

    return model


# def draw_curve(current_epoch):
#     x_epoch.append(current_epoch)
#     ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
#     ax0.plot(x_epoch, y_loss['test'], 'ro-', label='test')
#     ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
#     ax1.plot(x_epoch, y_err['test'], 'ro-', label='test')
#     if current_epoch == 0:
#         ax0.legend()
#         ax1.legend()
#     fig.savefig(os.path.join('./model', out_model_name, 'train.jpg'))


if __name__ == '__main__':
    opt = init_options()
    fp16, data_dir, out_model_name = init_parameter(opt)
    transform_train_list, data_transforms, image_datasets, dataloaders, dataset_sizes, class_names, use_gpu = load_data(
        opt)
    y_loss, y_err = init_loss_err()
    opt.nclasses = len(class_names)

    '''如果输入指定网络类型为教师网络'''
    if opt.net_type == 'teacher':
        if opt.resume:
            teacher_model, opt, start_epoch = opt_resume(opt)
        else:
            start_epoch = 0
            teacher_model = view_net(len(class_names), droprate=opt.droprate, stride=opt.stride, pool=opt.pool,
                                     share_weight=opt.share, VGG19=False, RESNET152=True, RESNET101=False, VGG16=False)
            # teacher_model = view_net_152(len(class_names), droprate=opt.droprate)
        dir_name = os.path.join('./model/teacher', out_model_name)
        print(teacher_model)
        teacher_model = teacher_model.cuda()
        if start_epoch >= 40:
            opt.lr = opt.lr * 0.1
        # 有些参数写多了，忽略掉

        ignored_params = list(map(id, teacher_model.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, teacher_model.parameters())
        # optimizer = torch.optim.SGD([
        #     {'params': base_params, 'lr': 0.1 * opt.lr},
        #     {'params': teacher_model.classifier.parameters(), 'lr': opt.lr}
        # ], weight_decay=5e-4, momentum=0.9, nesterov=True)

        optimizer = torch.optim.SGD([
            {'params': base_params, 'lr': 0},
            {'params': teacher_model.classifier.parameters(), 'lr': opt.lr*0.01}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)

        '''
        teacher_model.model.conv1.weight.requires_grad = False
        teacher_model.model.bn1.weight.requires_grad = False
        # teacher_model.model.layer1.weight.requires_grad = False
        # teacher_model.model.layer2.weight.requires_grad = False
        # teacher_model.model.layer3.weight.requires_grad = False
        # teacher_model.model.layer4.weight.requires_grad = False
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, teacher_model.parameters()), lr=0.1)
        # optimizer = torch.optim.SGD(teacher_model.parameters(), lr=opt.lr, weight_decay=5e-4, momentum=0.9,
        #                             nesterov=True)
        '''
        if not opt.resume:
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)
            # record every run
            shutil.copyfile('./train.py', dir_name + '/train.py')
            shutil.copyfile('./model.py', dir_name + '/model.py')
            # save opts
            with open('%s/opts.yaml' % dir_name, 'w') as fp:
                yaml.dump(vars(opt), fp, default_flow_style=False)

        if fp16:
            teacher_model, optimizer = amp.initialize(teacher_model, optimizer, opt_level="O1")
        criterion_lr = torch.nn.CrossEntropyLoss()  # 交叉熵
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)
        train_model(teacher_model, criterion_lr=criterion_lr, optimizer_view=optimizer,
                    exp_lr_scheduler=exp_lr_scheduler, dataset_sizes=dataset_sizes, start_epoch=start_epoch, opt=opt,
                    num_epochs=2)
    elif opt.net_type == 'student':
        '''如果输入指定网络类型为学生网络'''
        if opt.resume:
            student_model, opt, start_epoch = opt_resume_student(opt)
        else:
            start_epoch = 0
            # student_model = simple_10CNN(num_classes=len(class_names), droprate=opt.droprate, stride=opt.stride,
            #                            pool=opt.pool)
            student_model = simple_resnet_50(num_classes=len(class_names), droprate=opt.droprate, stride=opt.stride,
                                        pool=opt.pool)

        # TODO:
        # start_epoch = 356

        dir_name = os.path.join('./model/student', out_model_name)
        if start_epoch >= 40:
            opt.lr = opt.lr * 0.1

        ignored_params = list(map(id, student_model.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, student_model.parameters())
        optimizer = torch.optim.SGD([
            # {'params': base_params, 'lr': 0.1 * opt.lr},
            {'params': base_params, 'lr': 0},
            {'params': student_model.classifier.parameters(), 'lr': opt.lr}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)

        if not opt.resume:
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)
            # record every run
            shutil.copyfile('./train.py', dir_name + '/train.py')
            shutil.copyfile('./model.py', dir_name + '/model.py')
            # save opts
            with open('%s/opts.yaml' % dir_name, 'w') as fp:
                yaml.dump(vars(opt), fp, default_flow_style=False)
        student_model.cuda()
        if fp16:
            student_model, optimizer = amp.initialize(student_model, optimizer, opt_level="O1")
        # criterion_lr = torch.nn.CrossEntropyLoss()  # 交叉熵
        criterion_lr = torch.nn.CrossEntropyLoss().cuda()  # 交叉熵
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)
        train_model(student_model, criterion_lr=criterion_lr, optimizer_view=optimizer,
                    exp_lr_scheduler=exp_lr_scheduler, dataset_sizes=dataset_sizes, start_epoch=start_epoch, opt=opt,
                    num_epochs=50)
    elif opt.net_type == 'kd':
        '''如果输入指定为知识蒸馏，则读取教师模型用作推理，读取学生模型用作训练'''
        teacher_model, _, _ = utils.load_network_teacher(opt.out_model_name, opt, RESNET101=False, RESNET152=True,
                                                         VGG19=False)
        student_model, _, start_epoch = opt_resume_student(opt)
        teacher_model.cuda()
        teacher_model.eval()
        student_model.cuda()
        student_model.train()
        dir_name = os.path.join('./model/kd', out_model_name)
        # if start_epoch >= 40:
        #     opt.lr = opt.lr * 0.1
        # print("------------opt.batchsize =", opt.batchsize)

        # start_epoch = 80

        ignored_params = list(map(id, student_model.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, student_model.parameters())
        optimizer = torch.optim.SGD([
            {'params': base_params, 'lr': 0.1 * opt.lr},
            {'params': student_model.classifier.parameters(), 'lr': opt.lr}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)

        if fp16:
            student_model, optimizer = amp.initialize(student_model, optimizer, opt_level="O1")
        # criterion_lr = torch.nn.CrossEntropyLoss()  # 交叉熵
        criterion_lr = torch.nn.CrossEntropyLoss().cuda()  # 交叉熵
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)
        train_model_kd(teacher_model, student_model, criterion_lr=criterion_lr, optimizer_view=optimizer,
                       exp_lr_scheduler=exp_lr_scheduler,
                       dataset_sizes=dataset_sizes, start_epoch=start_epoch, opt=opt, num_epochs=280)

    # if not opt.resume:
    #     if not os.path.isdir(dir_name):
    #         os.mkdir(dir_name)
    #     # record every run
    #     shutil.copyfile('./train.py', dir_name + '/train.py')
    #     shutil.copyfile('./model.py', dir_name + '/model.py')
    #     # save opts
    #     with open('%s/opts.yaml' % dir_name, 'w') as fp:
    #         yaml.dump(vars(opt), fp, default_flow_style=False)

    # TODO: 画图代码，未执行
    # x_epoch = []
    # fig = plt.figure()
    # ax0 = fig.add_subplot(121, title="loss")
    # ax1 = fig.add_subplot(122, title="top1err")

    # print(model)
    # # 使用GPU
    # model = model.cuda()

    # if start_epoch >= 40:
    #     opt.lr = opt.lr * 0.1
    # # 有些参数写多了，忽略掉
    # ignored_params = list(map(id, model.classifier.parameters()))
    # base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    # view的优化器
    # optimizer_view = torch.optim.SGD([
    #     {'params': base_params, 'lr': 0.1 * opt.lr},
    #     {'params': model.classifier.parameters(), 'lr': opt.lr}
    # ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    # exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_view, step_size=80, gamma=0.1)

    # if fp16:
    #     model, optimizer_view = amp.initialize(model, optimizer_view, opt_level="O1")
    # 交叉熵
    # criterion_lr = torch.nn.CrossEntropyLoss()

    # if opt.moving_avg < 1.0:
    #     model_test = copy.deepcopy(model)
    #     num_epochs = 140
    # else:
    #     model_test = None
    #     num_epochs = 120

    # if opt.net_type == 'teacher':
    #     train_model(model, criterion_lr, optimizer_view, exp_lr_scheduler, dataset_sizes, start_epoch, opt,
    #             num_epochs=50)
    # if opt.net_type == 'student':
    #     train_model_kd(teacher_model, student_model, criterion_lr, optimizer_view, exp_lr_scheduler, dataset_sizes, start_epoch, opt,
    #                 num_epochs=50)

# python train.py --out_model_name view --droprate 0.75 --batchsize 4 --stride 1 --h 384  --w 384 --fp16 --net_type teacher;
# python train.py --out_model_name view --droprate 0.75 --batchsize 4 --stride 1 --h 384  --w 384 --fp16 --net_type teacher --resume;

# python train.py --out_model_name view --droprate 0.5 --batchsize 8 --lr 0.1 --stride 1 --h 384  --w 384 --fp16 --net_type student;
# python train.py --out_model_name view --droprate 0.5 --batchsize 8 --lr 0.1 --stride 1 --h 384  --w 384 --fp16 --net_type student --resume;

# python train.py --out_model_name view --droprate 0.5 --batchsize 8 --lr 0.1 --stride 1 --h 384  --w 384 --fp16 --net_type kd --resume;

# python train.py --out_model_name view --droprate 0.5 --batchsize 8 --lr 0.1 --stride 2 --h 384  --w 384 --fp16 --net_type student;
# python train.py --out_model_name view --droprate 0.5 --batchsize 8 --lr 0.1 --stride 2 --h 384  --w 384 --fp16 --net_type kd --resume;