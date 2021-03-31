import argparse
import time

import scipy.io
import torch
import numpy as np
import os
from torchvision import transforms
import torchvision
from PIL import Image
from torchvision import datasets
import matplotlib
import matplotlib.pyplot as plt
from model import simple_CNN, simple_2CNN, simple_3CNN, view_net
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn

'''
    input: one street_view and ten drone_view from different label
    output: a figure contains one street_view and ten drone_view picture, 
        define the true label of drone_view from street_view color red and others define color black 
    1、对于输入的街拍图片，获取对应的label；对于输入的无人机图片，获取对应的label；
        根据将全部图片输出到一张图片上，然后标注label与街拍图片label相同的为红色。
    2、对于输入的街拍图片，获取对应的512维度特征。对于10张无人机图片，获取对应的特征。
        对特征进行计算，按相似度得分排序，将label相同的标注为红色。
'''


def init_option():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--street_imgdir', default='./data/demo/street', type=str, help='street demo image path')
    parser.add_argument('--drone_imgdir', default='./data/demo/drone', type=str, help='drone demo image path')
    parser.add_argument('--out_dir', default='./data/demo_out', type=str, help='demo outputs path')
    opts = parser.parse_args()
    return opts


def load_model(iscuda=True):
    net = simple_CNN(num_classes=21, droprate=0.5, stride=1, pool='avg')
    save_filename = 'net_999.pth'
    save_path = os.path.join('./model/student', 'view', save_filename)
    net.load_state_dict(torch.load(save_path))
    net.eval()
    if iscuda:
        net.cuda()
    else:
        net.cpu()
    return net

def load_model_teacher():
    net = view_net(class_num=21, droprate=0.5, stride=1, pool='avg')
    save_filename = 'net_050.pth'
    save_path = os.path.join('./model/teacher', 'view', save_filename)
    net.load_state_dict(torch.load(save_path))
    net.eval()
    # net.cuda()
    net.cpu()
    return net


def pre_label(model, input_img_path, iscuda=True):
    data_transforms = transforms.Compose([
        transforms.Resize((384, 384), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_img = data_transforms(Image.open(input_img_path).convert('RGB'))
    # input_img_Tensor = data_transforms(input_img)
    if iscuda:
        input_img_Tensor = Variable(torch.unsqueeze(input_img.cuda(), dim=0).float(), requires_grad=False)
    else:
        input_img_Tensor = Variable(torch.unsqueeze(input_img.cpu(), dim=0).float(), requires_grad=False)
    # print(input_img_Tensor.shape)
    outputs_Tensor = model(input_img_Tensor)
    predict_label = torch.argmax(outputs_Tensor, dim=1).cpu().numpy()
    # print(predict_label)
    return predict_label


def load_demo_imgdir(is_demo1=True):
    if is_demo1:
        street_img_path = './data/demo/street/02_street_52.jpg'
        drone_img_v1_path = './data/demo/drone/02_drone_154.jpg'
        drone_img_v2_path = './data/demo/drone/03_drone_110.jpg'
        drone_img_v3_path = './data/demo/drone/07_drone_139.jpg'
        drone_img_v4_path = './data/demo/drone/07_drone_140.jpg'
        drone_img_v5_path = './data/demo/drone/09_drone_13.jpg'
        drone_img_v6_path = './data/demo/drone/09_drone_116.jpg'
        drone_img_v7_path = './data/demo/drone/14_drone_75.jpg'
        drone_img_v8_path = './data/demo/drone/16_drone_17.jpg'
        drone_img_v9_path = './data/demo/drone/17_drone_86.jpg'
        drone_img_v10_path = './data/demo/drone/20_drone_58.jpg'
    else:
        street_img_path = './data/demo_v1/street/image_937.jpg'
        drone_img_v1_path = './data/demo_v1/drone/02_drone_30.jpg'
        drone_img_v2_path = './data/demo_v1/drone/02_drone_31.jpg'
        drone_img_v3_path = './data/demo_v1/drone/03_drone_152.jpg'
        drone_img_v4_path = './data/demo_v1/drone/07_drone_32.jpg'
        drone_img_v5_path = './data/demo_v1/drone/09_2_drone_2.jpg'
        drone_img_v6_path = './data/demo_v1/drone/09_drone_9.jpg'
        drone_img_v7_path = './data/demo_v1/drone/14_drone_19.jpg'
        drone_img_v8_path = './data/demo_v1/drone/16_drone_64.jpg'
        drone_img_v9_path = './data/demo_v1/drone/17_drone_94.jpg'
        drone_img_v10_path = './data/demo_v1/drone/20_drone_13.jpg'

    drone_img_path_list = [drone_img_v1_path, drone_img_v2_path, drone_img_v3_path, drone_img_v4_path,
                           drone_img_v5_path, drone_img_v6_path, drone_img_v7_path, drone_img_v8_path,
                           drone_img_v9_path, drone_img_v10_path]
    return street_img_path, drone_img_path_list


def predict_label(model, street_img_path, drone_img_path_list, iscuda=True):
    street_label = pre_label(model=model, input_img_path=street_img_path, iscuda=iscuda)[0]
    drone_label_list = [pre_label(model=model, input_img_path=drone_img_path_list[i], iscuda=iscuda)[0] for i in range(0, 10)]
    return street_label, drone_label_list


def image_show(image_path, title=None):
    image = plt.imread(image_path)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.2)


def visualize_result_graph(street_img_path, street_label, drone_img_path_list, drone_label_list, save_name):
    try:
        fig = plt.figure(figsize=(16, 4))
        ax = plt.subplot(1, 11, 1)
        ax.axis('off')
        image_show(street_img_path, 'street')
        for i in range(0, 10):
            ax = plt.subplot(1, 11, i + 2)
            ax.axis('off')
            image_path = drone_img_path_list[i]
            label = drone_label_list[i]
            image_show(image_path=image_path)
            if label == street_label:
                ax.set_title('True', color='green')
            else:
                ax.set_title('False', color='red')
    except RuntimeError:
        print('Runtime Error')
    fig.savefig(save_name, format='jpg', bbox_inches='tight')


def load_demo_imgdir_v2(is_demo1=True):
    if is_demo1:
        street_img_path = './data/demo/street/02_street_52.jpg'
        drone_img_v1_path = './data/demo/drone_v2/02_drone_4.jpg'
        drone_img_v2_path = './data/demo/drone_v2/02_drone_17.jpg'
        drone_img_v3_path = './data/demo/drone_v2/02_drone_56.jpg'
        drone_img_v4_path = './data/demo/drone_v2/02_drone_59.jpg'
        drone_img_v5_path = './data/demo/drone_v2/02_drone_79.jpg'
        drone_img_v6_path = './data/demo/drone_v2/02_drone_87.jpg'
        drone_img_v7_path = './data/demo/drone_v2/02_drone_105.jpg'
        drone_img_v8_path = './data/demo/drone_v2/02_drone_127.jpg'
        drone_img_v9_path = './data/demo/drone_v2/02_drone_150.jpg'
        drone_img_v10_path = './data/demo/drone_v2/02_drone_154.jpg'
    else:
        street_img_path = './data/demo_v1/street/image_937.jpg'
        drone_img_v1_path = './data/demo_v1/drone_v2/07_drone_3.jpg'
        drone_img_v2_path = './data/demo_v1/drone_v2/07_drone_10.jpg'
        drone_img_v3_path = './data/demo_v1/drone_v2/07_drone_11.jpg'
        drone_img_v4_path = './data/demo_v1/drone_v2/07_drone_23.jpg'
        drone_img_v5_path = './data/demo_v1/drone_v2/07_drone_53.jpg'
        drone_img_v6_path = './data/demo_v1/drone_v2/07_drone_54.jpg'
        drone_img_v7_path = './data/demo_v1/drone_v2/07_drone_55.jpg'
        drone_img_v8_path = './data/demo_v1/drone_v2/07_drone_58.jpg'
        drone_img_v9_path = './data/demo_v1/drone_v2/07_drone_95.jpg'
        drone_img_v10_path = './data/demo_v1/drone_v2/07_drone_123.jpg'

    drone_img_path_list = [drone_img_v1_path, drone_img_v2_path, drone_img_v3_path, drone_img_v4_path,
                           drone_img_v5_path, drone_img_v6_path, drone_img_v7_path, drone_img_v8_path,
                           drone_img_v9_path, drone_img_v10_path]
    return street_img_path, drone_img_path_list


def pre_feature(model, input_img_path, iscuda=True):
    data_transforms = transforms.Compose([
        transforms.Resize((384, 384), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_img = data_transforms(Image.open(input_img_path).convert('RGB'))
    # input_img_Tensor = data_transforms(input_img)
    if iscuda:
        ff = torch.FloatTensor(1, 512).zero_().cuda()
    else:
        ff = torch.FloatTensor(1, 512).zero_().cpu()
    for i in range(0, 2):
        # if (i == 1):
        #     input_img = flip_horizontal(input_img)
        if iscuda:
            input_img_Tensor = Variable(torch.unsqueeze(input_img.cuda(), dim=0).float(), requires_grad=False)
        else:
            input_img_Tensor = Variable(torch.unsqueeze(input_img.cpu(), dim=0).float(), requires_grad=False)
        outputs = model(input_img_Tensor)
        ff += outputs
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))
    features = ff.data.cpu()
    return features


'''
    # input_img_Tensor = Variable(torch.unsqueeze(input_img.cuda(), dim=0).float(), requires_grad=False)
    # print(input_img_Tensor.shape)
    # predict_feature = model(input_img_Tensor)

    # predict_feature_norm = torch.norm(predict_feature, p=2, dim=1, keepdim=True)
    # predict_feature_v1 = predict_feature.div(predict_feature_norm.expand_as(predict_feature))
    # predict_feature_v1 = predict_feature_v1.data.cpu()

    # predict_feature = torch.argmax(outputs_Tensor, dim=1).cpu().numpy()
    # print(predict_feature.shape)
    # print(predict_feature_v1.shape)
    # print(predict_feature - predict_feature_v1 < 1e-6)

    # predict_feature = predict_feature.data.cpu()
'''


def cal_score(street_feature, drone_feature_list_v2):
    street_feature = street_feature.view(-1, 1)
    total_score = np.zeros(shape=[1, 10])
    for i in range(0, 10):
        score = torch.mm(drone_feature_list_v2[i], street_feature)
        score = score.squeeze(1).cpu()
        # print(type(score))
        score = score.numpy()
        # print(score)
        # print(score.shape)
        total_score[0, i] = score
    # print(total_score.shape)
    # print(type(np.argmax(total_score)))
    # return np.argmax(total_score)

    # print(np.argsort(total_score))
    return np.argsort(-total_score)


'''
    对每个图片计算特征矩阵，将street的特征矩阵和其他每个进行得分，标注特分最高的是第几个drone_view
'''


def compare_feature(model, street_img_path_v2, drone_img_path_list_v2, iscuda=True):
    street_feature = pre_feature(model, street_img_path_v2, iscuda=iscuda)
    drone_feature_list_v2 = [pre_feature(model, input_img_path=drone_img_path_list_v2[i], iscuda=iscuda) for i in range(0, 10)]
    # print(street_feature.shape)
    # for i in range(0, 10):
    #     print(drone_feature_list_v2[i].shape)

    # maxscore_num = cal_score(street_feature, drone_feature_list_v2)
    # return maxscore_num

    score_list = cal_score(street_feature, drone_feature_list_v2)
    return score_list


def visualize_result_graph_v2(street_img_path_v2, drone_img_path_list_v2, maxscore_num, save_name):
    try:
        fig = plt.figure(figsize=(16, 4))
        ax = plt.subplot(1, 11, 1)
        ax.axis('off')
        image_show(street_img_path_v2, 'street')
        for i in range(0, 10):
            ax = plt.subplot(1, 11, i + 2)
            ax.axis('off')
            image_path = drone_img_path_list_v2[i]
            image_show(image_path=image_path)
            if i == maxscore_num:
                ax.set_title('True', color='green')
            else:
                ax.set_title('False', color='red')
    except RuntimeError:
        print('Runtime Error')
    fig.savefig(save_name, format='jpg', bbox_inches='tight')
    # fig.savefig(save_name)


def visualize_result_graph_v3(street_img_path_v2, drone_img_path_list_v2, score_list, save_name):
    try:
        fig = plt.figure(figsize=(16, 4))
        ax = plt.subplot(1, 11, 1)
        ax.axis('off')
        image_show(street_img_path_v2, 'street')
        for i in range(0, 10):
            ax = plt.subplot(1, 11, i + 2)
            ax.axis('off')
            # print(score_list)
            image_path = drone_img_path_list_v2[score_list[0][i]]
            image_show(image_path=image_path)
            if i == 0:
                ax.set_title('Location', color='green')
            else:
                ax.set_title('Others', color='red')
    except RuntimeError:
        print('Runtime Error')
    fig.savefig(save_name, format='jpg', bbox_inches='tight')


if __name__ == '__main__':
    iscuda = True
    is_demo1 = False
    # model = load_model_teacher()


    model = load_model(iscuda=iscuda)

    # # TODO: demo_v1: street_view -> different label drone_view
    street_img_path, drone_img_path_list = load_demo_imgdir(is_demo1=is_demo1)
    start_time = time.time()
    street_label, drone_label_list = predict_label(model,street_img_path, drone_img_path_list, iscuda=iscuda)
    end_time = time.time()
    print(end_time-start_time)

    # visualize_result_graph(street_img_path, street_label, drone_img_path_list, drone_label_list,
    #                        save_name='demo_v9.jpg')

    # model.classifier.classifier = nn.Sequential()
    # TODO: demo_v2: street_viw -> different drone_view from true label building
    # street_img_path_v2, drone_img_path_list_v2 = load_demo_imgdir_v2(is_demo1=is_demo1)
    # start_time = time.time()
    # score_list = compare_feature(model, street_img_path_v2, drone_img_path_list_v2, iscuda=iscuda)
    # end_time = time.time()
    # print(end_time-start_time)
    # visualize_result_graph_v2(street_img_path_v2, drone_img_path_list_v2, maxscore_num, save_name='demo_v4.jpg')
    # visualize_result_graph_v3(street_img_path_v2, drone_img_path_list_v2, score_list, save_name='demo_v10.jpg')
