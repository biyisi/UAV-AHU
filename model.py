import pretrainedmodels
import torchvision
import torch
import torch.nn as nn
import os
import torch.nn.functional as F

import numpy as np


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class simple_ResNet9(nn.Module):
    def __init__(self, num_classes, out_channels_simple=256, droprate=0.5, stride=2, pool='avg'):
        super(simple_ResNet9, self).__init__()
        self.model_ft = ResNet(block=BasicBlock, num_blocks=[1, 1, 1, 1], num_classes=256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = ClassBlock(256, num_classes, droprate=droprate)

    def forward(self, x):
        x1 = self.model_ft(x)
        x = self.avgpool(x1)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return x1, x, y


class resblock(nn.Module):
    def __init__(self, in_channels, out_channels, return_before_act):
        super(resblock, self).__init__()
        self.return_before_act = return_before_act
        self.downsample = (in_channels != out_channels)
        if self.downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            self.ds = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels)
            ])
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.ds = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        pout = self.conv1(x)  # pout: pre out before activation
        pout = self.bn1(pout)
        pout = self.relu(pout)
        pout = self.conv2(pout)
        pout = self.bn2(pout)

        if self.downsample:
            residual = self.ds(x)

        pout += residual
        out = self.relu(pout)

        if not self.return_before_act:
            return out
        else:
            return pout, out


class ft_net_resnet18(torch.nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net_resnet18, self).__init__()
        model_ft = torchvision.models.resnet18(pretrained=True)
        # if stride == 1:
        #     model_ft.layer4[0].downsample[0].stride = (1, 1)
        #     model_ft.layer4[0].conv2.stride = (1, 1)

        self.pool = pool
        if pool == 'avg+max':
            model_ft.avgpool2 = torch.nn.AdaptiveAvgPool2d((1, 1))
            model_ft.maxpool2 = torch.nn.AdaptiveMaxPool2d((1, 1))
        elif pool == 'avg':
            model_ft.avgpool2 = torch.nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            model_ft.maxpool2 = torch.nn.AdaptiveMaxPool2d((1, 1))
        self.model = model_ft

        if init_model is not None:
            self.model = init_model.model
            self.pool = init_model.pool
        # self.fc = torch.nn.Linear(in_features=1000, out_features=512)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1, x2), dim=1)
            x = x.view(x.size(0), -1)
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
            x = x.view(x.size(0), -1)
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
            x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x


class simple_resnet_18(nn.Module):
    def __init__(self, num_classes, droprate=0.5, stride=2, pool='avg'):
        super(simple_resnet_18, self).__init__()
        self.model_net = ft_net_resnet18(num_classes, stride=stride, pool=pool)
        self.classifier = ClassBlock(512, num_classes, droprate)

    def forward(self, x, x2=None):
        x = self.model_net(x)
        y = self.classifier(x)
        if x2 is None:
            return y
        else:
            x2 = self.model_net(x2)
            y2 = self.classifier(x2)
            return y, y2


class view_net_152(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(view_net_152, self).__init__()
        model_ft = torchvision.models.resnet152(pretrained=True)
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)
        # model_ft.fc = torch.nn.Linear(1000, class_num)
        # self.pool = pool
        model_ft.avgpool2 = torch.nn.AdaptiveAvgPool2d((1, 1))
        model_ft.newfc = torch.nn.Linear(2048, class_num)
        self.model = model_ft

        # self.fc = torch.nn.Linear(in_features=1000, out_features=512)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool2(x)
        x = x.view(x.size(0), -1)
        x = self.model.newfc(x)
        return x


class net_vgg16(torch.nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2):
        super(net_vgg16, self).__init__()
        model_ft = torchvision.models.vgg16_bn(pretrained=True)
        model_ft.avgpool2 = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool2(x)
        x = x.view(x.size(0), -1)
        return x


class net_vgg19(torch.nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2):
        super(net_vgg19, self).__init__()
        model_ft = torchvision.models.vgg19_bn(pretrained=True)
        model_ft.avgpool2 = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool2(x)
        x = x.view(x.size(0), -1)
        return x


class net_resnet101(torch.nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2):
        super(net_resnet101, self).__init__()
        model_ft = torchvision.models.resnet101(pretrained=True)
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)
        model_ft.avgpool2 = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool2(x)
        x = x.view(x.size(0), -1)
        return x


class net_resnet152(torch.nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2):
        super(net_resnet152, self).__init__()
        self.class_num = class_num
        model_ft = torchvision.models.resnet152(pretrained=True)
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)
        model_ft.avgpool2 = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool2(x)
        x = x.view(x.size(0), -1)
        return x


class ori_net(torch.nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, VGG16=False, VGG19=False, RESNET101=False, RESNET152=True):
        super(ori_net, self).__init__()
        if VGG19:
            self.model_net = net_vgg19(class_num, stride=stride)
        elif RESNET152:
            self.model_net = net_resnet152(class_num, stride=stride)
        elif RESNET101:
            self.model_net = net_resnet101(class_num, stride=stride)
        elif VGG16:
            self.model_net = net_vgg16(class_num, stride=stride)

        if VGG16 or VGG19:
            self.classifier = OriBlock(512, class_num, droprate)
        else:
            self.classifier = OriBlock(2048, class_num, droprate)

    def forward(self, x, x2=None):
        x = self.model_net(x)
        y = self.classifier(x)
        if x2 is None:
            return y
        else:
            x2 = self.model_net(x2)
            y2 = self.classifier(x2)
            return y, y2


class OriBlock(torch.nn.Module):

    def __init__(self, input_dim, class_num, droprate, relu=True, bnorm=False, num_bottleneck=512, linear=True,
                 return_feature=False):
        super(OriBlock, self).__init__()
        self.return_feature = return_feature
        add_block = []
        add_block += [torch.nn.Linear(input_dim, num_bottleneck)]
        add_block = torch.nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [torch.nn.Linear(num_bottleneck, class_num)]
        classifier = torch.nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_feature:
            x_feature = x
            x = self.classifier(x)
            return x, x_feature
        else:
            x = self.classifier(x)
            return x


def save_network_teacher(network, dirname, epoch_label):
    if not os.path.isdir('./model/teacher' + dirname):
        os.mkdir('./model/teacher' + dirname)
    if isinstance(epoch_label, int):
        save_filename = 'net_%03d.pth' % epoch_label
    else:
        save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model/teacher', dirname, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda()


def count_parameters_in_MB(model):
    return sum(np.prod(v.size()) for name, v in model.named_parameters()) / 1e6


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, std=0.001)
        torch.nn.init.constant_(m.bias.data, 0.0)


'''
    好像是加了一个ClassBlock
    input_dim: 输入维度
    class_num: 
    droprate: 
    relu: 
    bnorm: 
    num_bottleneck: bottleneck意思是瓶颈层，核心目标是为了改变维度，减少参数量，这里感觉像是根据需求，决定后续操作的维度
    linear: 是否为线性判别器
    return_feature: 
'''


class ClassBlock(torch.nn.Module):

    def __init__(self, input_dim, class_num, droprate, relu=True, bnorm=False, num_bottleneck=512, linear=True,
                 return_feature=False):
        super(ClassBlock, self).__init__()
        self.return_feature = return_feature
        add_block = []
        if linear:
            add_block += [torch.nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [torch.nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [torch.nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [torch.nn.Dropout(p=droprate)]
        add_block = torch.nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [torch.nn.Linear(num_bottleneck, class_num)]
        classifier = torch.nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_feature:
            x_feature = x
            x = self.classifier(x)
            return x, x_feature
        else:
            x = self.classifier(x)
            return x


'''
    在原有模型上修改了池化方式
    droprate=0.5: 在每次训练的时候，随机让一半的特征检测器停过工作，这样可以提高网络的泛化能力
    stride=2: 卷积步长
    init_model:
    pool: 池化方式
'''


class ft_net_vgg19(torch.nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net_vgg19, self).__init__()
        model_ft = torchvision.models.vgg19_bn(pretrained=True)

        self.pool = pool
        if pool == 'avg+max':
            model_ft.avgpool2 = torch.nn.AdaptiveAvgPool2d((1, 1))
            model_ft.maxpool2 = torch.nn.AdaptiveMaxPool2d((1, 1))
        elif pool == 'avg':
            model_ft.avgpool2 = torch.nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            model_ft.maxpool2 = torch.nn.AdaptiveMaxPool2d((1, 1))
        self.model = model_ft

        if init_model is not None:
            self.model = init_model.model
            self.pool = init_model.pool

    def forward(self, x):
        x = self.model.features(x)
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1, x2), dim=1)
            x = x.view(x.size(0), -1)
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
            x = x.view(x.size(0), -1)
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
            x = x.view(x.size(0), -1)
        return x


class ft_net_vgg16(torch.nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net_vgg16, self).__init__()
        model_ft = torchvision.models.vgg16_bn(pretrained=True)

        self.pool = pool
        if pool == 'avg+max':
            model_ft.avgpool2 = torch.nn.AdaptiveAvgPool2d((1, 1))
            model_ft.maxpool2 = torch.nn.AdaptiveMaxPool2d((1, 1))
        elif pool == 'avg':
            model_ft.avgpool2 = torch.nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            model_ft.maxpool2 = torch.nn.AdaptiveMaxPool2d((1, 1))
        self.model = model_ft

        if init_model is not None:
            self.model = init_model.model
            self.pool = init_model.pool

    def forward(self, x):
        x = self.model.features(x)
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1, x2), dim=1)
            x = x.view(x.size(0), -1)
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
            x = x.view(x.size(0), -1)
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
            x = x.view(x.size(0), -1)
        return x


class ft_net_resnet101(torch.nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net_resnet101, self).__init__()
        model_ft = torchvision.models.resnet101(pretrained=True)
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)

        self.pool = pool
        if pool == 'avg+max':
            model_ft.avgpool2 = torch.nn.AdaptiveAvgPool2d((1, 1))
            model_ft.maxpool2 = torch.nn.AdaptiveMaxPool2d((1, 1))
        elif pool == 'avg':
            model_ft.avgpool2 = torch.nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            model_ft.maxpool2 = torch.nn.AdaptiveMaxPool2d((1, 1))
        self.model = model_ft

        if init_model is not None:
            self.model = init_model.model
            self.pool = init_model.pool
        # self.fc = torch.nn.Linear(in_features=1000, out_features=512)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1, x2), dim=1)
            x = x.view(x.size(0), -1)
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
            x = x.view(x.size(0), -1)
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
            x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x


class ft_net_resnet152(torch.nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net_resnet152, self).__init__()
        model_ft = torchvision.models.resnet152(pretrained=True)
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)

        self.pool = pool
        if pool == 'avg+max':
            model_ft.avgpool2 = torch.nn.AdaptiveAvgPool2d((1, 1))
            model_ft.maxpool2 = torch.nn.AdaptiveMaxPool2d((1, 1))
        elif pool == 'avg':
            model_ft.avgpool2 = torch.nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            model_ft.maxpool2 = torch.nn.AdaptiveMaxPool2d((1, 1))
        self.model = model_ft

        if init_model is not None:
            self.model = init_model.model
            self.pool = init_model.pool
        # self.fc = torch.nn.Linear(in_features=1000, out_features=512)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1, x2), dim=1)
            x = x.view(x.size(0), -1)
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
            x = x.view(x.size(0), -1)
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
            x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x


'''
    droprate=0.5: 在每次训练的时候，随机让一半的特征检测器停过工作，这样可以提高网络的泛化能力
    stride=2: 卷积步长
    init_model:
    pool: 池化方式
    VGG19: 是否为VGG19网络
'''


class view_net(torch.nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, pool='avg', share_weight=False, VGG19=False, RESNET152=True,
                 RESNET101=False, VGG16=False):
        super(view_net, self).__init__()
        if VGG19:
            self.model_net = ft_net_vgg19(class_num, stride=stride, pool=pool)
        elif RESNET152:
            self.model_net = ft_net_resnet152(class_num, stride=stride, pool=pool)
        elif RESNET101:
            self.model_net = ft_net_resnet101(class_num, stride=stride, pool=pool)
        elif VGG16:
            self.model_net = ft_net_vgg16(class_num, stride=stride, pool=pool)

        if VGG16:
            self.classifier = ClassBlock(512, class_num, droprate)
        else:
            self.classifier = ClassBlock(2048, class_num, droprate)
        if pool == 'avg+max' and VGG19:
            self.classifier = ClassBlock(512, class_num, droprate)
            self.classifier = ClassBlock(1024, class_num, droprate)
        elif pool == 'avg+max':
            self.classifier = ClassBlock(4096, class_num, droprate)
        elif VGG19:
            self.classifier = ClassBlock(512, class_num, droprate)

    def forward(self, x, x2=None):
        x = self.model_net(x)
        y = self.classifier(x)
        if x2 is None:
            return y
        else:
            x2 = self.model_net(x2)
            y2 = self.classifier(x2)
            return y, y2


class simple_CNN(nn.Module):
    def __init__(self, num_classes, out_channels_simple=256, droprate=0.5, stride=2, pool='avg'):
        super(simple_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels_simple),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(in_channels=out_channels_simple, out_channels=out_channels_simple * 4, kernel_size=3,
        #               stride=stride, padding=1),
        #     nn.BatchNorm2d(out_channels_simple * 4),
        #     nn.LeakyReLU(0.1),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        # )
        # self.fc = nn.Linear(32 * (image_size // 4) * (image_size // 4), num_classes)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # print("output_size=", self.avgpool.output_size)
        # self.fc = nn.Linear(out_channels_simple*4, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = ClassBlock(256, num_classes, droprate=droprate)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        # return x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y


class simple_2CNN(nn.Module):
    def __init__(self, num_classes, out_channels_simple=256, droprate=0.5, stride=2, pool='avg'):
        super(simple_2CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=stride, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels_simple),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = ClassBlock(256, num_classes, droprate=droprate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y


class simple_3CNN(nn.Module):
    def __init__(self, num_classes, out_channels_simple=256, droprate=0.5, stride=2, pool='avg'):
        super(simple_3CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=stride, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=stride, padding=1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = ClassBlock(256, num_classes, droprate=droprate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y


class simple_res(nn.Module):
    def __init__(self, num_classes, out_channels_simple=256, droprate=0.5, stride=2, pool='avg'):
        super(simple_res, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=stride, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=stride, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=stride, padding=1),
            # nn.BatchNorm2d(1024),
            # nn.LeakyReLU(0.1),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=stride, padding=1),
            # nn.BatchNorm2d(512),
            # nn.LeakyReLU(0.1),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=stride, padding=1),
            # nn.BatchNorm2d(256),
            # nn.LeakyReLU(0.1),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = ClassBlock(256, num_classes, droprate=droprate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total = %s, Trainable = %s" % (total_num, trainable_num))
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    '''
        原有分类器删掉，用relu替换batchNorm
        RESNET-152
    '''
    '''
    # net = simple_resnet_18(num_classes=21, droprate=0.5, stride=2)
    # net = view_net(21, droprate=0.5, share_weight=False, VGG19=False, RESNET152=False, RESNET18=True)
    # net = view_net(21, droprate=0.5, share_weight=False, VGG19=False, RESNET152=False, RESNET101=True)
    # net = view_net(21, droprate=0.5, share_weight=False, VGG19=False, RESNET152=True, RESNET18=False)
    # net = view_net(21, droprate=0.5, share_weight=False, VGG16=True, RESNET152=False)
    # net = ft_net_resnet152(21, stride=2, pool='avg')
    # net = ft_net_vgg19(21, stride=2, pool='avg')
    # net = ft_net_resnet101(21, stride=2, pool='avg')

    # net = simple_CNN(num_classes=21, out_channels_simple=256, droprate=0.5, stride=2)
    # net = simple_2CNN(num_classes=21, out_channels_simple=256, droprate=0.5, stride=2)
    # net = simple_3CNN(num_classes=21, out_channels_simple=256, droprate=0.5, stride=2)

    # net = view_net_152(21, droprate=0.5)
    # net = simple_resnet_18(21, droprate=0.5)
    # net = ori_net(class_num=21, droprate=0.5, stride=2, VGG16=False, VGG19=False, RESNET101=False, RESNET152=True)
    

    # net = ori_net(class_num=21, droprate=0.5, stride=2, VGG16=False, VGG19=False, RESNET101=True, RESNET152=False)
    # print(net)
    # print("param size = %f MB" % count_parameters_in_MB(net))

    # net = ori_net(class_num=21, droprate=0.5, stride=2, VGG16=False, VGG19=True, RESNET101=False, RESNET152=False)
    # print(net)
    # print("param size = %f MB" % count_parameters_in_MB(net))

    # net = ori_net(class_num=21, droprate=0.5, stride=2, VGG16=True, VGG19=False, RESNET101=False, RESNET152=False)
    # print(net)
    # print("param size = %f MB" % count_parameters_in_MB(net))

    # print('# generator parameters:', sum(param.numel() for param in net.parameters()))
    # get_parameter_number(net)

    # simple_CNN: 0.150037MB; simple_2CNN: 0.441621MB; simple_3CNN: 1.031701MB
    # RESNET18: 11.962941MB; RESNET152: 61.252669MB; VGG19: 143.951677MB
    # print("-------------------------------------------------------")

    # input_tensor = torch.autograd.Variable(torch.FloatTensor(8, 3, 16, 16))
    # 构造一个2*3*4*4的张量矩阵，矩阵元素维度为4*4，每3个4*4的矩阵为一个维度，这样3*4*4的维度有2个
    # print(input_tensor)
    # print("-------------------------------------------------------")

    # save_network_teacher(net, 'view', 15)
    # print(output_tensor.shape)

    # m = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    # q = torch.randn(1,2,5,5)
    # o = m(q)
    # print(o.shape)
    '''

    net = simple_2CNN(num_classes=21, out_channels_simple=256, droprate=0.5, stride=2)
    # net = ResNet(block=BasicBlock, num_blocks=[1, 1, 1, 1], num_classes=256)
    print(net)
    print("param size = %f MB" % count_parameters_in_MB(net))

    input_tensor = torch.autograd.Variable(torch.FloatTensor(8, 3, 384, 384))
    print('net input_tensor size:', input_tensor.shape)
    output_tensor = net(input_tensor)
    print('net output_tensor size:', output_tensor.shape)
