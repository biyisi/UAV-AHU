import pretrainedmodels
import torchvision
import torch
import torch.nn as nn
import utils


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


class ft_net_resnet18(torch.nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net_resnet18, self).__init__()
        model_ft = torchvision.models.resnet18(pretrained=True)
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
                 RESNET18=False):
        super(view_net, self).__init__()
        if VGG19:
            self.model_net = ft_net_vgg19(class_num, stride=stride, pool=pool)
        elif RESNET152:
            self.model_net = ft_net_resnet152(class_num, stride=stride, pool=pool)
        elif RESNET18:
            self.model_net = ft_net_resnet18(class_num, stride=stride, pool=pool)
        if RESNET18 == True:
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
    def __init__(self, num_classes, out_channels_simple=64, droprate=0.5, stride=2, pool='avg'):
        super(simple_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=out_channels_simple, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels_simple),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels_simple, out_channels=out_channels_simple*4, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels_simple*4),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # self.fc = nn.Linear(32 * (image_size // 4) * (image_size // 4), num_classes)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # print("output_size=", self.avgpool.output_size)
        # self.fc = nn.Linear(out_channels_simple*4, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = ClassBlock(out_channels_simple*4, num_classes, droprate=droprate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        # return x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y


if __name__ == '__main__':
    '''
        原有分类器删掉，用relu替换batchNorm
        RESNET-152
    '''
    # net = view_net(21, droprate=0.5, share_weight=False, VGG19=False, RESNET152=False, RESNET18=True)
    # net = view_net(21, droprate=0.5, share_weight=False, VGG19=True, RESNET152=False, RESNET18=False)
    # net = view_net(21, droprate=0.5, share_weight=False, VGG19=False, RESNET152=True, RESNET18=False)
    # net = view_net(21, droprate=0.5, share_weight=False, VGG19=True, RESNET152=False)

    net = simple_CNN(num_classes=21, out_channels_simple=128, droprate=0.5, stride=2)
    print(net)
    print("param size = %f MB" % utils.count_parameters_in_MB(net))
    # RESNET18: 11.962941MB; RESNET152: 61.252669MB; VGG19: 143.951677MB
    # print("-------------------------------------------------------")
    input_tensor = torch.autograd.Variable(torch.FloatTensor(8, 3, 512, 512))
    # input_tensor = torch.autograd.Variable(torch.FloatTensor(8, 3, 16, 16))
    '''构造一个2*3*4*4的张量矩阵，矩阵元素维度为4*4，每3个4*4的矩阵为一个维度，这样3*4*4的维度有2个'''
    # print(input_tensor)
    # print("-------------------------------------------------------")
    output_tensor = net(input_tensor)
    print('net output_tensor size:', output_tensor.shape)
    # print(output_tensor.shape)


    # m = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    # q = torch.randn(1,2,5,5)
    # o = m(q)
    # print(o.shape)

