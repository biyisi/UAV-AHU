import pretrainedmodels
import torchvision
import torch
import torch.nn as nn


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
    def __init__(self, class_num, droprate=0.5, stride=2, pool='avg', share_weight=False, VGG19=False, RESNET152=True):
        super(view_net, self).__init__()
        if VGG19:
            self.model_net = ft_net_vgg19(class_num, stride=stride, pool=pool)
        elif RESNET152:
            self.model_net = ft_net_resnet152(class_num, stride=stride, pool=pool)

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


if __name__ == '__main__':
    '''
        原有分类器删掉，用relu替换batchNorm
        RESNET-152
    '''
    net = view_net(21, droprate=0.5, share_weight=False, VGG19=False, RESNET152=True)
    # net = view_net(21, droprate=0.5, share_weight=False, VGG19=True, RESNET152=False)
    # net = view_net(751, droprate=0.5, VGG19=True, RESNET152=False)
    print(net)

    # print("-------------------------------------------------------")
    input_tensor = torch.autograd.Variable(torch.FloatTensor(8, 3, 256, 256))
    # input_tensor = torch.autograd.Variable(torch.FloatTensor(2, 3, 4, 4))
    '''构造一个2*3*4*4的张量矩阵，矩阵元素维度为4*4，每3个4*4的矩阵为一个维度，这样3*4*4的维度有2个'''
    # print(input_tensor)
    # print("-------------------------------------------------------")
    output_tensor = net(input_tensor)
    print('net output_tensor size:')
    print(output_tensor.shape)
