from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from model import view_net, simple_CNN, simple_2CNN, simple_3CNN, simple_res, simple_resnet_18

import os
import torch
import yaml
import shutil
import numpy as np


class Logits(torch.nn.Module):
    def __init__(self):
        super(Logits, self).__init__()

    def forward(self, out_student, out_teacher):
        loss = torch.nn.functional.mse_loss(out_student, out_teacher)
        return loss


class SoftTarget(torch.nn.Module):
    '''
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    '''

    def __init__(self, T):
        super(SoftTarget, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(out_s / self.T, dim=1),
                        torch.nn.functional.softmax(out_t / self.T, dim=1),
                        reduction='batchmean') * self.T * self.T

        return loss


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 统计模型的参数大小
def count_parameters_in_MB(model):
    return sum(np.prod(v.size()) for name, v in model.named_parameters()) / 1e6


def create_exp_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))


def save_checkpoint(state, save_root='./model/baseline', is_best=False):
    '''
        :param state: 保存的参数内容，以字典形式
        :param is_best: 是否为最佳结果
        :param save_root: 保存的文件目录
    '''
    save_path = os.path.join(save_root, 'checkpoint.pth.tar')
    torch.save(state, save_path)
    if is_best:
        best_save_path = os.path.join(save_root, 'model_best.pth.tar')
        shutil.copyfile(save_path, best_save_path)


def accuracy_k(output: torch, label, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # topk=(1,5)，那么就是找出最大的前5个，返回tensor值和index下标
    # topk=(1,5)表示为一个二元组，一个1，一个5
    maxk = max(topk)
    batch_size = label.size(0)
    '''
        求tensor中某个dim的前k大或者前k小的值以及对应的index
        torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) 
                    -> (Tensor, LongTensor)
        input：一个tensor数据
        k：指明是得到前k个数据以及其index
        dim： 指定在哪个维度上排序， 默认是最后一个维度
        largest：如果为True，按照大到小排序； 如果为False，按照小到大排序
        sorted：返回的结果按照顺序返回
        out：可缺省，不要
    '''
    _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
    # torch.t(): 矩阵转置
    pred = pred.t()
    '''
        torch.eq(self, other: Number) -> Tensor:
        torch.eq(input, other, out=None): 比较元素是否相等
        label是torch.Size([batch_size]), view(1, -1)从[128]变化为[1, 128]
        torch.expand_as(self, other: Tensor) -> Tensor: 将原有tensor复制扩展为指定维度tensor
        correct是一个pred同类型的张量，每一个位置如果相等则为True，否则为False
    '''
    # correct = pred.eq(label.view(1, -1).expand_as(pred))
    correct = torch.eq(pred, label.view(1, -1).expand_as(pred))

    res = []
    # k=1; k=5;
    for k in topk:
        # correct[:k]: 第一维取到k，然后进行降维处理, 转换成float, sum进行累加
        correct_k = correct[:k].view(-1).float().sum(0)
        # 计算出acc@1 和 acc@5，返回一个0-100的数字表示百分之多少
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def load_pretrained_model(model, pretrained_dict):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


def transform_time(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return h, m, s


# Get model list for resume, 找到最后的一个模型的名称，用于继续训练或者推理
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        print('no dir: %s' % dirname)
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pth" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


######################################################################
#  Load model for resume
# ---------------------------
def config_to_opt(config, opt):
    opt.out_model_name = config['out_model_name']
    opt.data_dir = config['data_dir']
    opt.train_all = config['train_all']
    opt.droprate = config['droprate']
    opt.color_jitter = config['color_jitter']
    opt.batchsize = config['batchsize']
    opt.h = config['h']
    opt.w = config['w']
    opt.share = config['share']
    opt.stride = config['stride']
    if 'pool' in config:
        opt.pool = config['pool']
    if 'h' in config:
        opt.h = config['h']
        opt.w = config['w']
    if 'gpu_ids' in config:
        opt.gpu_ids = config['gpu_ids']
    opt.erasing_p = config['erasing_p']
    opt.lr = config['lr']
    opt.nclasses = config['nclasses']
    opt.erasing_p = config['erasing_p']
    opt.use_dense = config['use_dense']
    opt.fp16 = config['fp16']
    opt.views = config['views']
    return opt


def load_network_teacher(out_model_name, opt, RESNET152=True, RESNET101=False, VGG19=False, VGG16=False):
    # Load config, 获取最后的epoch和网络名称
    dirname = os.path.join('./model/teacher', out_model_name)
    last_model_name = os.path.basename(get_model_list(dirname, 'net'))
    epoch = last_model_name.split('_')[1]
    epoch = epoch.split('.')[0]
    if not epoch == 'last':
        epoch = int(epoch)
    config_path = os.path.join(dirname, 'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream)

    opt = config_to_opt(config, opt)

    # if opt.use_dense:
    #     model = ft_net_dense(opt.nclasses, opt.droprate, opt.stride, None, opt.pool)
    # if opt.PCB:
    #     model = PCB(opt.nclasses)
    if RESNET152 == True:
        model = view_net(opt.nclasses, opt.droprate, stride=opt.stride, pool=opt.pool, share_weight=opt.share,
                         VGG19=False,
                         RESNET152=True, RESNET101=False, VGG16=False)
    elif RESNET101 == True:
        model = view_net(opt.nclasses, opt.droprate, stride=opt.stride, pool=opt.pool, share_weight=opt.share,
                         VGG19=False,
                         RESNET152=False, RESNET101=True, VGG16=False)
    elif VGG16 == True:
        model = view_net(opt.nclasses, opt.droprate, stride=opt.stride, pool=opt.pool, share_weight=opt.share,
                         VGG19=False,
                         RESNET152=False, RESNET101=False, VGG16=True)
    else:
        model = view_net(opt.nclasses, opt.droprate, stride=opt.stride, pool=opt.pool, share_weight=opt.share,
                         VGG19=True,
                         RESNET152=False, RESNET101=False, VGG16=False)

    # if 'use_vgg19' in config:
    #     opt.use_vgg19 = config['use_vgg19']
    #     model = view_net(opt.nclasses, opt.droprate, stride=opt.stride, pool=opt.pool, share_weight=opt.share,
    #                              VGG19=opt.use_vgg19)

    # load model
    if isinstance(epoch, int):
        save_filename = 'net_%03d.pth' % epoch
    else:
        save_filename = 'net_%s.pth' % epoch

    save_filename = 'net_050.pth'

    save_path = os.path.join('./model/teacher', out_model_name, save_filename)
    print('Load the model from %s' % save_path)
    network = model
    network.load_state_dict(torch.load(save_path))
    return network, opt, epoch


def load_teacher_infer_model(opt, RESNET18, RESNET152, VGG19):
    model, _, epoch = load_network_teacher('view', opt, RESNET18=RESNET18, RESNET152=RESNET152, VGG19=VGG19)
    return model


def load_student_infer_model(opt):
    model, _, epoch = load_network_student(opt.name, opt)
    return model


def load_network_student(out_model_name, opt):
    # Load config, 获取最后的epoch和网络名称
    dirname = os.path.join('./model/student', out_model_name)
    last_model_name = os.path.basename(get_model_list(dirname, 'net'))
    epoch = last_model_name.split('_')[1]
    epoch = epoch.split('.')[0]
    if not epoch == 'last':
        epoch = int(epoch)
    config_path = os.path.join(dirname, 'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream)

    opt = config_to_opt(config, opt)

    # TODO: 修改加载模型
    model = simple_2CNN(num_classes=opt.nclasses, droprate=opt.droprate, stride=opt.stride, pool=opt.pool)

    # load model
    if isinstance(epoch, int):
        save_filename = 'net_%03d.pth' % epoch
    else:
        save_filename = 'net_%s.pth' % epoch

    # save_filename = 'net_000.pth'
    # save_filename = 'net_400.pth'
    # save_filename = 'net_600.pth'
    # save_filename = 'net_800.pth'
    # save_filename = 'net_900.pth'

    save_path = os.path.join('./model/student', out_model_name, save_filename)
    print('Load the model from %s' % save_path)
    network = model
    network.load_state_dict(torch.load(save_path))
    return network, opt, epoch


# 基本没用上
def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


# 基本没用上
def update_average(model_tgt, model_src, beta):
    toogle_grad(model_src, False)
    toogle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

    toogle_grad(model_src, True)


######################################################################
# Save model
# ---------------------------
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


def save_network_student(network, dirname, epoch_label):
    if not os.path.isdir('./model/student' + dirname):
        os.mkdir('./model/student' + dirname)
    if isinstance(epoch_label, int):
        save_filename = 'net_%03d.pth' % epoch_label
    else:
        save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model/student', dirname, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda()


def save_network_kd(network, dirname, epoch_label):
    if not os.path.isdir('./model/kd' + dirname):
        os.mkdir('./model/kd' + dirname)
    if isinstance(epoch_label, int):
        save_filename = 'net_%03d.pth' % epoch_label
    else:
        save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model/kd', dirname, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda()
