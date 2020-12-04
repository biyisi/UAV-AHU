import os
import torch
import yaml
from model import view_net

# Get model list for resume
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
def load_network(name, opt):
    # Load config
    dirname = os.path.join('./model', name)
    last_model_name = os.path.basename(get_model_list(dirname, 'net'))
    epoch = last_model_name.split('_')[1]
    epoch = epoch.split('.')[0]
    if not epoch == 'last':
        epoch = int(epoch)
    config_path = os.path.join(dirname, 'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream)

    opt.name = config['name']
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

    # if opt.use_dense:
    #     model = ft_net_dense(opt.nclasses, opt.droprate, opt.stride, None, opt.pool)
    # if opt.PCB:
    #     model = PCB(opt.nclasses)

    model = view_net(opt.nclasses, opt.droprate, stride=opt.stride, pool=opt.pool, share_weight=opt.share, VGG19=False, RESNET152=True)

    # if 'use_vgg19' in config:
    #     opt.use_vgg19 = config['use_vgg19']
    #     model = view_net(opt.nclasses, opt.droprate, stride=opt.stride, pool=opt.pool, share_weight=opt.share,
    #                              VGG19=opt.use_vgg19)

    # load model
    if isinstance(epoch, int):
        save_filename = 'net_%03d.pth' % epoch
    else:
        save_filename = 'net_%s.pth' % epoch

    # save_filename = 'net_094.pth'
    save_path = os.path.join('./model', name, save_filename)
    print('Load the model from %s' % save_path)
    network = model
    network.load_state_dict(torch.load(save_path))
    return network, opt, epoch


def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


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
def save_network(network, dirname, epoch_label):
    if not os.path.isdir('./model/' + dirname):
        os.mkdir('./model/' + dirname)
    if isinstance(epoch_label, int):
        save_filename = 'net_%03d.pth' % epoch_label
    else:
        save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', dirname, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda()
