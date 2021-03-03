import functools
import torch
import torch.nn as nn
from torch.nn import init
from .UnetGenerator import UnetGenerator
from .n_layer_discriminator import NLayerDiscriminator


def define_g(input_nc, output_nc, ngf,
            use_dropout=False, gpu_ids=None):
    net = None
    norm_layer = get_norm_layer()
    net = UnetGenerator(input_nc, output_nc, 7, ngf,
                        norm_layer=norm_layer, use_dropout=use_dropout)
    return init_net(net, 0.02, gpu_ids)


def define_d(input_nc, ndf, n_layers=3, gpu_ids=None):
    net = None
    norm_layer = get_norm_layer()
    net = NLayerDiscriminator(
        input_nc, ndf, n_layers=n_layers, norm_layer=norm_layer)
    return init_net(net, 0.02, gpu_ids)


def get_norm_layer():
    """
    docstring
    """
    norm_layer = functools.partial(
        nn.BatchNorm2d, affine=True, track_running_stats=True)
    return norm_layer


def init_net(net, init_gain=0.02, gpu_ids=None):
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs

    def init_func(module):
        classname = module.__class__.__name__
        if hasattr(module, 'weight') \
                and ('Conv' in classname) \
                or ('Linear' in classname):
            init.normal_(module.weight.data, 0, init_gain)
        elif 'BatchNorm2d' in classname:
            init.normal_(module.weight.data, 1.0, init_gain)
            init.constant_(module.bias.data, 0.0)

    net.apply(init_func)
    return net
