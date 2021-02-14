from .UnetGenerator import UnetGenerator
from .GANLoss import GANLoss
import functools
import torch.nn as nn
from torch.nn import init

def defineG(input_nc, output_nc, ngf, norm='batch', use_dropout=False):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    return init_net(net, 'normal', 0.02)

def defineD():
    pass

def get_norm_layer(norm_type):
    """
    docstring
    """
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    return norm_layer

def init_net(net, init_type='normal', init_gain=0.02):
    """
    docstring
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and ('Conv' in classname) or ('Linear' in classname):
            init.normal(m, 0, init_gain)
        elif 'BatchNorm2d' in classname:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    
    net.apply(init_func)
    return net