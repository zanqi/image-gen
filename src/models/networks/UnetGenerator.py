import torch.nn as nn
from .UnetSkipConnectionBlock import UnetSkipConnectionBlock


class UnetGenerator(nn.Module):
    """
    docstring
    """

    def __init__(self, input_nc, output_nc, num_downs, ngf, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None, 
            submodule=None, norm_layer=norm_layer, 
            is_innermost=True,
            is_outermost=False,
            use_dropout=False)  # add the innermost layer
