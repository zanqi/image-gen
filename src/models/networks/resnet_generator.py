import torch.nn as nn

class ResnetGenerator(nn.Module):
    """
    Resnet-based generator
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d):
        super().__init__()

    def forward(self, input_x):
        pass
